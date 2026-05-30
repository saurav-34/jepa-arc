"""LePong — play Pong driven by the trained JEPA world model.

Usage:
    python scripts/play/lepong.py [--wm PATH] [--statehead PATH] [--K INT]

Checkpoint paths default to $STABLEWM_HOME/... (or ~/.stable_worldmodel/...).
Pass --wm to point at a Lightning .ckpt or a weights .pt file.
Pass --statehead to point at the statehead folder or statehead.pt directly.
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--wm', type=str, default=None,
                    help='Path to world-model weights (.pt or Lightning .ckpt). '
                         'Falls back to load_pretrained("lewm").')
parser.add_argument('--statehead', type=str, default=None,
                    help='Path to statehead folder or statehead.pt. '
                         'Falls back to $STABLEWM_HOME/checkpoints/statehead/.')
parser.add_argument('--K', type=int, default=3,
                    help='Re-anchor every K steps (1 = every step). Default 3.')
parser.add_argument('--fps', type=int, default=30,
                    help='Target frames per second. Default 30.')
parser.add_argument('--scale', type=float, default=3.0,
                    help='Scale factor from ALE coords to screen pixels. Default 3.')
parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

# ──────────────────────────────────────────────────────────────
# Constants matching training
# ──────────────────────────────────────────────────────────────

IMG_SIZE     = 224
EMBED_DIM    = 192
STATE_DIM    = 6
STATE_COLS   = ['ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'player_y', 'opp_y']
HISTORY_SIZE = 3

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ALE Pong geometry (approximate)
ALE_W, ALE_H  = 160, 210
PADDLE_H_ALE  = 15   # paddle height in ALE pixels
BALL_R_ALE    = 2
PLAYER_X_ALE  = 140  # right paddle x
OPP_X_ALE     = 16   # left paddle x

# ALE action indices
NOOP = 0
UP   = 2
DOWN = 3

# ──────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────

def _stablewm_home() -> Path:
    raw = os.getenv('STABLEWM_HOME', '~/.stable_worldmodel')
    return Path(os.path.expanduser(raw))


def _latest_checkpoint(folder: Path) -> Path:
    """Return the highest-epoch .pt file in a checkpoint folder."""
    pts = sorted(folder.glob('*.pt'))
    if not pts:
        raise FileNotFoundError(f'No .pt files in {folder}')
    def _epoch(p: Path) -> int:
        try:
            return int(p.stem.rsplit('_', 1)[-1])
        except ValueError:
            return 0
    return max(pts, key=_epoch)


def load_world_model(path_override: str | None, device: str):
    """Load LeWM — accepts a Lightning .ckpt, a plain .pt, or auto-detects from
    $STABLEWM_HOME/checkpoints/lewm/."""
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from stable_worldmodel.wm.utils import load_pretrained

    if path_override is None:
        default_dir = _stablewm_home() / 'checkpoints' / 'lewm'
        if default_dir.is_dir():
            pt = _latest_checkpoint(default_dir)
            print(f'Loading world model: {pt}')
            model = load_pretrained(str(pt))
        else:
            print('Loading world model via load_pretrained("lewm")')
            model = load_pretrained('lewm')
    else:
        p = Path(path_override).expanduser()
        if p.suffix == '.ckpt':
            model = _load_from_lightning_ckpt(p)
            return model.to(device).eval()
        else:
            model = load_pretrained(str(p))

    return model.to(device).eval()


def _load_from_lightning_ckpt(ckpt_path: Path):
    """Load a LeWM from a raw Lightning checkpoint."""
    import json
    from hydra.utils import instantiate

    ckpt = torch.load(ckpt_path, map_location='cpu')

    raw = ckpt.get('state_dict', ckpt)
    stripped = {k.removeprefix('model.'): v for k, v in raw.items()}

    cfg_path = ckpt_path.parent / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(
            f'config.json not found next to {ckpt_path}.\n'
            f'Either place a config.json there or use a weights.pt checkpoint.'
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = instantiate(cfg.get('model', cfg))
    model.load_state_dict(stripped)
    return model


def load_state_head(path_override: str | None, device: str):
    """Load statehead.pt + normalization stats."""
    if path_override is not None:
        p = Path(path_override)
        folder = p if p.is_dir() else p.parent
    else:
        folder = _stablewm_home() / 'checkpoints' / 'statehead'

    pt   = folder / 'statehead.pt'
    mean = folder / 'state_mean.npy'
    std  = folder / 'state_std.npy'

    missing = [f for f in (pt, mean, std) if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f'Missing statehead files in {folder}:\n' +
            '\n'.join(f'  {f}' for f in missing)
        )

    head = nn.Linear(EMBED_DIM, STATE_DIM)
    head.load_state_dict(torch.load(pt, map_location='cpu'))
    head = head.to(device).eval()

    print(f'State head loaded from {folder}')
    return head, np.load(mean), np.load(std)


# ──────────────────────────────────────────────────────────────
# Frame preprocessing / encoding
# ──────────────────────────────────────────────────────────────

def preprocess_frame(rgb_uint8: np.ndarray) -> torch.Tensor:
    """ALE RGB frame → normalised (1, 3, 224, 224) float tensor."""
    img = Image.fromarray(rgb_uint8).resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(x).unsqueeze(0)


@torch.no_grad()
def encode_frame(model, frame_tensor: torch.Tensor, device: str) -> torch.Tensor:
    """frame_tensor: (1, 3, 224, 224) → z: (192,)"""
    x   = frame_tensor.to(device, dtype=next(model.encoder.parameters()).dtype)
    cls = model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    z   = model.projector(cls)
    return z.squeeze(0).float()


@torch.no_grad()
def predict_next(model, z_history: torch.Tensor,
                 action_history: list[int], device: str) -> torch.Tensor:
    """Predict next latent given history.

    z_history:      (T, 192) — last T encoded frames
    action_history: list of T ints — actions taken at each history step
    Returns z_next: (192,)
    """
    emb     = z_history.unsqueeze(0).to(device)                                    # (1, T, 192)
    act_idx = torch.tensor([action_history], dtype=torch.long, device=device)      # (1, T)
    act_emb = model.action_encoder(act_idx)                                        # (1, T, 192)
    preds   = model.predictor(emb, act_emb)                                        # (1, T, 192)
    z_next  = model.pred_proj(preds[:, -1])                                        # (1, 192)
    return z_next.squeeze(0).float()


@torch.no_grad()
def decode_state(head, z: torch.Tensor, state_mean, state_std, device: str) -> dict:
    """z: (192,) → dict of ALE-coordinate state values."""
    raw  = head(z.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
    vals = raw * state_std + state_mean
    return dict(zip(STATE_COLS, vals.tolist()))


# ──────────────────────────────────────────────────────────────
# Seeding helper
# ──────────────────────────────────────────────────────────────

def collect_seed(model, env, device: str):
    """Step ALE three times with NOOP and return initial z_history and action_history."""
    z_list = []
    a_list = []
    for _ in range(HISTORY_SIZE):
        frame, _, term, trunc, _ = env.step(NOOP)
        if term or trunc:
            env.reset()
        z_list.append(encode_frame(model, preprocess_frame(frame), device))
        a_list.append(NOOP)
    return torch.stack(z_list), a_list   # (3, 192), [int, int, int]


# ──────────────────────────────────────────────────────────────
# Pygame renderer
# ──────────────────────────────────────────────────────────────

def make_renderer(scale: float):
    import pygame
    pygame.init()

    W = int(ALE_W * scale)
    H = int(ALE_H * scale)
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('LePong — JEPA world model')
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', int(14 * scale / 3))

    BG    = (0,   0,   0)
    WHITE = (255, 255, 255)
    GREEN = (0,   200, 60)
    RED   = (200, 50,  50)

    def render(state: dict, score: tuple[int, int], step: int,
               K: int, anchored: bool):
        screen.fill(BG)

        bx = int(state['ball_x'] * scale)
        by = int(state['ball_y'] * scale)
        pygame.draw.circle(screen, WHITE, (bx, by), max(2, int(BALL_R_ALE * scale)))

        pw = max(4, int(4 * scale))
        ph = int(PADDLE_H_ALE * scale)

        py = int(state['player_y'] * scale)
        pygame.draw.rect(screen, GREEN, (int(PLAYER_X_ALE * scale), py, pw, ph))

        oy = int(state['opp_y'] * scale)
        pygame.draw.rect(screen, RED, (int(OPP_X_ALE * scale), oy, pw, ph))

        net_w = int(5 * scale)
        for yy in range(0, H, int(10 * scale)):
            pygame.draw.rect(screen, (60, 60, 60), (W // 2 - 1, yy, 2, net_w))

        anchor_tag = '[ALE]' if anchored else '[WM] '
        hud = (f'Opp {score[0]} : {score[1]} You   K={K}  step={step}  {anchor_tag}')
        screen.blit(font.render(hud, True, (180, 180, 180)), (4, 4))
        screen.blit(font.render('↑/↓ move    Q quit    +/- change K',
                                True, (100, 100, 100)),
                    (4, H - int(18 * scale / 3) - 2))

        pygame.display.flip()

    return screen, clock, render


# ──────────────────────────────────────────────────────────────
# Score detection
# ──────────────────────────────────────────────────────────────

def check_score(state: dict, prev_score: tuple[int, int],
                prev_ball_x: float) -> tuple[tuple[int, int], bool]:
    opp, you = prev_score
    bx       = state['ball_x']
    scored   = False
    if bx < 5 and prev_ball_x >= 5:
        you   += 1
        scored = True
    if bx > 155 and prev_ball_x <= 155:
        opp   += 1
        scored = True
    return (opp, you), scored


# ──────────────────────────────────────────────────────────────
# Main game loop
# ──────────────────────────────────────────────────────────────

def main():
    import pygame

    print('Loading world model...')
    model = load_world_model(args.wm, args.device)
    print('Loading state head...')
    head, state_mean, state_std = load_state_head(args.statehead, args.device)

    print('Initialising ALE (Pong)...')
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make('ALE/Pong-v5', render_mode='rgb_array', obs_type='rgb')
    env.reset(seed=42)

    print('Collecting seed frames...')
    z_history, action_history = collect_seed(model, env, args.device)
    state = decode_state(head, z_history[-1], state_mean, state_std, args.device)

    _, clock, render = make_renderer(args.scale)

    K           = args.K
    score       = (0, 0)
    step        = 0
    prev_ball_x = state['ball_x']
    running     = True

    print(f'\nGame started! K={K}  device={args.device}')
    print('Controls: ↑/↓ = move paddle, Q = quit, +/- = change K')

    while running:
        # ── input ──
        action = NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    K = min(K + 1, 20)
                    print(f'K → {K}')
                if event.key == pygame.K_MINUS:
                    K = max(K - 1, 1)
                    print(f'K → {K}')
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = UP
        elif keys[pygame.K_DOWN]:
            action = DOWN

        # ── world model step ──
        z_next = predict_next(model, z_history, action_history, args.device)
        state  = decode_state(head, z_next, state_mean, state_std, args.device)

        # ── ALE step (every frame to stay in sync) ──
        ale_frame, _, terminated, truncated, _ = env.step(action)

        # ── slide history windows ──
        z_history      = torch.cat([z_history[1:], z_next.unsqueeze(0)], dim=0)
        action_history = action_history[1:] + [action]

        # ── episode reset ──
        anchored = False
        if terminated or truncated:
            env.reset()
            z_history, action_history = collect_seed(model, env, args.device)
            state = decode_state(head, z_history[-1], state_mean, state_std, args.device)
            anchored = True
        elif step % K == 0:
            # ── re-anchor: replace latest z with real encoded frame ──
            z_anchor  = encode_frame(model, preprocess_frame(ale_frame), args.device)
            z_history = torch.cat([z_history[:-1], z_anchor.unsqueeze(0)], dim=0)
            state     = decode_state(head, z_anchor, state_mean, state_std, args.device)
            anchored  = True

        # ── score & render ──
        score, _ = check_score(state, score, prev_ball_x)
        prev_ball_x = state['ball_x']

        render(state, score, step, K, anchored)
        clock.tick(args.fps)
        step += 1

    env.close()
    pygame.quit()
    print(f'Final score — Opponent: {score[0]}  You: {score[1]}')


if __name__ == '__main__':
    main()
