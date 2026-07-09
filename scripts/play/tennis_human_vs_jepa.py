"""Realtime 1v1 Tennis — human (keyboard, bottom) vs JEPA (world model, top).

The real PettingZoo `tennis_v3` env is the referee and renderer (same as the
reference lepong: the neural net is the AI's *perception*, not the game engine).
Every control step the JEPA side sees only pixels: the current frame is encoded
by the frozen LeWM encoder, a linear state head decodes {player_x, ball_x, ...}
in RAM units, and a reactive controller chases the ball's x and swings when
aligned — the same policy family the training data was collected with, but
driven entirely by the world model's belief instead of RAM.

Usage:
    python scripts/play/tennis_human_vs_jepa.py \
        [--wm PATH] [--statehead DIR] [--fps 15] [--scale 4]

Controls:  ←/→ move   SPACE swing/serve   Q quit

Conventions (verified by wiggle-probe on this ROM):
  RAM[26]=`player_x` tracks the BOTTOM paddle, RAM[27]=`enemy_x` the TOP one —
  addresses are court-side-locked, not player-locked. The ROM swaps the agents'
  ends after odd-numbered games (real tennis rules, tracked via games won
  RAM[71]+[72]); we swap which agent the keyboard/JEPA drive in lockstep, so
  the human ALWAYS plays the bottom paddle and JEPA the top one.
  Control stride = 4 ALE frames (fs4), matching the training data stride.

Needs a display (run locally, or `ssh -X`; headless -> use --record).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--wm', type=str, default=None,
                    help='World-model weights (.pt). Default: highest-epoch file in '
                         '$STABLEWM_HOME/checkpoints/lewm_tennis/.')
parser.add_argument('--statehead', type=str, default=None,
                    help='Statehead folder. Default: $STABLEWM_HOME/checkpoints/statehead_tennis/.')
parser.add_argument('--fps', type=int, default=15,
                    help='Control steps per second. 15 * frameskip 4 = 60 ALE fps (real time).')
parser.add_argument('--frameskip', type=int, default=4,
                    help='ALE frames per control step. MUST match the data collection stride (4).')
parser.add_argument('--scale', type=int, default=4,
                    help='Window upscale of the 160x210 ALE frame.')
parser.add_argument('--deadzone', type=float, default=6.0,
                    help='|ball_x - player_x| (RAM px) within which JEPA swings instead of moving.')
parser.add_argument('--record', type=str, default=None,
                    help='Write an .mp4 of the match instead of/in addition to the window.')
parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

IMG_SIZE  = 128
EMBED_DIM = 192

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ALE full-action-set ids (same as collector)
NOOP, FIRE, RIGHT, LEFT = 0, 1, 3, 4
# env agent ids. Roles are anchored to court ENDS, not agents: the ROM swaps
# the agents' ends after odd games (real tennis rules), so which agent the
# keyboard/JEPA drives flips with it — human is always the BOTTOM paddle,
# JEPA always the TOP.
FIRST_0, SECOND_0 = 'first_0', 'second_0'

# ──────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────

def _stablewm_home() -> Path:
    return Path(os.path.expanduser(os.getenv('STABLEWM_HOME', '~/.stable_worldmodel')))


def _latest_pt(folder: Path) -> Path:
    pts = list(folder.glob('weights_epoch_*.pt'))
    if not pts:
        raise FileNotFoundError(f'no weights_epoch_*.pt in {folder}')
    return max(pts, key=lambda p: int(p.stem.rsplit('_', 1)[-1]))


def load_models():
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from stable_worldmodel.wm.utils import load_pretrained

    wm_path = Path(args.wm) if args.wm else _latest_pt(_stablewm_home() / 'checkpoints' / 'lewm_tennis')
    print(f'World model: {wm_path}')
    model = load_pretrained(str(wm_path)).to(args.device).eval()

    head_dir = Path(args.statehead) if args.statehead else _stablewm_home() / 'checkpoints' / 'statehead_tennis'
    with open(head_dir / 'state_cols.json') as f:
        cols = json.load(f)
    head = nn.Linear(EMBED_DIM, len(cols))
    head.load_state_dict(torch.load(head_dir / 'statehead.pt', map_location='cpu'))
    head = head.to(args.device).eval()
    mean = np.load(head_dir / 'state_mean.npy')
    std  = np.load(head_dir / 'state_std.npy')
    print(f'State head: {head_dir}  cols={cols}')
    return model, head, cols, mean, std


# ──────────────────────────────────────────────────────────────
# JEPA side: pixels -> belief -> action
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_belief(model, head, cols, mean, std, frame_rgb: np.ndarray) -> dict:
    """Raw ALE frame (210x160x3 uint8) -> decoded state dict (RAM units)."""
    # identical preprocessing to the collector (cv2 INTER_AREA to 128) + statehead trainer
    x = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32).transpose(2, 0, 1) / 255.0
    x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    t = torch.from_numpy(x).unsqueeze(0).to(args.device,
                                            dtype=next(model.encoder.parameters()).dtype)
    cls = model.encoder(t, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    z   = model.projector(cls).float()
    raw = head(z).squeeze(0).cpu().numpy() * std + mean
    return dict(zip(cols, raw.tolist()))


def jepa_policy(belief: dict, deadzone: float, own_col: str) -> int:
    """Reactive intercept: chase decoded ball_x with own paddle, swing when aligned.

    `own_col` is the decoded column of the paddle JEPA controls — always
    'enemy_x' (RAM[27], top court), since roles are anchored to court ends.
    """
    dx = belief['ball_x'] - belief[own_col]
    if abs(dx) <= deadzone:
        return FIRE
    return RIGHT if dx > 0 else LEFT


def games_total(ale) -> int:
    """Cumulative games won by both agents (RAM[71] + RAM[72])."""
    r = ale.getRAM()
    return int(r[71]) + int(r[72])


def first0_is_bottom(ale) -> bool:
    """Current end assignment. Agents change ends after odd-numbered cumulative
    games (verified by wiggle-probe: game 1 first_0=RAM[26]=bottom, games 2-3
    top, games 4-5 bottom, game 6 top, ...)."""
    return ((games_total(ale) + 1) // 2) % 2 == 0


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────

def main():
    model, head, cols, mean, std = load_models()

    from pettingzoo.atari import tennis_v3

    def _rom_dir():
        import ale_py
        return os.path.join(os.path.dirname(ale_py.__file__), 'roms')

    env = tennis_v3.parallel_env(render_mode='rgb_array', auto_rom_install_path=_rom_dir())
    env.reset(seed=int(time.time()) % 100000)
    ale = env.unwrapped.ale                      # referee info only (side assignment)
    frame = env.render()
    H, W = frame.shape[:2]                       # 210, 160

    import pygame
    try:
        pygame.init()
        screen = pygame.display.set_mode((W * args.scale, H * args.scale + 28))
        pygame.display.set_caption('Tennis — you (bottom) vs JEPA (top)')
    except pygame.error as e:
        sys.exit(f'pygame needs a display ({e}). Run locally / `ssh -X`, or use --record.')
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont('monospace', 16)

    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*'mp4v'),
                                 args.fps, (W * args.scale, H * args.scale))

    score = {'human': 0, 'jepa': 0}              # points won, by role
    belief, a_jepa = {}, NOOP
    step, t_infer = 0, 0.0
    running = True

    print('\nMatch on! You are the BOTTOM player.  ←/→ move, SPACE swing/serve, Q quit.')

    while running:
        # ── human input ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            a_human = FIRE
        elif keys[pygame.K_RIGHT]:
            a_human = RIGHT
        elif keys[pygame.K_LEFT]:
            a_human = LEFT
        else:
            a_human = NOOP

        # ── anchor roles to court ends: human=BOTTOM, JEPA=TOP, always ──
        # (the ROM swaps the agents' ends after odd games; we swap which agent
        # the keyboard/JEPA drive so each player keeps their side of the court)
        human_agent = FIRST_0 if first0_is_bottom(ale) else SECOND_0
        jepa_agent  = SECOND_0 if human_agent == FIRST_0 else FIRST_0
        jepa_col = 'enemy_x'                     # top-court paddle column

        # ── JEPA sees pixels, decodes belief, reacts ──
        t0 = time.perf_counter()
        belief = decode_belief(model, head, cols, mean, std, frame)
        a_jepa = jepa_policy(belief, args.deadzone, jepa_col)
        t_infer = (time.perf_counter() - t0) * 1000

        # ── advance the real game: hold both actions for `frameskip` ALE frames ──
        for _ in range(args.frameskip):
            if not env.agents:
                break
            _, rewards, _, _, _ = env.step({jepa_agent: a_jepa, human_agent: a_human})
            for ag, r in rewards.items():
                if r > 0:
                    score['human' if ag == human_agent else 'jepa'] += 1
        if not env.agents:                       # match over -> new match
            env.reset()
        frame = env.render()

        # ── draw ──
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (W * args.scale, H * args.scale))
        screen.blit(surf, (0, 0))
        act_name = {NOOP: '·', FIRE: 'SWING', RIGHT: '→', LEFT: '←'}.get(a_jepa, '?')
        hud = (f'You (BOTTOM) '
               f'{score["human"]} : {score["jepa"]} JEPA   '
               f'belief ball_x={belief.get("ball_x", 0):5.1f} '
               f'jepa_x={belief.get(jepa_col, 0):5.1f} act={act_name:5s} '
               f'{t_infer:4.1f}ms')
        screen.blit(font.render(hud, True, (230, 230, 230)),
                    (6, H * args.scale + 6))
        pygame.display.flip()

        if writer is not None:
            arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        clock.tick(args.fps)
        step += 1

    if writer is not None:
        writer.release()
        print(f'Recording saved to {args.record}')
    env.close()
    pygame.quit()
    print(f'Final — You {score["human"]} : {score["jepa"]} JEPA')


if __name__ == '__main__':
    main()
