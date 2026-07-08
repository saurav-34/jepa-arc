"""Linear state-probe for a trained LeWM on Tennis.

Answers one question: do the embeddings the planner uses actually encode the
game state (ball / paddle positions)? We freeze the world model, extract
embeddings, and fit a *linear* head to ground-truth coordinates. High R² means
the representation is informative (planning can work); low R² means the pretty
loss curves came from an encoder-predictor shortcut, not real dynamics.

Two embedding spaces are probed and reported side by side:

  encoder    z = projector(encoder(frame)[CLS])            -> state_t
             "does the encoder even see the ball?"
  predictor  z = pred_proj(predictor(z_ctx, a_ctx))[:, -1] -> state_{t+H}
             "does the space the PLANNER rolls out in encode the ball?"
             (this is the one that matters for the game)

Tennis differs from Pong: paddles move in 2D, so the state is 8-dim
(ball x/y/vx/vy + player x/y + enemy x/y) and frames are 128px.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/eval/probe_tennis_state.py \
      --wm lewm_tennis/weights_epoch_3.pt \
      --dataset datasets/tennis_ma_128x128_400k.lance
"""

import argparse
import io
from collections import defaultdict

import lance
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from stable_worldmodel.wm.utils import load_pretrained

# ── Tennis constants (2D paddles, 128px) ──────────────────────────────────────
IMG_SIZE   = 128
EMBED_DIM  = 192
HISTORY    = 3
STATE_COLS = ['ball_x', 'ball_y', 'ball_vx', 'ball_vy',
              'player_x', 'player_y', 'enemy_x', 'enemy_y']
STATE_DIM  = len(STATE_COLS)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', type=str, default='lewm_tennis/weights_epoch_3.pt',
                   help='Checkpoint path resolved under $STABLEWM_HOME/checkpoints/. '
                        'Can be a "run/weights_epoch_N.pt" or an absolute .pt path.')
    p.add_argument('--dataset', type=str,
                   default='datasets/tennis_ma_128x128_400k.lance')
    p.add_argument('--max-rows', type=int, default=120000,
                   help='Cap frames for a fast estimate (0 = use all).')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--train-split', type=float, default=0.9)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_frames(ds, idxs):
    """Decode a list of flat row indices into a normalized (N,3,H,W) tensor."""
    tbl = ds.take(idxs, columns=['pixels'])
    frames = []
    for jpeg in tbl['pixels'].to_pylist():
        img = Image.open(io.BytesIO(bytes(jpeg))).convert('RGB').resize(
            (IMG_SIZE, IMG_SIZE))
        frames.append(np.asarray(img, dtype=np.float32))
    px = np.stack(frames).transpose(0, 3, 1, 2) / 255.0
    px = (px - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    return px


@torch.no_grad()
def compute_encoder_embeddings(model, ds, n, device, batch_size=256):
    """z = projector(encoder(frame)[CLS]) for every frame 0..n-1."""
    model.encoder.eval().to(device)
    model.projector.eval().to(device)
    dtype = next(model.encoder.parameters()).dtype
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        px = load_frames(ds, list(range(start, end)))
        x = torch.from_numpy(px).to(device, dtype=dtype)
        cls = model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        embs[start:end] = model.projector(cls).float().cpu().numpy()
        if (start // batch_size) % 25 == 0:
            print(f'  encoder {end}/{n}')
    return embs


def build_windows(ds, n):
    """Group rows by episode and return H-step context windows.

    Returns list of (i0,i1,i2, tgt_i, a0,a1,a2) in step order per episode.
    """
    tbl = ds.take(list(range(n)),
                  columns=['episode_idx', 'step_idx', 'action'] + STATE_COLS)
    ep = tbl['episode_idx'].to_pylist()
    step = tbl['step_idx'].to_pylist()
    action = [int(a) for a in tbl['action'].to_pylist()]
    states = np.stack([tbl[c].to_pylist() for c in STATE_COLS], axis=1).astype(np.float32)

    by_ep = defaultdict(list)
    for i, (e, s) in enumerate(zip(ep, step)):
        by_ep[e].append((s, i))
    for e in by_ep:
        by_ep[e].sort()

    windows = []
    for e, frames in by_ep.items():
        idxs = [f[1] for f in frames]
        for i in range(len(idxs) - HISTORY):
            i0, i1, i2, i3 = idxs[i:i + HISTORY + 1]
            windows.append((i0, i1, i2, i3, action[i0], action[i1], action[i2]))
    return windows, states


@torch.no_grad()
def compute_predictor_embeddings(model, enc_embs, windows, device, batch_size=512):
    """z = pred_proj(predictor(z_ctx, a_ctx))[:, -1] per window; target = state_{t+H}."""
    model.predictor.eval().to(device)
    model.pred_proj.eval().to(device)
    model.action_encoder.eval().to(device)
    dtype = next(model.predictor.parameters()).dtype
    preds = np.zeros((len(windows), EMBED_DIM), dtype=np.float32)
    for start in range(0, len(windows), batch_size):
        end = min(start + batch_size, len(windows))
        batch = windows[start:end]
        z_ctx = np.stack([[enc_embs[w[0]], enc_embs[w[1]], enc_embs[w[2]]] for w in batch])
        acts = [[w[4], w[5], w[6]] for w in batch]
        z_t = torch.from_numpy(z_ctx).to(device, dtype=dtype)
        a_t = torch.tensor(acts, dtype=torch.long, device=device)
        a_emb = model.action_encoder(a_t)
        out = model.predictor(z_t, a_emb)
        z_pred = model.pred_proj(out[:, -1]).float().cpu().numpy()
        preds[start:end] = z_pred
        if (start // batch_size) % 25 == 0:
            print(f'  predictor {end}/{len(windows)}')
    return preds


class ProbeDS(Dataset):
    def __init__(self, embs, states, mean, std):
        self.x = torch.from_numpy(embs)
        self.y = torch.from_numpy(((states - mean) / std).astype(np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def fit_probe(embs, states, args, tag):
    """Train a linear head embs->states, return per-variable R² on val split."""
    mean, std = states.mean(0), states.std(0) + 1e-8
    ds = ProbeDS(embs, states, mean, std)
    n_tr = int(len(ds) * args.train_split)
    tr, va = random_split(ds, [n_tr, len(ds) - n_tr],
                          generator=torch.Generator().manual_seed(42))
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False)

    head = nn.Linear(EMBED_DIM, STATE_DIM).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float('inf')
    best_state = None
    for ep in range(1, args.epochs + 1):
        head.train()
        for x, y in tl:
            x, y = x.to(args.device), y.to(args.device)
            loss = nn.functional.mse_loss(head(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        head.eval()
        vloss = 0.0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(args.device), y.to(args.device)
                vloss += nn.functional.mse_loss(head(x), y).item() * len(x)
        vloss /= len(va)
        if vloss < best:
            best = vloss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    P, T = [], []
    with torch.no_grad():
        for x, y in vl:
            P.append(head(x.to(args.device)).cpu().numpy())
            T.append(y.numpy())
    P = np.concatenate(P) * std + mean
    T = np.concatenate(T) * std + mean

    print(f'\n=== R² per variable [{tag}] (val, best={best:.4f}) ===')
    r2s = {}
    for i, c in enumerate(STATE_COLS):
        ss_res = ((T[:, i] - P[:, i]) ** 2).sum()
        ss_tot = ((T[:, i] - T[:, i].mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        r2s[c] = r2
        # also report MAE in raw coordinate units
        mae = np.abs(T[:, i] - P[:, i]).mean()
        print(f'  {c:10s}: R² = {r2:6.3f}   MAE = {mae:6.2f}')
    return r2s


def main():
    args = parse_args()
    print(f'Loading world model: {args.wm}')
    model = load_pretrained(args.wm)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    ds = lance.dataset(args.dataset)
    n = ds.count_rows()
    if args.max_rows and args.max_rows < n:
        n = args.max_rows
    print(f'Using {n} frames from {args.dataset}')

    print('\n[1/3] encoder embeddings ...')
    enc = compute_encoder_embeddings(model, ds, n, args.device)

    print('\n[2/3] predictor embeddings ...')
    windows, states = build_windows(ds, n)
    tgt_states = np.stack([states[w[3]] for w in windows])
    pred = compute_predictor_embeddings(model, enc, windows, args.device)

    # encoder-space targets: state at each frame (aligned to enc rows)
    enc_states = states  # states[i] is the state at frame i

    print('\n[3/3] fitting linear probes ...')
    r2_enc = fit_probe(enc, enc_states, args, 'ENCODER  z=proj(enc(frame))')
    r2_pred = fit_probe(pred, tgt_states, args, 'PREDICTOR z=pred_proj(predictor(...))  <-- planner space')

    print('\n────────── VERDICT ──────────')
    key = ['ball_x', 'ball_y']
    enc_ball = np.mean([r2_enc[k] for k in key])
    pred_ball = np.mean([r2_pred[k] for k in key])
    print(f'ball R²   encoder={enc_ball:.3f}   predictor={pred_ball:.3f}')
    if pred_ball > 0.7:
        print('→ Planner space encodes the ball well. Model is usable; build the planner.')
    elif enc_ball > 0.7:
        print('→ Encoder sees the ball but predictor space loses it. Predictor/roll-out is the problem.')
    else:
        print('→ Ball is NOT linearly decodable. Curves were a shortcut; fix architecture '
              '(stop-grad/EMA target or a reconstruction term).')


if __name__ == '__main__':
    main()
