"""What horizon does the LeWM predictor actually encode, and is the paddle controllable?

Two questions that decide whether a planner can intercept the ball:

1. HORIZON: predict(ctx)[:, -1] was trained (num_preds=3) to hit a frame N steps
   after the last context frame. The game (lepong) uses it as 1-step. We fit a
   ridge head predictor_emb -> player_x/ball_x/ball_y at each candidate horizon h
   and see which h gives the best R². The peak = the model's true horizon.

2. CONTROLLABILITY: with the same context, swap the last action for a canonical
   RIGHT vs LEFT and see whether decoded player_x shifts in the correct direction.
   If it doesn't, no planner can steer the paddle.

Run: STABLEWM_HOME=... CUDA_VISIBLE_DEVICES=1 python scripts/eval/horizon_tennis.py
"""

import argparse
import io
from collections import defaultdict

import lance
import numpy as np
import torch
from PIL import Image

from stable_worldmodel.wm.utils import load_pretrained

IMG_SIZE, EMBED_DIM, HISTORY = 128, 192, 3
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RIGHT_ACT, LEFT_ACT, NOOP_ACT = 3, 4, 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', default='lewm_tennis/weights_epoch_3.pt')
    p.add_argument('--dataset', default='datasets/tennis_ma_128x128_400k.lance')
    p.add_argument('--max-rows', type=int, default=60000)
    p.add_argument('--max-horizon', type=int, default=6)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def load_frames(ds, idxs):
    tbl = ds.take(idxs, columns=['pixels'])
    fr = []
    for j in tbl['pixels'].to_pylist():
        img = Image.open(io.BytesIO(bytes(j))).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        fr.append(np.asarray(img, dtype=np.float32))
    px = np.stack(fr).transpose(0, 3, 1, 2) / 255.0
    return (px - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]


@torch.no_grad()
def encoder_embs(model, ds, n, device, bs=256):
    model.encoder.eval().to(device); model.projector.eval().to(device)
    dt = next(model.encoder.parameters()).dtype
    out = np.zeros((n, EMBED_DIM), dtype=np.float32)
    for s in range(0, n, bs):
        e = min(s + bs, n)
        x = torch.from_numpy(load_frames(ds, list(range(s, e)))).to(device, dtype=dt)
        cls = model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        out[s:e] = model.projector(cls).float().cpu().numpy()
        if (s // bs) % 40 == 0:
            print(f'  enc {e}/{n}')
    return out


@torch.no_grad()
def predict_last(model, z_ctx, acts, device, bs=1024):
    """pred_proj(predictor(z_ctx, act_emb))[:, -1] — the inference embedding."""
    model.predictor.eval().to(device); model.pred_proj.eval().to(device)
    model.action_encoder.eval().to(device)
    dt = next(model.predictor.parameters()).dtype
    out = np.zeros((len(z_ctx), EMBED_DIM), dtype=np.float32)
    for s in range(0, len(z_ctx), bs):
        e = min(s + bs, len(z_ctx))
        z = torch.from_numpy(z_ctx[s:e]).to(device, dtype=dt)
        a = torch.tensor(acts[s:e], dtype=torch.long, device=device)
        ae = model.action_encoder(a)
        p = model.predictor(z, ae)
        out[s:e] = model.pred_proj(p[:, -1]).float().cpu().numpy()
    return out


def ridge_r2(X, y, lam=10.0, split=0.9):
    """Closed-form ridge; return R² on held-out split (y can be (N,) or (N,K))."""
    n = len(X)
    perm = np.random.RandomState(42).permutation(n)  # shuffle: episodes are ordered
    X, y = X[perm], y[perm]
    ntr = int(n * split)
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1), np.float32)])
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ W
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0)) ** 2).sum(0) + 1e-9
    return 1 - ss_res / ss_tot


def main():
    args = parse_args()
    print(f'Loading {args.wm}')
    model = load_pretrained(args.wm)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = lance.dataset(args.dataset)
    n = min(args.max_rows, ds.count_rows())
    print(f'{n} frames')

    tbl = ds.take(list(range(n)),
                  columns=['episode_idx', 'step_idx', 'action', 'player_x', 'ball_x', 'ball_y'])
    ep = np.array(tbl['episode_idx'].to_pylist())
    stp = np.array(tbl['step_idx'].to_pylist())
    act = np.array([int(a) for a in tbl['action'].to_pylist()])
    px = np.array(tbl['player_x'].to_pylist(), dtype=np.float32)
    bx = np.array(tbl['ball_x'].to_pylist(), dtype=np.float32)
    by = np.array(tbl['ball_y'].to_pylist(), dtype=np.float32)

    print('Encoding frames ...')
    enc = encoder_embs(model, ds, n, args.device)

    # windows: context [i,i+1,i+2]; keep only those with i+2+maxH in same episode
    by_ep = defaultdict(list)
    for i, (e, s) in enumerate(zip(ep, stp)):
        by_ep[e].append((s, i))
    for e in by_ep:
        by_ep[e].sort()
    H = HISTORY
    maxH = args.max_horizon
    ctx_idx, ctx_act = [], []
    for e, frames in by_ep.items():
        idxs = [f[1] for f in frames]
        for i in range(len(idxs) - (H - 1) - maxH):
            ctx_idx.append((idxs[i], idxs[i + 1], idxs[i + 2]))
            ctx_act.append((act[idxs[i]], act[idxs[i + 1]], act[idxs[i + 2]]))
    ctx_idx = np.array(ctx_idx); ctx_act = np.array(ctx_act)
    print(f'{len(ctx_idx)} windows')

    z_ctx = np.stack([enc[ctx_idx[:, 0]], enc[ctx_idx[:, 1]], enc[ctx_idx[:, 2]]], axis=1)

    # ── Q1: horizon sweep with the REAL last action ──
    print('\nComputing predictor embeddings (real actions) ...')
    z_pred = predict_last(model, z_ctx, ctx_act, args.device)
    last_frame = ctx_idx[:, 2]  # frame index of last context frame
    print('\n=== HORIZON SWEEP: R² of predictor_emb -> state at (last_ctx + h) ===')
    print(f'{"h":>3} {"player_x":>9} {"ball_x":>9} {"ball_y":>9}')
    for h in range(1, maxH + 1):
        tgt = last_frame + h
        Y = np.stack([px[tgt], bx[tgt], by[tgt]], axis=1)
        r2 = ridge_r2(z_pred, Y)
        print(f'{h:>3} {r2[0]:9.3f} {r2[1]:9.3f} {r2[2]:9.3f}')
    print('(peak column per variable = the horizon that embedding actually encodes)')

    # ── Q2: controllability — swap last action, decode player_x ──
    # Train a player_x head at the model's apparent horizon (use h=3, num_preds).
    print('\n=== CONTROLLABILITY: swap last context action, decode player_x ===')
    # fit head on real-action preds -> player_x at h=num_preds(3) and h=1 for reference
    for h in (1, 3):
        tgt = last_frame + h
        # fit ridge head on real preds
        y = px[tgt].astype(np.float32)
        mu, sd = z_pred.mean(0), z_pred.std(0) + 1e-6
        Xn = (z_pred - mu) / sd
        Xb = np.hstack([Xn, np.ones((len(Xn), 1), np.float32)])
        W = np.linalg.solve(Xb.T @ Xb + 10 * np.eye(Xb.shape[1], dtype=np.float32), Xb.T @ y)

        def decode(acts):
            zp = predict_last(model, z_ctx, acts, args.device)
            zn = (zp - mu) / sd
            return (np.hstack([zn, np.ones((len(zn), 1), np.float32)]) @ W)

        a_right = ctx_act.copy(); a_right[:, 2] = RIGHT_ACT
        a_left = ctx_act.copy(); a_left[:, 2] = LEFT_ACT
        a_noop = ctx_act.copy(); a_noop[:, 2] = NOOP_ACT
        pr, pl, pn = decode(a_right), decode(a_left), decode(a_noop)
        print(f'h={h}: decoded player_x  RIGHT={pr.mean():6.2f}  NOOP={pn.mean():6.2f}  '
              f'LEFT={pl.mean():6.2f}   RIGHT-LEFT={pr.mean()-pl.mean():+.3f}px/step')
    print('(RIGHT-LEFT should be clearly positive if the model learned paddle control)')


if __name__ == '__main__':
    main()
