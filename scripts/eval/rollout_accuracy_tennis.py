"""Autoregressive rollout accuracy for the Tennis LeWM — the real planning path.

The planner rolls the predictor forward feeding its own output back in. Errors
compound. We seed 3 real frames + real actions, roll forward T steps feeding the
REAL action sequence, decode ball/paddle state at each step with a frozen linear
head, and compare to ground truth vs horizon. Persistence (hold last observed
state) is the baseline the rollout must beat to be worth planning on.

Run: STABLEWM_HOME=... CUDA_VISIBLE_DEVICES=1 python scripts/eval/rollout_accuracy_tennis.py
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
VARS = ['player_x', 'ball_x', 'ball_y']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', default='lewm_tennis/weights_epoch_3.pt')
    p.add_argument('--dataset', default='datasets/tennis_ma_128x128_400k.lance')
    p.add_argument('--max-rows', type=int, default=60000)
    p.add_argument('--horizon', type=int, default=12)
    p.add_argument('--n-starts', type=int, default=4000)
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
def predict_last_np(model, z_ctx, acts, device, bs=1024):
    """pred_proj(predictor(z_ctx, act_emb))[:, -1] as numpy — the rollout embedding."""
    dt = next(model.predictor.parameters()).dtype
    out = np.zeros((len(z_ctx), EMBED_DIM), dtype=np.float32)
    for s in range(0, len(z_ctx), bs):
        e = min(s + bs, len(z_ctx))
        z = torch.from_numpy(z_ctx[s:e]).to(device, dtype=dt)
        a = torch.tensor(acts[s:e], dtype=torch.long, device=device)
        p = model.predictor(z, model.action_encoder(a))
        out[s:e] = model.pred_proj(p[:, -1]).float().cpu().numpy()
    return out


def fit_head(X, Y, lam=10.0):
    """Ridge X->Y with standardization; returns callable + reports val R²."""
    perm = np.random.RandomState(0).permutation(len(X))
    X, Y = X[perm], Y[perm]
    ntr = int(0.9 * len(X))
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xb = np.hstack([Xn, np.ones((len(Xn), 1), np.float32)])
    A = Xb[:ntr].T @ Xb[:ntr] + lam * np.eye(Xb.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xb[:ntr].T @ Y[:ntr])
    pred = Xb[ntr:] @ W
    ss_res = ((Y[ntr:] - pred) ** 2).sum(0)
    ss_tot = ((Y[ntr:] - Y[ntr:].mean(0)) ** 2).sum(0) + 1e-9
    r2 = 1 - ss_res / ss_tot
    print('  head R²:', {v: round(float(r2[i]), 3) for i, v in enumerate(VARS)})

    def decode(emb):  # emb: (M,D) -> (M,K) raw units
        en = (emb - mu) / sd
        return np.hstack([en, np.ones((len(en), 1), np.float32)]) @ W
    return decode


def main():
    args = parse_args()
    print(f'Loading {args.wm}')
    model = load_pretrained(args.wm); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.action_encoder.to(args.device); model.predictor.to(args.device)
    model.pred_proj.to(args.device)
    dt = next(model.predictor.parameters()).dtype

    ds = lance.dataset(args.dataset)
    n = min(args.max_rows, ds.count_rows())
    tbl = ds.take(list(range(n)),
                  columns=['episode_idx', 'step_idx', 'action'] + VARS)
    ep = np.array(tbl['episode_idx'].to_pylist())
    stp = np.array(tbl['step_idx'].to_pylist())
    act = np.array([int(a) for a in tbl['action'].to_pylist()])
    state = np.stack([np.array(tbl[v].to_pylist(), np.float32) for v in VARS], axis=1)

    print(f'Encoding {n} frames ...')
    enc = encoder_embs(model, ds, n, args.device)

    # Fit head on PREDICTOR-space embeddings (matched to the rollout): for each
    # 1-step window [i,i+1,i+2] predict state at i+3 from pred_proj(predictor(...)).
    print('Building 1-step windows for head fit (predictor space) ...')
    by_ep0 = defaultdict(list)
    for i, (e, s) in enumerate(zip(ep, stp)):
        by_ep0[e].append((s, i))
    for e in by_ep0:
        by_ep0[e].sort()
    ctx3, act3, tgt1 = [], [], []
    for e, frames in by_ep0.items():
        idxs = [f[1] for f in frames]
        for i in range(len(idxs) - HISTORY):
            ctx3.append((idxs[i], idxs[i + 1], idxs[i + 2]))
            act3.append((act[idxs[i]], act[idxs[i + 1]], act[idxs[i + 2]]))
            tgt1.append(idxs[i + HISTORY])
    ctx3 = np.array(ctx3); act3 = np.array(act3); tgt1 = np.array(tgt1)
    z_ctx_fit = np.stack([enc[ctx3[:, 0]], enc[ctx3[:, 1]], enc[ctx3[:, 2]]], axis=1)
    z_pred_fit = predict_last_np(model, z_ctx_fit, act3, args.device)
    print('Fitting state head (predictor space, matched to rollout) ...')
    decode = fit_head(z_pred_fit, state[tgt1])

    # contiguous episode runs long enough for HISTORY + horizon
    by_ep = defaultdict(list)
    for i, (e, s) in enumerate(zip(ep, stp)):
        by_ep[e].append((s, i))
    for e in by_ep:
        by_ep[e].sort()
    starts = []
    need = HISTORY + args.horizon
    for e, frames in by_ep.items():
        idxs = [f[1] for f in frames]
        for i in range(len(idxs) - need):
            # require truly consecutive steps
            if frames[i + need - 1][0] - frames[i][0] == need - 1:
                starts.append([idxs[i + k] for k in range(need)])
    starts = np.array(starts)
    if len(starts) > args.n_starts:
        sel = np.random.RandomState(1).choice(len(starts), args.n_starts, replace=False)
        starts = starts[sel]
    M = len(starts)
    print(f'{M} rollout starts, horizon {args.horizon}')

    # seed
    z_hist = torch.from_numpy(
        np.stack([enc[starts[:, 0]], enc[starts[:, 1]], enc[starts[:, 2]]], axis=1)
    ).to(args.device, dtype=dt)  # (M,3,D)
    a_hist = torch.tensor(
        np.stack([act[starts[:, 0]], act[starts[:, 1]], act[starts[:, 2]]], axis=1),
        dtype=torch.long, device=args.device)  # (M,3)

    last_seed_state = state[starts[:, 2]]  # (M,K) persistence baseline

    roll_mae = np.zeros((args.horizon, len(VARS)))
    pers_mae = np.zeros((args.horizon, len(VARS)))

    with torch.no_grad():
        for t in range(1, args.horizon + 1):
            a_emb = model.action_encoder(a_hist)
            p = model.predictor(z_hist, a_emb)
            z_next = model.pred_proj(p[:, -1])  # (M,D)
            pred_state = decode(z_next.float().cpu().numpy())
            gt = state[starts[:, HISTORY - 1 + t]]  # frame i+2+t
            roll_mae[t - 1] = np.abs(pred_state - gt).mean(0)
            pers_mae[t - 1] = np.abs(last_seed_state - gt).mean(0)
            # feed back: append predicted emb + REAL action at frame i+2+t
            z_hist = torch.cat([z_hist[:, 1:], z_next[:, None]], dim=1)
            a_next = torch.tensor(act[starts[:, HISTORY - 1 + t]],
                                  dtype=torch.long, device=args.device)
            a_hist = torch.cat([a_hist[:, 1:], a_next[:, None]], dim=1)

    print('\n=== Rollout MAE (px) vs horizon  [rollout | persistence] ===')
    hdr = '  '.join(f'{v:>18s}' for v in VARS)
    print(f'{"h":>3}  {hdr}')
    for t in range(args.horizon):
        cells = []
        for k in range(len(VARS)):
            cells.append(f'{roll_mae[t,k]:6.2f} | {pers_mae[t,k]:6.2f}')
        print(f'{t+1:>3}  ' + '  '.join(f'{c:>18s}' for c in cells))
    print('\nrollout should stay below persistence — that is the value of the model.')


if __name__ == '__main__':
    main()
