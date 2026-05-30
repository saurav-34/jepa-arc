"""Train a linear state head on top of a frozen LeWM encoder.

Two embedding modes (--emb-mode):

  encoder   (original)
    frozen ViT → projector → 192-dim z → head → state_t
    Trains fast, but mismatches inference: at game time the head receives
    pred_proj(predictor(...)), not projector(encoder(...)).

  predictor (default, recommended)
    For each 3-frame context window [t, t+1, t+2] with actions [a_t, a_{t+1}, a_{t+2}]:
      z_pred = pred_proj(predictor([z_t, z_{t+1}, z_{t+2}], [a_t, a_{t+1}, a_{t+2}]))[:, -1]
    Head trained: z_pred → state_{t+3}
    Matches exactly what lepong.py uses at inference.

Usage:
  python scripts/train/train_statehead.py --dataset ~/.stable-wm/datasets/pong_with_state.lance
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import lance
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from stable_worldmodel.wm.utils import load_pretrained

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="lewm")
parser.add_argument("--dataset", type=str, default="datasets/pong_with_state.lance")
parser.add_argument("--emb-mode", type=str, default="predictor",
                    choices=["encoder", "predictor"],
                    help="Which embedding space to train on. 'predictor' matches inference.")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--train-split", type=float, default=0.9)
parser.add_argument("--encoder-cache", type=str, default=None)
parser.add_argument("--pred-cache", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

IMG_SIZE   = 224
EMBED_DIM  = 192
STATE_DIM  = 6
STATE_COLS = ['ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'player_y', 'opp_y']
HISTORY    = 3   # predictor context length

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Stage 1a: encoder embeddings ────────────────────────────────────────────

def precompute_encoder_embeddings(model, dataset_path, cache_path, device, batch_size=64):
    print(f"Pre-computing encoder embeddings → {cache_path}")
    ds = lance.dataset(dataset_path)
    n  = ds.count_rows()

    model.encoder.eval().to(device)
    model.projector.eval().to(device)

    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        tbl  = ds.take(list(range(start, end)), columns=['pixels'])
        frames = []
        for jpeg_bytes in tbl['pixels'].to_pylist():
            img = Image.open(__import__('io').BytesIO(bytes(jpeg_bytes))).resize((IMG_SIZE, IMG_SIZE))
            frames.append(np.array(img, dtype=np.float32))
        pixels = np.stack(frames).transpose(0, 3, 1, 2) / 255.0
        pixels = (pixels - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]

        with torch.no_grad():
            x   = torch.from_numpy(pixels).to(device, dtype=next(model.encoder.parameters()).dtype)
            cls = model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            emb = model.projector(cls).float().cpu().numpy()

        embeddings[start:end] = emb
        if (start // batch_size) % 20 == 0:
            print(f"  {end}/{n} frames embedded")

    np.save(cache_path, embeddings)
    print(f"Encoder embeddings saved → {cache_path}")
    return embeddings


# ─── Stage 1b: predictor embeddings ──────────────────────────────────────────

def precompute_predictor_embeddings(model, dataset_path, encoder_embs,
                                    cache_path, device, batch_size=256):
    """For every valid 3-frame context window within an episode, run the predictor
    and store pred_proj(predictor(z_ctx, a_ctx))[:, -1].

    Returns (pred_embs, target_states) where each row i corresponds to a window
    [t, t+1, t+2] and the target is state_{t+3}.
    """
    print(f"Pre-computing predictor embeddings → {cache_path}")

    ds  = lance.dataset(dataset_path)
    tbl = ds.to_table(columns=['episode_idx', 'step_idx', 'action'] + STATE_COLS)

    episode_idxs = tbl['episode_idx'].to_pylist()
    step_idxs    = tbl['step_idx'].to_pylist()
    actions      = tbl['action'].to_pylist()          # already int32
    states       = np.stack([tbl[c].to_pylist() for c in STATE_COLS], axis=1).astype(np.float32)

    # group flat indices by episode, sorted by step
    ep_to_flat = defaultdict(list)
    for flat_i, (ep, step) in enumerate(zip(episode_idxs, step_idxs)):
        ep_to_flat[ep].append((step, flat_i))
    for ep in ep_to_flat:
        ep_to_flat[ep].sort()

    # collect all valid windows: (ctx_i0, ctx_i1, ctx_i2, tgt_i, a0, a1, a2)
    windows = []
    for ep, frames in ep_to_flat.items():
        idxs = [f[1] for f in frames]   # flat indices in step order
        for i in range(len(idxs) - HISTORY):   # need HISTORY context + 1 target
            i0, i1, i2, i3 = idxs[i], idxs[i+1], idxs[i+2], idxs[i+3]
            windows.append((i0, i1, i2, i3,
                            int(actions[i0]), int(actions[i1]), int(actions[i2])))

    print(f"  {len(windows)} valid windows from {ds.count_rows()} frames")

    model.predictor.eval().to(device)
    model.pred_proj.eval().to(device)
    model.action_encoder.eval().to(device)
    enc_dtype = next(model.predictor.parameters()).dtype

    pred_embs    = np.zeros((len(windows), EMBED_DIM), dtype=np.float32)
    target_states = np.zeros((len(windows), STATE_DIM), dtype=np.float32)

    for start in range(0, len(windows), batch_size):
        end   = min(start + batch_size, len(windows))
        batch = windows[start:end]

        z_ctx = np.stack([
            [encoder_embs[w[0]], encoder_embs[w[1]], encoder_embs[w[2]]]
            for w in batch
        ])  # (B, 3, 192)

        acts = [[w[4], w[5], w[6]] for w in batch]   # (B, 3)

        with torch.no_grad():
            z_t    = torch.from_numpy(z_ctx).to(device, dtype=enc_dtype)    # (B, 3, 192)
            a_t    = torch.tensor(acts, dtype=torch.long, device=device)     # (B, 3)
            a_emb  = model.action_encoder(a_t)                               # (B, 3, 192)
            preds  = model.predictor(z_t, a_emb)                             # (B, 3, 192)
            z_pred = model.pred_proj(preds[:, -1])                           # (B, 192)
            z_pred = z_pred.float().cpu().numpy()

        pred_embs[start:end]     = z_pred
        target_states[start:end] = np.stack([states[w[3]] for w in batch])

        if (start // batch_size) % 20 == 0:
            print(f"  {end}/{len(windows)} windows processed")

    np.save(cache_path, pred_embs)
    print(f"Predictor embeddings saved → {cache_path}")
    return pred_embs, target_states


# ─── Dataset ─────────────────────────────────────────────────────────────────

class StateHeadDataset(Dataset):
    def __init__(self, embeddings, states, state_mean, state_std):
        self.embeddings = torch.from_numpy(embeddings)
        states_norm = (states - state_mean) / state_std
        self.states = torch.from_numpy(states_norm.astype(np.float32))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.states[idx]


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model '{args.model}' ...")
    model = load_pretrained(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    dataset_path   = args.dataset
    encoder_cache  = args.encoder_cache or dataset_path.replace('.lance', '_encoder_embs.npy')
    pred_cache     = args.pred_cache    or dataset_path.replace('.lance', '_pred_embs.npy')

    # ── encoder embeddings (always needed: predictor mode uses them as input) ──
    if os.path.exists(encoder_cache):
        print(f"Loading cached encoder embeddings from {encoder_cache}")
        encoder_embs = np.load(encoder_cache)
    else:
        encoder_embs = precompute_encoder_embeddings(
            model, dataset_path, encoder_cache, args.device)

    # ── choose embedding mode ──
    if args.emb_mode == "predictor":
        print("\n── Predictor embedding mode (matches inference) ──")
        if os.path.exists(pred_cache):
            print(f"Loading cached predictor embeddings from {pred_cache}")
            embeddings = np.load(pred_cache)
            # reconstruct target states for matching windows
            ds  = lance.dataset(dataset_path)
            tbl = ds.to_table(columns=['episode_idx', 'step_idx', 'action'] + STATE_COLS)
            episode_idxs = tbl['episode_idx'].to_pylist()
            step_idxs    = tbl['step_idx'].to_pylist()
            all_states   = np.stack([tbl[c].to_pylist() for c in STATE_COLS], axis=1).astype(np.float32)
            ep_to_flat = defaultdict(list)
            for i, (ep, step) in enumerate(zip(episode_idxs, step_idxs)):
                ep_to_flat[ep].append((step, i))
            for ep in ep_to_flat:
                ep_to_flat[ep].sort()
            target_idxs = []
            for ep, frames in ep_to_flat.items():
                idxs = [f[1] for f in frames]
                for i in range(len(idxs) - HISTORY):
                    target_idxs.append(idxs[i + HISTORY])
            states = all_states[target_idxs]
        else:
            embeddings, states = precompute_predictor_embeddings(
                model, dataset_path, encoder_embs, pred_cache, args.device)
    else:
        print("\n── Encoder embedding mode (original) ──")
        embeddings = encoder_embs
        ds     = lance.dataset(dataset_path)
        tbl    = ds.to_table(columns=STATE_COLS)
        states = np.stack([tbl[c].to_pylist() for c in STATE_COLS], axis=1).astype(np.float32)

    state_mean = states.mean(0)
    state_std  = states.std(0) + 1e-8
    print(f"State mean: {state_mean}")
    print(f"State std:  {state_std}")

    # save normalization stats
    from stable_worldmodel.data.utils import get_cache_dir
    cache_dir = get_cache_dir(sub_folder='checkpoints') / 'statehead'
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "state_mean.npy", state_mean)
    np.save(cache_dir / "state_std.npy",  state_std)

    # dataloaders
    dataset  = StateHeadDataset(embeddings, states, state_mean, state_std)
    n_train  = int(len(dataset) * args.train_split)
    n_val    = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # state head
    state_head = nn.Linear(EMBED_DIM, STATE_DIM).to(args.device)
    optimizer  = torch.optim.AdamW(state_head.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining state head on {args.emb_mode} embeddings "
          f"({sum(p.numel() for p in state_head.parameters())} params) "
          f"for {args.epochs} epochs on {args.device}")
    print(f"Train: {n_train}  Val: {n_val}\n")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        state_head.train()
        train_loss = 0.0
        for emb, target in train_loader:
            emb, target = emb.to(args.device), target.to(args.device)
            loss = nn.functional.mse_loss(state_head(emb), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(emb)
        train_loss /= n_train

        state_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for emb, target in val_loader:
                emb, target = emb.to(args.device), target.to(args.device)
                val_loss += nn.functional.mse_loss(state_head(emb), target).item() * len(emb)
        val_loss /= n_val

        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state_head.state_dict(), cache_dir / "statehead.pt")
            print(f"           ↑ saved (best val={best_val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")

    # R² per state variable
    state_head.load_state_dict(torch.load(cache_dir / "statehead.pt"))
    state_head.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for emb, target in val_loader:
            all_preds.append(state_head(emb.to(args.device)).cpu().numpy())
            all_targets.append(target.numpy())
    all_preds   = np.concatenate(all_preds)   * state_std + state_mean
    all_targets = np.concatenate(all_targets) * state_std + state_mean

    print(f"\nR² per state variable ({args.emb_mode} embeddings):")
    for i, col in enumerate(STATE_COLS):
        ss_res = ((all_targets[:, i] - all_preds[:, i]) ** 2).sum()
        ss_tot = ((all_targets[:, i] - all_targets[:, i].mean()) ** 2).sum()
        print(f"  {col:12s}: R² = {1 - ss_res / ss_tot:.4f}")

    print(f"\nState head saved → {cache_dir}/statehead.pt")
    print(f"Norm stats saved → {cache_dir}/state_{{mean,std}}.npy")


if __name__ == '__main__':
    main()
