"""Train a linear state head on top of a frozen LeWM encoder.

Pipeline:
  frozen ViT encoder → 192-dim CLS embedding → linear head → 6-dim game state
  (ball_x, ball_y, ball_vx, ball_vy, player_y, opp_y)

Stage 1: Pre-compute and cache all encoder embeddings (runs once).
Stage 2: Train linear head on cached embeddings.

Usage:
  python scripts/train/train_statehead.py --model lewm --dataset datasets/pong_with_state.lance
"""

import argparse
import os
from pathlib import Path

import lance
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader, Dataset, random_split

from stable_worldmodel.wm.utils import load_pretrained, save_pretrained

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="lewm", help="checkpoint name passed to load_pretrained")
parser.add_argument("--dataset", type=str, default="datasets/pong_with_state.lance")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--train-split", type=float, default=0.9)
parser.add_argument("--embed-cache", type=str, default=None, help="path to cache embeddings (auto-derived if not set)")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

IMG_SIZE = 224
EMBED_DIM = 192
STATE_DIM = 6
STATE_COLS = ['ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'player_y', 'opp_y']

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Stage 1: pre-compute embeddings ────────────────────────────────────────

def precompute_embeddings(model, dataset_path, cache_path, device, batch_size=64):
    print(f"Pre-computing embeddings → {cache_path}")
    ds = lance.dataset(dataset_path)
    n = ds.count_rows()

    model.encoder.eval().to(device)

    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tbl = ds.take(list(range(start, end)), columns=['pixels'])
        pixels = np.array(tbl['pixels'].to_pylist(), dtype=np.uint8)
        pixels = pixels.reshape(end - start, IMG_SIZE, IMG_SIZE, 3)
        pixels = pixels.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        pixels = (pixels - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]

        with torch.no_grad():
            x = torch.from_numpy(pixels).to(device)
            out = model.encoder(x.to(next(model.encoder.parameters()).dtype),
                                interpolate_pos_encoding=True)
            emb = out.last_hidden_state[:, 0].float().cpu().numpy()

        embeddings[start:end] = emb

        if (start // batch_size) % 20 == 0:
            print(f"  {end}/{n} frames embedded")

    np.save(cache_path, embeddings)
    print(f"Embeddings saved to {cache_path}")
    return embeddings


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
    # load frozen world model
    print(f"Loading model '{args.model}'...")
    model = load_pretrained(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # pre-compute or load cached embeddings
    cache_path = args.embed_cache or args.dataset.replace('.lance', '_embeddings.npy')
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)
    else:
        embeddings = precompute_embeddings(model, args.dataset, cache_path, args.device)

    # load state labels
    print("Loading state labels...")
    ds = lance.dataset(args.dataset)
    tbl = ds.to_table(columns=STATE_COLS)
    states = np.stack([tbl[c].to_pylist() for c in STATE_COLS], axis=1).astype(np.float32)

    state_mean = states.mean(0)
    state_std = states.std(0) + 1e-8
    print(f"State mean: {state_mean}")
    print(f"State std:  {state_std}")

    # save normalization stats alongside the model
    cache_dir = Path(os.path.expanduser("~/.stable_worldmodel/checkpoints/statehead"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "state_mean.npy", state_mean)
    np.save(cache_dir / "state_std.npy", state_std)

    # dataset / dataloaders
    dataset = StateHeadDataset(embeddings, states, state_mean, state_std)
    n_train = int(len(dataset) * args.train_split)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # state head: single linear layer (~1158 params)
    state_head = nn.Linear(EMBED_DIM, STATE_DIM).to(args.device)
    optimizer = torch.optim.AdamW(state_head.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining state head ({sum(p.numel() for p in state_head.parameters())} params) "
          f"for {args.epochs} epochs on {args.device}")
    print(f"Train: {n_train}  Val: {n_val}\n")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # train
        state_head.train()
        train_loss = 0.0
        for emb, target in train_loader:
            emb, target = emb.to(args.device), target.to(args.device)
            pred = state_head(emb)
            loss = nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(emb)
        train_loss /= n_train

        # val
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
    print(f"State head saved to {cache_dir}/statehead.pt")
    print(f"Normalization stats saved to {cache_dir}/state_{{mean,std}}.npy")


if __name__ == '__main__':
    main()
