"""Treino em trajetorias reais extraidas pelo tracker.
Uso: python -m src.train_real --epochs 40
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .seq_model import SeqForecastSNN

K_PAST = 16
HORIZONS = [4, 8, 12, 16, 20]


class RealTrajDataset(Dataset):
    def __init__(self, root="data/trajs"):
        self.windows = []
        for f in Path(root).glob("*.npy"):
            tr = np.load(f)
            for s in range(K_PAST, len(tr) - max(HORIZONS)):
                self.windows.append(tr)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        tr = self.windows[i]
        s = np.random.randint(K_PAST, len(tr) - max(HORIZONS))
        past = tr[s - K_PAST:s] / np.array([320.0, 240.0])
        fut = np.concatenate([tr[s + h] / np.array([320.0, 240.0])
                              for h in HORIZONS])
        return (torch.from_numpy(past.astype(np.float32)),
                torch.from_numpy(fut.astype(np.float32)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RealTrajDataset()
    n_test = max(1, int(0.15 * len(ds)))
    tr, te = random_split(ds, [len(ds) - n_test, n_test],
                          generator=torch.Generator().manual_seed(0))
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    dl_te = DataLoader(te, batch_size=256)
    print(f"device={device} train={len(tr)} test={len(te)}")

    model = SeqForecastSNN(len(HORIZONS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        errs = []
        with torch.no_grad():
            for x, y in dl_te:
                p = model(x.to(device)).cpu().numpy().reshape(-1, len(HORIZONS), 2)
                e = np.linalg.norm((p - y.numpy().reshape(-1, len(HORIZONS), 2))
                                   * np.array([320, 240]), axis=2)
                errs.append(e)
        e = np.concatenate(errs)
        mae = float(e.mean())
        if mae < best:
            best = mae
            torch.save({"model": model.state_dict(), "horizons": HORIZONS,
                        "k_past": K_PAST}, "checkpoints/forecast_real.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep}: erro={mae:.1f}px (best={best:.1f})")
    print(f"salvo | melhor={best:.1f}px")


if __name__ == "__main__":
    main()
