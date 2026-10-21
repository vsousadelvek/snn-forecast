"""Treino do preditor de trajetoria — dados sinteticos infinitos.
Uso: python -m src.train --epochs 30
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .sim import simulate, make_clip
from .events import DVSSimulator
from .model import ForecastSNN

HORIZONS = [4, 8, 12, 16, 20]   # frames a frente


class TrajDataset(Dataset):
    """Gera uma trajetoria nova por amostra — dataset infinito."""

    def __init__(self, n=4000):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        pos = simulate(120, np.random.default_rng(i))
        t0 = np.random.randint(50, 95)
        clip = make_clip(pos, t0, DVSSimulator(size=(64, 64), threshold=0.1))
        y = np.concatenate([pos[min(t0 + h, 119)] / 64.0 for h in HORIZONS])
        return torch.from_numpy(clip), torch.from_numpy(y.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, te = TrajDataset(4000), TrajDataset(400)
    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dl_te = DataLoader(te, batch_size=128)
    print(f"device={device}")

    model = ForecastSNN(len(HORIZONS)).to(device)
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
                e = np.linalg.norm((p - y.numpy().reshape(-1, len(HORIZONS), 2)) * 64, axis=2)
                errs.append(e)
        e = np.concatenate(errs)
        mae = float(e.mean())
        if mae < best:
            best = mae
            torch.save({"model": model.state_dict(), "horizons": HORIZONS},
                       "checkpoints/forecast_snn.pt")
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep}: erro medio={mae:.2f}px por horizonte | "
                  f"min={e.min():.2f} max={e.max():.2f} (best={best:.2f})")
    print(f"salvo | melhor erro medio={best:.2f}px")


if __name__ == "__main__":
    main()
