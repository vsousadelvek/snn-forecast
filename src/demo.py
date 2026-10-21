"""Demo: bola em movimento -> eventos -> SNN preve a trajetoria futura.

3 paineis: mundo (trilha real + futuro previsto vs real) | eventos | erro.
Uso: python -m src.demo -o demo/forecast.mp4
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch

from .sim import simulate, render_ball, W, H, R
from .events import DVSSimulator
from .model import ForecastSNN
from .train import HORIZONS

SCALE = 6
PREDICT_EVERY = 4


def render_events_map(evs):
    acc = np.clip(sum(evs[-4:]), -1, 1) if evs else np.zeros((H, W))
    img = np.zeros((H, W, 3), np.uint8)
    img[acc > 0] = (255, 255, 0); img[acc < 0] = (255, 0, 255)
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_LINEAR)
    return cv2.add(img, cv2.GaussianBlur(img, (0, 0), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="demo/forecast.mp4")
    ap.add_argument("--seconds", type=int, default=25)
    args = ap.parse_args()

    ckpt = torch.load("checkpoints/forecast_snn.pt", map_location="cpu",
                      weights_only=False)
    horizons = ckpt["horizons"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ForecastSNN(len(horizons)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rng = np.random.default_rng(7)
    pos = simulate(160, rng)                     # trajetoria real completa
    dvs = DVSSimulator(size=(W, H), threshold=0.1)
    ev_buf = deque(maxlen=32)
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                          30, (1200, 400))

    pred_pts, pred_err = None, 0.0
    N = min(160, int(args.seconds * 30))
    for t in range(N):
        ev_buf.append(dvs.process(render_ball(*pos[t])))

        if t >= 32 and t % PREDICT_EVERY == 0:
            evs = list(ev_buf)
            k = 8
            buckets = [evs[i * 4:(i + 1) * 4] for i in range(k)]
            clip = np.stack([np.stack([np.clip(sum(b), 0, 1),
                                       np.clip(-sum(b), 0, 1)])
                             for b in buckets]).astype(np.float32)
            x = torch.from_numpy(clip).unsqueeze(0).to(device)
            with torch.no_grad():
                p = model(x)[0].cpu().numpy().reshape(-1, 2)
            pred_pts = np.clip(p, 0, 1) * W
            fut = np.array([pos[min(t + h, len(pos) - 1)] for h in horizons])
            pred_err = float(np.linalg.norm(pred_pts - fut, axis=1).mean())

        # painel 1: mundo — renderizado direto em 400px p/ ficar nitido
        s = 400 / W
        P = lambda p: (int(p[0] * s), int(p[1] * s))
        world = np.full((400, 400, 3), (14, 14, 18), np.uint8)
        hist = pos[max(0, t - 32):t + 1]
        for a, b in zip(hist, hist[1:]):
            cv2.line(world, P(a), P(b), (255, 200, 0), 2)
        if t + 20 < len(pos):
            fut_true = pos[t:t + 21]
            for a, b in zip(fut_true, fut_true[1:]):
                cv2.line(world, P(a), P(b), (80, 80, 80), 1)
        if pred_pts is not None:
            pts = np.vstack([pos[t], pred_pts])
            for a, b in zip(pts, pts[1:]):
                cv2.line(world, P(a), P(b), (0, 255, 0), 2)
            for p in pred_pts:
                cv2.circle(world, P(p), 5, (0, 255, 0), 2)
        cv2.circle(world, P(pos[t]), int(R * s), (255, 255, 255), -1)

        # painel 2: eventos
        ev_img = render_events_map(list(ev_buf))

        # painel 3: info
        panel = np.zeros((400, 400, 3), np.uint8)
        cv2.putText(panel, "SNN FORECAST", (90, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (120, 120, 120), 2)
        cv2.putText(panel, "horizonte: +4..+20 frames", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.putText(panel, f"erro medio: {pred_err:.1f}px", (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(panel, "verde = previsto | cinza = real", (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

        cv2.putText(world, "mundo (previsao)", (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ev_img, "eventos (DVS)", (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(np.hstack([world, ev_img, panel]))

    out.release()
    print(f"salvo: {args.out}")


if __name__ == "__main__":
    main()
