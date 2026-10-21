"""Demo em video real: tracker -> SNN preve trajetoria futura de cada objeto.

Uso: python -m src.demo_real ../snn-tracker/raw/c02.mp4 -o demo/real.mp4
"""
import argparse
from collections import deque

import cv2
import numpy as np
import torch

from .events import DVSSimulator
from .detect import detect
from .tracker import Tracker
from .seq_model import SeqForecastSNN
from .train_real import K_PAST

EV_W, EV_H = 320, 240
PREDICT_EVERY = 4
COLORS = [(255, 200, 0), (0, 255, 200), (255, 100, 200), (100, 200, 255),
          (255, 255, 0), (0, 200, 255), (200, 255, 100), (255, 0, 200)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="demo/real.mp4")
    args = ap.parse_args()

    ckpt = torch.load("checkpoints/forecast_real.pt", map_location="cpu",
                      weights_only=False)
    horizons = ckpt["horizons"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeqForecastSNN(len(horizons)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (1200, 400))
    dvs = DVSSimulator(size=(EV_W, EV_H), threshold=0.15)
    tracker = Tracker()
    ev_trail = deque(maxlen=3)
    tick = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tick += 1
        ev_trail.append(dvs.process(frame))
        acc = np.clip(sum(ev_trail), -1, 1)
        boxes = [b for b in detect(acc, min_area=30, dilate_k=7,
                                   min_density=0.10)
                 if b[2] >= 5 and b[3] >= 5]
        tracks = tracker.step(boxes)

        sx = sy = 400 / 400  # dummy
        sx, sy = 400 / EV_W, 400 / EV_H
        view = cv2.resize(frame, (400, 400))
        ev_img = cv2.resize(cv2.cvtColor(
            (np.abs(acc) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            (400, 400))
        ev_img = cv2.add(ev_img, cv2.GaussianBlur(ev_img, (0, 0), 5))

        for t in tracks:
            x, y, w, h = t.box
            color = COLORS[t.id % len(COLORS)]
            for img in (view, ev_img):
                p1 = (int(x * sx), int(y * sy))
                p2 = (int((x + w) * sx), int((y + h) * sy))
                cv2.rectangle(img, p1, p2, color, 2)
            pts = [(int(c[0] * sx), int(c[1] * sy)) for c in t.trail]
            for a, b in zip(pts, pts[1:]):
                cv2.line(view, a, b, color, 2)

            # previsao a cada PREDICT_EVERY frames, desenhada sempre
            if len(t.trail) >= K_PAST and tick % PREDICT_EVERY == 0:
                past = np.array(t.trail[-K_PAST:], dtype=np.float32)
                kk = np.ones(3) / 3
                past[:, 0] = np.convolve(past[:, 0], kk, "same")
                past[:, 1] = np.convolve(past[:, 1], kk, "same")
                deltas = np.diff(past, axis=0)
                speed = np.linalg.norm(deltas, axis=1).mean() + 1e-3
                xin = torch.from_numpy(
                    (deltas / speed)[None].astype(np.float32)).to(device)
                with torch.no_grad():
                    p = model(xin)[0].cpu().numpy().reshape(-1, 2)
                new_pred = t.centroid + p * speed * 16.0
                # EMA entre previsoes consecutivas -> caminho estavel
                if getattr(t, "pred_abs", None) is not None:
                    t.pred_abs = 0.55 * t.pred_abs + 0.45 * new_pred
                else:
                    t.pred_abs = new_pred
            if getattr(t, "pred_abs", None) is not None:
                # suaviza os pontos por horizonte (media movel)
                sm = t.pred_abs.copy()
                sm[1:-1] = (t.pred_abs[:-2] + t.pred_abs[1:-1]
                            + t.pred_abs[2:]) / 3
                fut = np.vstack([t.centroid, sm]) * [sx, sy]
                pts_f = [tuple(p.astype(int)) for p in fut]
                for a, b in zip(pts_f, pts_f[1:]):
                    cv2.line(view, a, b, (0, 255, 0), 2)
                for p in pts_f[1:]:
                    cv2.circle(view, p, 4, (0, 255, 0), -1)

        panel = np.zeros((400, 400, 3), np.uint8)
        cv2.putText(panel, "SNN FORECAST", (80, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (120, 120, 120), 2)
        cv2.putText(panel, "video real", (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (180, 180, 180), 1)
        cv2.putText(panel, f"tracks: {len(tracks)}", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(panel, "verde = trajetoria prevista", (20, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(view, "camera + forecast", (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ev_img, "eventos (DVS)", (10, 390),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(np.hstack([view, ev_img, panel]))

    cap.release()
    out.release()
    print(f"salvo: {args.out}")


if __name__ == "__main__":
    main()
