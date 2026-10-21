"""Extrai trajetorias reais de videos via DVS + tracker (auto-labeling).

Uso: python -m src.extract_trajs ../snn-tracker/raw/c01.mp4 ...
"""
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .events import DVSSimulator
from .detect import detect
from .tracker import Tracker

EV_W, EV_H = 320, 240


def extract(video_path):
    cap = cv2.VideoCapture(str(video_path))
    dvs = DVSSimulator(size=(EV_W, EV_H), threshold=0.15)
    tracker = Tracker()
    trail = deque(maxlen=3)
    n_fr = 0
    while n_fr < 900:
        ok, f = cap.read()
        if not ok:
            break
        n_fr += 1
        trail.append(dvs.process(f))
        if len(trail) < 3:
            continue
        acc = np.clip(sum(trail), -1, 1)
        boxes = [b for b in detect(acc, min_area=30, dilate_k=7,
                                   min_density=0.10)
                 if b[2] >= 5 and b[3] >= 5]
        tracker.step(boxes)
    cap.release()
    return [t.trail for t in tracker.tracks + tracker.dead if len(t.trail) > 30]


def main():
    out_dir = Path("data/trajs")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for vid in sys.argv[1:]:
        trajs = extract(vid)
        for tr in trajs:
            np.save(out_dir / f"{Path(vid).stem}_{total:04d}.npy",
                    np.array(tr, dtype=np.float32))
            total += 1
        print(f"{vid}: {len(trajs)} trajetorias")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
