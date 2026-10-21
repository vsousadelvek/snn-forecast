"""Simulador de camera de eventos (DVS) — mesma base do snn-gesture."""
import cv2
import numpy as np


class DVSSimulator:
    """Emite eventos (+1 ON / -1 OFF) quando |log(I_t) - log(I_ref)| > C."""

    def __init__(self, size=(160, 120), threshold=0.18):
        self.size = size  # (W, H)
        self.C = threshold
        self.ref = None

    def _preprocess(self, frame_bgr):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, self.size, interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        return np.log1p(g.astype(np.float32) / 255.0)

    def reset(self):
        self.ref = None

    def process(self, frame_bgr):
        cur = self._preprocess(frame_bgr)
        if self.ref is None:
            self.ref = cur
            return np.zeros(self.size[::-1], np.float32)
        diff = cur - self.ref
        events = np.zeros_like(diff)
        events[diff > self.C] = 1.0
        events[diff < -self.C] = -1.0
        self.ref = np.where(events != 0, cur, self.ref)
        return events
