"""Deteccao: eventos esparsos -> bounding boxes.

Objetos em movimento geram clusters densos de eventos. Dilatamos para
agregar eventos vizinhos e extraimos componentes conectados.
"""
import cv2
import numpy as np


def detect(events, min_area=8, dilate_k=5, min_density=0.15):
    """events: (H,W) {-1,0,1} -> boxes (x,y,w,h) com densidade minima
    de eventos reais dentro (filtra dilatacao sem sinal)."""
    raw = (events != 0).astype(np.uint8)
    active = cv2.dilate(raw, np.ones((dilate_k, dilate_k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(active)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        if raw[labels == i].mean() < min_density:
            continue
        out.append((int(x), int(y), int(w), int(h)))
    return out
