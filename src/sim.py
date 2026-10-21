"""Simulador de trajetoria balistica -> frames para o DVS + posicoes.

Mundo = grid 64x64 (mesmo espaco dos eventos). Bola com gravidade,
quique no chao e reflexao nas paredes.
"""
import numpy as np

W, H = 64, 64
R = 3
G = 0.18           # gravidade por frame
REST = 0.82        # restituicao do quique


def simulate(n_frames=120, rng=None):
    rng = rng or np.random.default_rng()
    x = rng.uniform(8, 56)
    y = rng.uniform(40, 58)
    vx = rng.uniform(0.8, 1.8) * rng.choice([-1, 1])
    vy = rng.uniform(-4.5, -2.0)
    pos = []
    for _ in range(n_frames):
        vy += G
        x += vx
        y += vy
        if x < R or x > W - R:
            vx = -vx
            x = np.clip(x, R, W - R)
        if y > H - R:
            vy = -vy * REST
            y = H - R
            vx *= 0.95                        # atrito no quique
        pos.append((x, y))
    return np.array(pos)


def render_ball(x, y):
    """Frame 64x64 com bola branca."""
    img = np.zeros((H, W, 3), np.uint8)
    import cv2
    cv2.circle(img, (int(round(x)), int(round(y))), R, (255, 255, 255), -1)
    return img


def make_clip(positions, t0, dvs, hist=32, k=8):
    """Frames [t0-32, t0) -> K event frames (K,2,64,64)."""
    dvs.reset()
    evs = []
    for i in range(hist):
        evs.append(dvs.process(render_ball(*positions[t0 - hist + i])))
    buckets = [evs[i * (hist // k):(i + 1) * (hist // k)] for i in range(k)]
    clip = np.stack([np.stack([np.clip(sum(b), 0, 1),
                               np.clip(-sum(b), 0, 1)]) for b in buckets])
    return clip.astype(np.float32)
