"""Tracker multi-objeto com filtro de Kalman implementado do zero.

Estado: [x, y, vx, vy] (centroide + velocidade), modelo de velocidade
constante. Matching guloso por distancia ao centroide PREDITO —
resolve bem a 30fps sem Hungarian.
"""
import numpy as np


class Kalman:
    """Kalman 4D: [x, y, vx, vy], velocidade constante, dt=1."""

    def __init__(self, x, y):
        self.X = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([10.0, 10.0, 50.0, 50.0])
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                           [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.Q = np.diag([0.5, 0.5, 2.0, 2.0])
        self.R = np.eye(2) * 4.0

    def predict(self):
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X[:2]

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + K @ (np.asarray(z) - self.H @ self.X)
        self.P = (np.eye(4) - K @ self.H) @ self.P


class Track:
    _next_id = 0

    def __init__(self, box):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = np.array(box, float)
        self.kf = Kalman(*self.centroid)
        self.trail = [self.centroid]
        self.votes = []
        self.hits, self.misses = 1, 0

    @property
    def centroid(self):
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2])

    def predict(self):
        c = self.kf.predict()
        self.box[:2] = c - self.box[2:] / 2
        self.misses += 1

    def update(self, box):
        self.kf.update([box[0] + box[2] / 2, box[1] + box[3] / 2])
        self.box = np.array(box, float)
        self.trail.append(self.centroid.copy())
        if len(self.trail) > 40:
            self.trail.pop(0)
        self.hits += 1
        self.misses = 0


class Tracker:
    def __init__(self, max_dist=45, max_misses=25, min_hits=8):
        self.tracks = []
        self.dead = []
        self.max_dist, self.max_misses, self.min_hits = \
            max_dist, max_misses, min_hits

    def step(self, boxes):
        for t in self.tracks:
            t.predict()

        unmatched = set(range(len(boxes)))
        for t in self.tracks:
            best, best_d = None, self.max_dist
            for i in unmatched:
                x, y, w, h = boxes[i]
                d = np.linalg.norm(t.centroid - (x + w / 2, y + h / 2))
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                t.update(boxes[best])
                unmatched.discard(best)

        for i in unmatched:
            self.tracks.append(Track(boxes[i]))
        still = []
        for t in self.tracks:
            (still if t.misses <= self.max_misses else self.dead).append(t)
        self.tracks = still
        return [t for t in self.tracks if t.hits >= self.min_hits]
