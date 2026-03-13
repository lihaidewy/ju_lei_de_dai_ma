import numpy as np


class EMATracker:
    def __init__(self, alpha=0.8):
        self.alpha = float(alpha)
        self.state = {}

    def reset(self):
        self.state = {}

    def update(self, track_id, center):
        center = np.asarray(center, dtype=float)

        if track_id not in self.state:
            self.state[track_id] = center.copy()
            return center.copy()

        prev = self.state[track_id]
        smoothed = self.alpha * center + (1.0 - self.alpha) * prev
        self.state[track_id] = smoothed
        return smoothed.copy()
