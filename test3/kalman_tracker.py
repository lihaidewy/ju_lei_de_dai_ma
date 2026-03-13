import numpy as np


class KalmanTrackCV:
    """
    Constant-Velocity Kalman Filter
    state = [x, y, vx, vy]
    measurement = [x, y]
    """

    def __init__(self, dt=1.0, q_pos=1.0, q_vel=1.0, r_pos=1.0):
        self.dt = float(dt)

        # State transition
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        # Measurement model
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)

        # Process noise
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)

        # Measurement noise
        self.R = np.diag([r_pos, r_pos]).astype(float)

        self.x = None   # state
        self.P = None   # covariance
        self.initialized = False

    def init_from_measurement(self, z):
        z = np.asarray(z, dtype=float).reshape(2)

        self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([10.0, 10.0, 25.0, 25.0]).astype(float)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            raise RuntimeError("Kalman track not initialized")

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(2)

        if not self.initialized:
            self.init_from_measurement(z)
            return self.x[:2].copy()

        # predict
        self.predict()

        # innovation
        y = z - (self.H @ self.x)

        # innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # update state
        self.x = self.x + K @ y

        # update covariance
        I = np.eye(self.P.shape[0], dtype=float)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:2].copy()


class KalmanTrackerManager:
    """
    Manage multiple tracks using GT gid as track_id (validation stage).
    """

    def __init__(self, dt=1.0, q_pos=1.0, q_vel=1.0, r_pos=1.0):
        self.dt = float(dt)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_pos = float(r_pos)
        self.tracks = {}

    def reset(self):
        self.tracks = {}

    def update(self, track_id, center):
        if track_id not in self.tracks:
            self.tracks[track_id] = KalmanTrackCV(
                dt=self.dt,
                q_pos=self.q_pos,
                q_vel=self.q_vel,
                r_pos=self.r_pos,
            )

        return self.tracks[track_id].update(center)
