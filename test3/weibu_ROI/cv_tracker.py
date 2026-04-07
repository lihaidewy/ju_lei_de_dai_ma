import numpy as np


def infer_init_velocity_from_x(init_pos, params):
    init_pos = np.asarray(init_pos, dtype=float).reshape(2)
    x0 = float(init_pos[0])

    eps = float(params.get("CV_INIT_X_SIGN_EPS", 1.0))
    vy_pos = float(params.get("CV_INIT_VY_POS", 1.1))
    vy_neg = float(params.get("CV_INIT_VY_NEG", -1.1))
    vy_center = float(params.get("CV_INIT_VY_CENTER", 0.0))

    if x0 > eps:
        return vy_pos
    elif x0 < -eps:
        return vy_neg
    else:
        return vy_center


class KalmanCVTrack:
    """
    专用 Kalman CV（强化 Y 精度）

    state = [x, y, vy]
    """

    def __init__(self, track_id, init_pos, init_frame, init_vy, params):
        self.track_id = int(track_id)
        self.last_frame = int(init_frame)
        self.miss_count = 0

        x0 = float(init_pos[0])
        y0 = float(init_pos[1])
        vy0 = float(init_vy)

        # ===== 状态 =====
        self.x = np.array([[x0], [y0], [vy0]], dtype=float)

        # ===== 协方差 =====
        self.P = np.diag([
            params.get("KF_INIT_P_X", 1.0),
            params.get("KF_INIT_P_Y", 4.0),
            params.get("KF_INIT_P_VY", 4.0),
        ]).astype(float)

        # ===== 噪声 =====
        self.q_x = params.get("KF_Q_X", 0.01)
        self.q_y = params.get("KF_Q_Y", 0.10)
        self.q_vy = params.get("KF_Q_VY", 0.20)

        self.r_x = params.get("KF_R_X", 0.3 ** 2)
        self.r_y = params.get("KF_R_Y", 0.8 ** 2)

        # ===== 控制 =====
        self.vy_decay = params.get("KF_MISS_VY_DECAY", 0.98)
        self.max_abs_vy = params.get("KF_MAX_ABS_VY", 5.0)

        self.output_center = np.array([x0, y0], dtype=float)

    def _F(self, dt):
        return np.array([
            [1, 0, 0],
            [0, 1, dt],
            [0, 0, 1]
        ], dtype=float)

    def _Q(self, dt):
        scale = max(dt, 1.0)
        return np.diag([
            self.q_x * scale,
            self.q_y * scale,
            self.q_vy * scale
        ])

    def predict(self, frame_id):
        dt = float(frame_id - self.last_frame)
        if dt <= 0:
            return self.output_center.copy()

        F = self._F(dt)
        Q = self._Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # 限速
        self.x[2, 0] = np.clip(self.x[2, 0], -self.max_abs_vy, self.max_abs_vy)

        self.last_frame = frame_id
        self.output_center = self.x[:2, 0].copy()
        return self.output_center.copy()

    def update_with_measurement(self, frame_id, z):
        z = np.asarray(z, dtype=float).reshape(2, 1)

        self.predict(frame_id)

        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=float)

        R = np.diag([self.r_x, self.r_y])

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

        # 再限一次速度
        self.x[2, 0] = np.clip(self.x[2, 0], -self.max_abs_vy, self.max_abs_vy)

        self.miss_count = 0
        self.output_center = self.x[:2, 0].copy()
        return self.output_center.copy()

    def update_without_measurement(self, frame_id):
        self.predict(frame_id)

        # ===== 核心：防漂 =====
        self.x[2, 0] *= self.vy_decay

        self.miss_count += 1
        self.output_center = self.x[:2, 0].copy()
        return self.output_center.copy()


def update_cv_track(tracks, gid, frame_id, z, params):
    max_misses = int(params.get("CV_MAX_MISSES", 10))

    if gid not in tracks:
        if z is None:
            return None, 0

        init_vy = infer_init_velocity_from_x(z, params)

        tracks[gid] = KalmanCVTrack(
            track_id=gid,
            init_pos=z,
            init_frame=frame_id,
            init_vy=init_vy,
            params=params
        )
        return tracks[gid].output_center.copy(), 1

    trk = tracks[gid]

    if z is not None:
        return trk.update_with_measurement(frame_id, z), 1

    out = trk.update_without_measurement(frame_id)

    if trk.miss_count > max_misses:
        del tracks[gid]
        return None, 0

    return out, 0
