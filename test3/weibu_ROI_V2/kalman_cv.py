import numpy as np


class BaseKalmanTrack:
    def __init__(
        self,
        track_id,
        center,
        dt=1.0,
        r_pos=1.50,
        init_pos_var=4.0,
        init_vel_var=9.0,
        init_velocity=None,
    ):
        self.track_id = int(track_id)
        self.dt = float(dt)

        self.age = 1
        self.hit_count = 1
        self.hit_streak = 1
        self.miss_count = 0

        self.raw_center = np.asarray(center, dtype=float).reshape(2)
        self.filtered_center = self.raw_center.copy()
        self.output_center = self.raw_center.copy()

        self.base_r_pos = float(r_pos)
        self._init_pos_var = float(init_pos_var)
        self._init_vel_var = float(init_vel_var)

        if init_velocity is None:
            self._init_velocity = np.array([0.0, 0.0], dtype=float)
        else:
            self._init_velocity = np.asarray(init_velocity, dtype=float).reshape(2).copy()

        self.state = None
        self.P = None
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        self._build_model(center)

    def _build_model(self, center):
        raise NotImplementedError

    @property
    def position(self):
        return self.state[:2].copy()

    @property
    def velocity(self):
        return self.state[2:4].copy()

    def on_update_success(self):
        self.hit_count += 1
        self.hit_streak += 1
        self.miss_count = 0
        self.age += 1

    def on_missed(self):
        self.miss_count += 1
        self.hit_streak = 0
        self.age += 1

    def predict(self):
        self.state = self.F.dot(self.state)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        self.filtered_center = self.state[:2].copy()
        self.output_center = self.filtered_center.copy()
        return self.output_center.copy()

    def update(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(2)

        y = z - self.H.dot(self.state)
        s = self.H.dot(self.P).dot(self.H.T) + self.R
        pht = self.P.dot(self.H.T)

        try:
            k = np.linalg.solve(s, pht.T).T
        except np.linalg.LinAlgError:
            k = pht.dot(np.linalg.pinv(s))

        self.state = self.state + k.dot(y)

        i = np.eye(self.P.shape[0], dtype=float)
        kh = k.dot(self.H)
        self.P = (i - kh).dot(self.P).dot((i - kh).T) + k.dot(self.R).dot(k.T)

        self.filtered_center = self.state[:2].copy()
        self.output_center = self.filtered_center.copy()

        self.raw_center = z.copy()
        self.on_update_success()
        return self.output_center.copy()

    def mark_missed(self):
        self.on_missed()


class KalmanTrackCV(BaseKalmanTrack):
    def __init__(self, track_id, center, dt=1.0, q_pos=0.30, q_vel=0.50, **kwargs):
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        super().__init__(track_id=track_id, center=center, dt=dt, **kwargs)

    def _build_model(self, center):
        x, y = np.asarray(center, dtype=float).reshape(2)
        vx0, vy0 = self._init_velocity.astype(float)

        self.state = np.array([x, y, vx0, vy0], dtype=float)
        self.P = np.diag([
            self._init_pos_var,
            self._init_pos_var,
            self._init_vel_var,
            self._init_vel_var,
        ]).astype(float)

        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=float)

        self.Q = np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_vel]).astype(float)
        self.R = np.eye(2, dtype=float) * float(self.base_r_pos)


def infer_init_velocity_from_center(center_xy, params):
    """
    根据目标初始位置的 x 符号给出数据驱动的速度方向先验：
      - x > +eps : 远离组，使用 KF_INIT_VY_POS
      - x < -eps : 接近组，使用 KF_INIT_VY_NEG
      - |x| <= eps : 中间组，使用 KF_INIT_VY_CENTER

    vx 默认固定为 KF_INIT_VX。
    """
    center_xy = np.asarray(center_xy, dtype=float).reshape(2)
    x0 = float(center_xy[0])

    x_sign_eps = float(params.get("KF_INIT_X_SIGN_EPS", 1.0))
    vx0 = float(params.get("KF_INIT_VX", 0.0))

    vy_pos = float(params.get("KF_INIT_VY_POS", +8.0))
    vy_neg = float(params.get("KF_INIT_VY_NEG", -8.0))
    vy_center = float(params.get("KF_INIT_VY_CENTER", 0.0))

    if x0 > x_sign_eps:
        vy0 = vy_pos
    elif x0 < -x_sign_eps:
        vy0 = vy_neg
    else:
        vy0 = vy_center

    return np.array([vx0, vy0], dtype=float)


def infer_init_vel_var_from_center(center_xy, params):
    """
    根据初始位置所在分组，选择对应的初始速度方差。
    """
    center_xy = np.asarray(center_xy, dtype=float).reshape(2)
    x0 = float(center_xy[0])

    x_sign_eps = float(params.get("KF_INIT_X_SIGN_EPS", 1.0))
    default_var = float(params.get("KF_INIT_VEL_VAR", 9.0))

    var_pos = float(params.get("KF_INIT_VEL_VAR_POS", default_var))
    var_neg = float(params.get("KF_INIT_VEL_VAR_NEG", default_var))
    var_center = float(params.get("KF_INIT_VEL_VAR_CENTER", default_var))

    if x0 > x_sign_eps:
        return var_pos
    if x0 < -x_sign_eps:
        return var_neg
    return var_center


def make_cv_track(track_id, init_center, params):
    init_velocity = infer_init_velocity_from_center(init_center, params)
    init_vel_var = infer_init_vel_var_from_center(init_center, params)

    return KalmanTrackCV(
        track_id=track_id,
        center=init_center,
        dt=params["KF_DT"],
        q_pos=params["KF_Q_POS"],
        q_vel=params["KF_Q_VEL"],
        r_pos=params["KF_R_POS"],
        init_pos_var=params["KF_INIT_POS_VAR"],
        init_vel_var=init_vel_var,
        init_velocity=init_velocity,
    )


def _should_delete_track(track, params):
    max_misses = int(params.get("KF_MAX_MISSES", 10))
    return int(getattr(track, "miss_count", 0)) > max_misses


def update_track(tracks, gid, z, params):
    """
    返回:
        output_xy: np.array([x, y]) 或 None
        used_measurement: 1/0

    逻辑:
      - 新轨迹:
          * 有量测 -> 创建并输出
          * 无量测 -> 不创建
      - 老轨迹:
          * 先 predict
          * 有量测 -> update
          * 无量测 -> mark_missed 后输出预测位置
      - 连续 miss 过多 -> 删除轨迹
    """
    if gid not in tracks:
        if z is None:
            return None, 0
        tracks[gid] = make_cv_track(track_id=gid, init_center=z, params=params)
        return tracks[gid].output_center.copy(), 1

    track = tracks[gid]

    pred_xy = track.predict().copy()

    if z is not None:
        updated_xy = track.update(z).copy()
        return updated_xy, 1

    track.mark_missed()

    if _should_delete_track(track, params):
        del tracks[gid]
        return None, 0

    return pred_xy, 0
