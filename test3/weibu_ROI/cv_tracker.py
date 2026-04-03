import numpy as np


def infer_init_velocity_from_x(init_pos, params):
    """
    根据目标初始位置 x 的符号，给初始速度方向先验：
      - x > +eps : 远离组，vy 为正
      - x < -eps : 接近组，vy 为负
      - |x| <= eps : 中间组，vy 置 0 或给较小默认值

    注意：
      vx 在当前场景中固定为 0。
    """
    init_pos = np.asarray(init_pos, dtype=float).reshape(2)
    x0 = float(init_pos[0])

    eps = float(params.get("CV_INIT_X_SIGN_EPS", 1.0))
    vx0 = 0.0

    vy_pos = float(params.get("CV_INIT_VY_POS", 8.0))
    vy_neg = float(params.get("CV_INIT_VY_NEG", -8.0))
    vy_center = float(params.get("CV_INIT_VY_CENTER", 0.0))

    if x0 > eps:
        vy0 = vy_pos
    elif x0 < -eps:
        vy0 = vy_neg
    else:
        vy0 = vy_center

    return np.array([vx0, vy0], dtype=float)


class SimpleCVTrack:
    """
    一个非常简单的 CV（constant velocity）递推器：
    - 状态只保留 pos, vel
    - 有量测时：位置直接用量测，速度由相邻两次位置差估计
    - 无量测时：按常速度外推
    """

    def __init__(self, track_id, init_pos, init_frame, init_vel=None):
        self.track_id = int(track_id)
        self.pos = np.asarray(init_pos, dtype=float).reshape(2)
        self.vel = (
            np.zeros(2, dtype=float)
            if init_vel is None
            else np.asarray(init_vel, dtype=float).reshape(2)
        )
        self.last_frame = int(init_frame)
        self.miss_count = 0
        self.output_center = self.pos.copy()

    def predict_to(self, frame_id):
        frame_id = int(frame_id)
        dt = float(frame_id - self.last_frame)
        if dt < 0:
            raise ValueError(
                f"frame_id 倒退: last_frame={self.last_frame}, frame_id={frame_id}"
            )
        return self.pos + self.vel * dt

    def update_with_measurement(self, frame_id, z):
        """
        有量测时：
        1) 先按旧速度预测到当前帧
        2) 当前位置直接采用量测 z
        3) 新速度由 (z - 上一次输出位置) / dt 估计
        """
        frame_id = int(frame_id)
        z = np.asarray(z, dtype=float).reshape(2)

        dt = float(frame_id - self.last_frame)
        prev_output = self.output_center.copy()

        _pred = self.predict_to(frame_id)

        self.pos = z.copy()

        if dt > 0:
            new_vel = (self.pos - prev_output) / dt

            # 当前场景中，vx 固定为 0
            self.vel[0] = 0.0
            self.vel[1] = new_vel[1]

        self.last_frame = frame_id
        self.miss_count = 0
        self.output_center = self.pos.copy()
        return self.output_center.copy()

    def update_without_measurement(self, frame_id):
        """
        无量测时：
        直接做常速度外推，速度不变
        """
        frame_id = int(frame_id)
        pred = self.predict_to(frame_id)

        self.pos = pred.copy()
        self.last_frame = frame_id
        self.miss_count += 1
        self.output_center = self.pos.copy()
        return self.output_center.copy()


def update_cv_track(tracks, gid, frame_id, z, params):
    """
    返回:
        output_xy: np.array([x, y]) 或 None
        used_measurement: 1/0

    逻辑:
      - 新轨迹:
          * 有量测 -> 创建并输出
          * 无量测 -> 不创建
      - 老轨迹:
          * 有量测 -> 更新
          * 无量测 -> 常速度外推
      - 连续 miss 过多 -> 删除轨迹
    """
    max_misses = int(params.get("CV_MAX_MISSES", 10))

    if gid not in tracks:
        if z is None:
            return None, 0

        init_vel = infer_init_velocity_from_x(z, params)

        tracks[gid] = SimpleCVTrack(
            track_id=gid,
            init_pos=z,
            init_frame=frame_id,
            init_vel=init_vel,
        )
        return tracks[gid].output_center.copy(), 1

    trk = tracks[gid]

    if z is not None:
        out = trk.update_with_measurement(frame_id, z)
        return out, 1

    out = trk.update_without_measurement(frame_id)

    if trk.miss_count > max_misses:
        del tracks[gid]
        return None, 0

    return out, 0
