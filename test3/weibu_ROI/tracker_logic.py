import numpy as np
from online_tracker import KalmanTrackCV

"""负责跟踪相关的逻辑，包括根据 GT 生成 ROI，计算 ROI 内的量测，以及维护 Kalman 跟踪器。"""


def infer_init_velocity_from_center(center_xy, params):
    """
    根据目标初始位置的 x 符号给出数据驱动的速度方向先验：
      - x > +eps : 远离组，使用 KF_INIT_VY_POS
      - x < -eps : 接近组，使用 KF_INIT_VY_NEG
      - |x| <= eps : 中间组，使用 KF_INIT_VY_CENTER

    vx 暂时默认固定为 KF_INIT_VX。
    """
    center_xy = np.asarray(center_xy, dtype=float).reshape(2)
    x0 = float(center_xy[0])

    x_sign_eps = float(params.get("KF_INIT_X_SIGN_EPS", 1.0))
    vx0 = float(params.get("KF_INIT_VX", 0.0))

    vy_pos = float(params.get("KF_INIT_VY_POS", +1.1))
    vy_neg = float(params.get("KF_INIT_VY_NEG", -1.1))
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
    elif x0 < -x_sign_eps:
        return var_neg
    else:
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
        enable_output_ema=False,
        use_adaptive_r=False,
        use_quality_aware_r=False,
    )


def measurement_from_roi_points(roi_pts):
    if roi_pts.shape[0] == 0:
        return None
    z = np.median(roi_pts, axis=0)
    return np.asarray(z, dtype=float).reshape(2)


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

    # 无论是否有量测，先做一步 CV 预测
    pred_xy = track.predict().copy()

    if z is not None:
        updated_xy = track.update(z).copy()
        return updated_xy, 1

    track.mark_missed()

    if _should_delete_track(track, params):
        del tracks[gid]
        return None, 0

    return pred_xy, 0
