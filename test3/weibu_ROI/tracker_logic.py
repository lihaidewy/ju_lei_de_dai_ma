import numpy as np
from online_tracker import KalmanTrackCV


def make_cv_track(track_id, init_center, params):
    return KalmanTrackCV(
        track_id=track_id,
        center=init_center,
        dt=params["KF_DT"],
        q_pos=params["KF_Q_POS"],
        q_vel=params["KF_Q_VEL"],
        r_pos=params["KF_R_POS"],
        init_pos_var=params["KF_INIT_POS_VAR"],
        init_vel_var=params["KF_INIT_VEL_VAR"],
        enable_output_ema=False,
        use_adaptive_r=False,
        use_quality_aware_r=False,
    )


def measurement_from_roi_points(roi_pts):
    if roi_pts.shape[0] == 0:
        return None
    z = np.median(roi_pts, axis=0)
    return np.asarray(z, dtype=float).reshape(2)


def update_track(tracks, gid, z, params):
    if gid not in tracks:
        if z is None:
            return None, 0
        tracks[gid] = make_cv_track(track_id=gid, init_center=z, params=params)
        return tracks[gid].output_center.copy(), 1

    track = tracks[gid]
    track.predict()

    if z is not None:
        return track.update(z).copy(), 1

    track.mark_missed()
    return track.output_center.copy(), 0
