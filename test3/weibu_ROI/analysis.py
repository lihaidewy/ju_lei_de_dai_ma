import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from roi_analysis import resolve_target_side_geometry, get_roi_points
from tracker_logic import measurement_from_roi_points, update_track


def build_valid_frames(radar_data, gt_df, start_frame, end_frame):
    target_frames = list(range(start_frame, end_frame + 1))
    gt_frames = set(gt_df["Frame"].astype(int).unique())
    return [fid for fid in target_frames if fid in radar_data and fid in gt_frames]


def fit_cv_window(meas_history, target_frame, window_size=5, min_points=2):
    """
    对单个 gid 的历史原始量测做滑动窗口匀速拟合。
    meas_history: deque([(frame_id, np.array([x, y])), ...])
    target_frame: 当前要输出估计的帧号
    返回:
        np.array([x_hat, y_hat]) 或 None
    """
    if meas_history is None or len(meas_history) < min_points:
        return None

    hist = list(meas_history)[-window_size:]
    if len(hist) < min_points:
        return None

    frames = np.array([item[0] for item in hist], dtype=float)
    xs = np.array([item[1][0] for item in hist], dtype=float)
    ys = np.array([item[1][1] for item in hist], dtype=float)

    # 点数不足时无法稳定拟合
    if len(np.unique(frames)) < min_points:
        return None

    # 一阶拟合：x = ax * t + bx, y = ay * t + by
    coef_x = np.polyfit(frames, xs, deg=1)
    coef_y = np.polyfit(frames, ys, deg=1)

    x_hat = coef_x[0] * float(target_frame) + coef_x[1]
    y_hat = coef_y[0] * float(target_frame) + coef_y[1]

    return np.array([x_hat, y_hat], dtype=float)


def process_one_target(fid, gt_row, meas_df, tracks, fit_histories, params):
    gid = int(gt_row.ID)
    model = int(gt_row.model)

    gt_dim = params["GT_DIM"]
    if model not in gt_dim:
        return None, None

    gx = float(gt_row.X)
    gy = float(gt_row.Y)

    yaw_deg = float(gt_row.YAW) if hasattr(gt_row, "YAW") and gt_row.YAW is not None else 0.0
    yaw_rad = math.radians(yaw_deg)

    length = float(gt_dim[model]["L"])
    width = float(gt_dim[model]["W"])

    geometry = resolve_target_side_geometry(
        center_x=gx,
        center_y=gy,
        length=length,
        width=width,
        yaw_rad=yaw_rad,
    )

    side_info = geometry["side_info"]
    outward_sign = geometry["outward_sign"]

    # 当前评估真值仍然是“近端边中点”
    x_gt, y_gt = geometry["midpoint_world"]

    roi_pts, _ = get_roi_points(
        meas_df=meas_df,
        gx=gx,
        gy=gy,
        yaw_rad=yaw_rad,
        side_info=side_info,
        outward_sign=outward_sign,
        outer_margin=params["ROI_OUTER"],
        inner_margin=params["ROI_INNER"],
    )

    n_roi = int(roi_pts.shape[0])
    z = measurement_from_roi_points(roi_pts) if n_roi >= params["MIN_ROI_POINTS"] else None

    estimation_mode = params.get("ESTIMATION_MODE", "kalman").lower()

    # 统一输出变量
    output_xy = None
    used_measurement = 0
    track_exists = 0
    fit_points = 0

    if estimation_mode == "kalman":
        output_xy, used_measurement = update_track(tracks, gid, z, params)
        track_exists = 1 if gid in tracks else 0

    elif estimation_mode == "raw":
        output_xy = z
        used_measurement = 1 if z is not None else 0
        track_exists = 0

    elif estimation_mode == "cv_fit":
        history = fit_histories[gid]

        # 先把当前帧的原始量测加入历史，再做窗口拟合
        if z is not None:
            history.append((fid, z.copy()))
            used_measurement = 1
        else:
            used_measurement = 0

        fit_points = len(history)

        output_xy = fit_cv_window(
            meas_history=history,
            target_frame=fid,
            window_size=params["CVFIT_WINDOW_SIZE"],
            min_points=params["CVFIT_MIN_POINTS"],
        )
        track_exists = 1 if len(history) >= params["CVFIT_MIN_POINTS"] else 0

    else:
        raise ValueError(
            f"未知 ESTIMATION_MODE={estimation_mode}，仅支持: kalman / raw / cv_fit"
        )

    if output_xy is None:
        x_hat = np.nan
        y_hat = np.nan
        y_error = np.nan
        abs_y_error = np.nan
        fit_ok = 0
    else:
        x_hat = float(output_xy[0])
        y_hat = float(output_xy[1])
        y_error = y_hat - y_gt
        abs_y_error = abs(y_error)
        fit_ok = 1

    row = {
        "Frame": fid,
        "gid": gid,
        "model": model,
        "side": side_info["name"],
        "n_roi": n_roi,
        "used_measurement": used_measurement,
        "track_exists": track_exists,
        "method": estimation_mode,
        "fit_points": fit_points,
        "x_gt": x_gt,
        "y_gt": y_gt,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "y_error": y_error,
        "abs_y_error": abs_y_error,
        "ok": fit_ok,
    }

    vis = {
        "gid": gid,
        "model": model,
        "gx": gx,
        "gy": gy,
        "yaw_rad": yaw_rad,
        "length": length,
        "width": width,
        "side_info": side_info,
        "outward_sign": outward_sign,
        "roi_pts": roi_pts,
        "n_roi": n_roi,
        "used_measurement": used_measurement,
        "x_gt": x_gt,
        "y_gt": y_gt,
        "x_hat": x_hat,
        "y_hat": y_hat,
        "ok": fit_ok,
        "method": estimation_mode,
        "fit_points": fit_points,
    }

    return row, vis


def run_analysis(radar_data, gt_df, frame_ids, params):
    tracks = {}
    rows = []
    frame_cache = {}

    # 仅给 cv_fit 用：每个 gid 维护一个历史原始量测窗口
    fit_histories = defaultdict(
        lambda: deque(maxlen=params.get("CVFIT_HISTORY_MAXLEN", 20))
    )

    for fid in frame_ids:
        meas_df = radar_data[fid].copy()
        gt_frame_df = gt_df[gt_df["Frame"].astype(int) == fid].copy()

        print(f"\n===== Frame {fid} =====")
        vis_targets = []

        for gt_row in gt_frame_df.itertuples(index=False):
            row, vis = process_one_target(
                fid=fid,
                gt_row=gt_row,
                meas_df=meas_df,
                tracks=tracks,
                fit_histories=fit_histories,
                params=params,
            )
            if row is None:
                continue

            rows.append(row)
            vis_targets.append(vis)

            if np.isfinite(row["y_hat"]):
                print(
                    f"gid={row['gid']}, method={row['method']}, side={row['side']}, "
                    f"n_roi={row['n_roi']}, meas_used={row['used_measurement']}, "
                    f"fit_points={row['fit_points']}, "
                    f"y_gt={row['y_gt']:.3f}, y_hat={row['y_hat']:.3f}, err={row['y_error']:.3f}"
                )
            else:
                print(
                    f"gid={row['gid']}, method={row['method']}, side={row['side']}, "
                    f"n_roi={row['n_roi']}, meas_used={row['used_measurement']}, "
                    f"fit_points={row['fit_points']}, "
                    f"y_gt={row['y_gt']:.3f}, y_hat=NaN"
                )

        frame_cache[fid] = {
            "meas_df": meas_df,
            "gt_frame_df": gt_frame_df,
            "targets": vis_targets,
        }

    return pd.DataFrame(rows), frame_cache


def print_summary(df, params=None):
    estimation_mode = "kalman" if params is None else params.get("ESTIMATION_MODE", "kalman").lower()

    if estimation_mode == "kalman":
        print("\n===== 尾部 ROI 点 + Kalman CV 滤波结果 =====")
    elif estimation_mode == "raw":
        print("\n===== 尾部 ROI 点原始量测结果（无 Kalman） =====")
    elif estimation_mode == "cv_fit":
        print("\n===== 尾部 ROI 点 + 滑动窗口匀速拟合结果 =====")
    else:
        print(f"\n===== 结果汇总（mode={estimation_mode}） =====")

    if df.empty:
        print("没有结果。")
        return

    ok_df = df[df["ok"] == 1].copy()

    print(f"\n总记录数: {len(df)}")
    print(f"有效估计数: {len(ok_df)}")

    if len(ok_df) == 0:
        print("没有有效估计。")
        return

    mae = ok_df["abs_y_error"].mean()
    medae = ok_df["abs_y_error"].median()
    rmse = np.sqrt(np.mean(ok_df["y_error"].values ** 2))
    bias = ok_df["y_error"].mean()

    print("\n===== 全局误差统计 =====")
    print(f"MAE   = {mae:.4f} m")
    print(f"MedAE = {medae:.4f} m")
    print(f"RMSE  = {rmse:.4f} m")
    print(f"Bias  = {bias:.4f} m")

    print("\n===== 每帧平均绝对误差 =====")
    frame_group = ok_df.groupby("Frame")["abs_y_error"].mean()
    for fid, val in frame_group.items():
        print(f"Frame {fid}: mean_abs_y_error = {val:.4f} m")
