import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from geometry import world_to_local
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

    if len(np.unique(frames)) < min_points:
        return None

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

    x_gt, y_gt = geometry["midpoint_world"]

    roi_pts, roi_mask = get_roi_points(
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
        "roi_mask": np.asarray(roi_mask, dtype=bool),
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


def _safe_normalize(values, near_is_max=True):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.empty(0, dtype=float)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    if span <= 1e-9:
        return np.full(values.shape, 0.5, dtype=float)

    if near_is_max:
        norm = (vmax - values) / span
    else:
        norm = (values - vmin) / span
    return norm.astype(float)


def _majority_non_noise_label(labels):
    labels = np.asarray(labels)
    labels = labels[labels >= 0]
    if labels.size == 0:
        return None
    uniq, counts = np.unique(labels, return_counts=True)
    return int(uniq[np.argmax(counts)])


def analyze_cluster_roi_core_metrics(
    frame_ids,
    frame_cache,
    num_u_bins=20,
    candidate_windows=None,
    success_threshold=0.5,
):
    """
    只保留最核心的 4 个指标：
    1) u: 点在主 cluster 内的纵向相对位置
    2) P(ROI | u_bin): 位置到 ROI 概率的映射
    3) Coverage(W): 给定候选窗口 W 时，对 ROI 的覆盖率
    4) Target Success Rate(W, tau): 对目标级别的稳定性评估

    返回:
        point_df: 仅保留点级核心字段 [Frame, gid, model, main_cluster, is_roi, u]
        prob_df:  按 u bin 统计的概率表
        window_df: 各候选窗口的整体评估
    """
    if candidate_windows is None:
        candidate_windows = [
            {"name": "front_20", "intervals": [(0.00, 0.20)]},
            {"name": "front_30", "intervals": [(0.00, 0.30)]},
            {"name": "dual_peak", "intervals": [(0.05, 0.25), (0.50, 0.60)]},
        ]

    point_rows = []
    target_cover_rows = []

    for fid in frame_ids:
        cache = frame_cache.get(fid)
        if cache is None:
            continue

        meas_df = cache.get("meas_df")
        labels = cache.get("cluster_labels")
        targets = cache.get("targets", [])

        if meas_df is None or labels is None or len(meas_df) == 0:
            continue

        pts_world = meas_df[["X", "Y"]].values.astype(float)
        labels = np.asarray(labels)
        if len(labels) != len(pts_world):
            raise ValueError(f"Frame {fid}: cluster_labels 与 meas_df 长度不一致")

        for t in targets:
            roi_mask = np.asarray(
                t.get("roi_mask", np.zeros(len(pts_world), dtype=bool)),
                dtype=bool,
            )
            if roi_mask.shape[0] != len(pts_world):
                raise ValueError(f"Frame {fid}, gid={t['gid']}: roi_mask 长度与点数不一致")

            roi_labels = labels[roi_mask]
            main_cluster = _majority_non_noise_label(roi_labels)
            if main_cluster is None:
                continue

            cluster_mask = labels == main_cluster
            cluster_pts_world = pts_world[cluster_mask]
            cluster_roi_mask = roi_mask[cluster_mask]
            if cluster_pts_world.shape[0] == 0:
                continue

            cluster_local = world_to_local(
                cluster_pts_world,
                t["gx"],
                t["gy"],
                t["yaw_rad"],
            )
            local_y = cluster_local[:, 1]
            near_axis = float(t["outward_sign"]) * local_y
            u = _safe_normalize(near_axis, near_is_max=True)

            for idx in range(len(u)):
                point_rows.append({
                    "Frame": int(fid),
                    "gid": int(t["gid"]),
                    "model": int(t["model"]),
                    "main_cluster": int(main_cluster),
                    "is_roi": int(cluster_roi_mask[idx]),
                    "u": float(u[idx]),
                })

            roi_u = u[cluster_roi_mask]
            n_roi = int(roi_u.size)
            if n_roi == 0:
                continue

            for window in candidate_windows:
                cover = _coverage_for_u(roi_u, window["intervals"])
                target_cover_rows.append({
                    "Frame": int(fid),
                    "gid": int(t["gid"]),
                    "model": int(t["model"]),
                    "window_name": str(window["name"]),
                    "coverage": float(cover),
                    "success": int(cover >= float(success_threshold)),
                })

    point_df = pd.DataFrame(point_rows)
    target_cover_df = pd.DataFrame(target_cover_rows)

    if point_df.empty:
        prob_df = pd.DataFrame(columns=[
            "u_bin", "u_left", "u_right", "n_points", "n_roi_points", "p_roi_given_u"
        ])
        window_df = pd.DataFrame(columns=[
            "window_name", "intervals", "mean_coverage", "median_coverage",
            "target_success_rate", "n_targets"
        ])
        return point_df, prob_df, window_df

    edges = np.linspace(0.0, 1.0, int(num_u_bins) + 1)
    point_df = point_df.copy()
    point_df["u_bin"] = np.digitize(point_df["u"].values, edges[1:-1], right=False).astype(int)

    prob_rows = []
    for bin_idx in range(len(edges) - 1):
        mask = point_df["u_bin"].values == bin_idx
        n_points = int(np.sum(mask))
        n_roi_points = int(point_df.loc[mask, "is_roi"].sum()) if n_points > 0 else 0
        p_roi = (n_roi_points / n_points) if n_points > 0 else np.nan
        prob_rows.append({
            "u_bin": int(bin_idx),
            "u_left": float(edges[bin_idx]),
            "u_right": float(edges[bin_idx + 1]),
            "n_points": n_points,
            "n_roi_points": n_roi_points,
            "p_roi_given_u": p_roi,
        })
    prob_df = pd.DataFrame(prob_rows)

    window_rows = []
    if not target_cover_df.empty:
        interval_map = {str(w["name"]): str(w["intervals"]) for w in candidate_windows}
        for window_name, group in target_cover_df.groupby("window_name"):
            window_rows.append({
                "window_name": str(window_name),
                "intervals": interval_map.get(str(window_name), ""),
                "mean_coverage": float(group["coverage"].mean()),
                "median_coverage": float(group["coverage"].median()),
                "target_success_rate": float(group["success"].mean()),
                "n_targets": int(len(group)),
            })
    window_df = pd.DataFrame(window_rows)

    return point_df, prob_df, window_df


def _coverage_for_u(roi_u, intervals):
    roi_u = np.asarray(roi_u, dtype=float)
    if roi_u.size == 0:
        return np.nan

    keep_mask = np.zeros(roi_u.shape, dtype=bool)
    for left, right in intervals:
        keep_mask |= (roi_u >= float(left)) & (roi_u <= float(right))
    return float(np.mean(keep_mask))


def print_cluster_roi_core_metrics_summary(prob_df, window_df, success_threshold=0.5):
    print("\n===== Cluster 内 ROI 核心指标 =====")
    print("1) u: 点在主 cluster 内的纵向相对位置，0=近端，1=远端")
    print("2) P(ROI|u_bin): 点落在某个 u 区间时属于 ROI 的概率")
    print("3) Coverage(W): 候选窗口 W 对真实 ROI 的覆盖率")
    print(
        f"4) Target Success Rate(W, tau={success_threshold:.2f}): "
        "Coverage(W) 达到阈值 tau 的目标占比"
    )

    if not prob_df.empty:
        print("\n===== P(ROI | u_bin) =====")
        for row in prob_df.itertuples(index=False):
            if row.n_points <= 0:
                continue
            print(
                f"u in [{row.u_left:.2f}, {row.u_right:.2f}): "
                f"n={row.n_points}, roi={row.n_roi_points}, P={row.p_roi_given_u:.4f}"
            )

    if not window_df.empty:
        print("\n===== Candidate Window Evaluation =====")
        for row in window_df.itertuples(index=False):
            print(
                f"{row.window_name}: intervals={row.intervals}, "
                f"mean_coverage={row.mean_coverage:.4f}, "
                f"median_coverage={row.median_coverage:.4f}, "
                f"target_success_rate={row.target_success_rate:.4f}, "
                f"n_targets={row.n_targets}"
            )


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
