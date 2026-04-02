import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from geometry import world_to_local
from roi_analysis import resolve_target_side_geometry, get_roi_points
from tracker_logic import measurement_from_roi_points, update_track
"""这个文件实现了基于概率加权的聚类 ROI 评估分析工具函数，供 main2.py 调用。"""

# =========================
# 基础工具
# =========================

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


def _majority_non_noise_label(labels):
    labels = np.asarray(labels)
    labels = labels[labels >= 0]
    if labels.size == 0:
        return None
    uniq, counts = np.unique(labels, return_counts=True)
    return int(uniq[np.argmax(counts)])


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


# =========================
# 概率表与加权量测
# =========================

def load_probability_lookup(prob_csv_path, min_weight=0.0):
    """
    从 cluster_roi_probability_by_u.csv 读取概率表。

    需要列:
        u_left, u_right, p_roi_given_u
    """
    prob_df = pd.read_csv(prob_csv_path)
    required_cols = {"u_left", "u_right", "p_roi_given_u"}
    missing = required_cols.difference(prob_df.columns)
    if missing:
        raise ValueError(f"概率表缺少字段: {sorted(missing)}")

    prob_df = prob_df[["u_left", "u_right", "p_roi_given_u"]].copy()
    prob_df["u_left"] = prob_df["u_left"].astype(float)
    prob_df["u_right"] = prob_df["u_right"].astype(float)
    prob_df["p_roi_given_u"] = prob_df["p_roi_given_u"].astype(float)
    prob_df["p_roi_given_u"] = prob_df["p_roi_given_u"].fillna(0.0).clip(lower=float(min_weight))
    prob_df = prob_df.sort_values(["u_left", "u_right"]).reset_index(drop=True)
    return prob_df


def lookup_probability_from_u(u_values, prob_df):
    """
    对每个 u 值查表，返回对应的 P(ROI|u_bin)。
    规则:
      - 优先使用 [u_left, u_right) 区间
      - 最后一个 bin 允许包含右端点 1.0
      - 越界时 clip 到 [0, 1]
    """
    u = np.asarray(u_values, dtype=float)
    if u.size == 0:
        return np.empty(0, dtype=float)

    u = np.clip(u, 0.0, 1.0)
    lefts = prob_df["u_left"].values.astype(float)
    rights = prob_df["u_right"].values.astype(float)
    probs = prob_df["p_roi_given_u"].values.astype(float)

    weights = np.zeros_like(u, dtype=float)
    assigned = np.zeros_like(u, dtype=bool)

    for idx in range(len(prob_df)):
        if idx == len(prob_df) - 1:
            mask = (u >= lefts[idx]) & (u <= rights[idx])
        else:
            mask = (u >= lefts[idx]) & (u < rights[idx])
        weights[mask] = probs[idx]
        assigned[mask] = True

    if not np.all(assigned):
        # 理论上不会走到，保底用最近 bin
        centers = 0.5 * (lefts + rights)
        miss_idx = np.where(~assigned)[0]
        for i in miss_idx:
            nearest = int(np.argmin(np.abs(centers - u[i])))
            weights[i] = probs[nearest]

    return weights


def weighted_mean_xy(points_xy, weights, min_weight_sum=1e-9):
    points = np.asarray(points_xy, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if points.shape[0] == 0 or w.shape[0] != points.shape[0]:
        return None

    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= float(min_weight_sum):
        return None

    out = np.sum(points * w[:, None], axis=0) / s
    return np.asarray(out, dtype=float).reshape(2)


def weighted_median_1d(values, weights):
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or weights.size != values.size:
        return np.nan

    weights = np.clip(weights, 0.0, None)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.nan

    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cdf = np.cumsum(w_sorted) / total
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(max(idx, 0), len(v_sorted) - 1)
    return float(v_sorted[idx])


def weighted_median_xy(points_xy, weights):
    points = np.asarray(points_xy, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if points.shape[0] == 0 or w.shape[0] != points.shape[0]:
        return None

    x = weighted_median_1d(points[:, 0], w)
    y = weighted_median_1d(points[:, 1], w)
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return np.array([x, y], dtype=float)


def apply_prob_weights_and_measure(
    cluster_pts_world,
    gx,
    gy,
    yaw_rad,
    outward_sign,
    prob_df,
    params,
):
    """
    在一个主 cluster 内:
      1) 用 GT 局部系定义 near_axis
      2) 归一化得到 u
      3) 查表得到权重 weight = P(ROI|u)
      4) 按权重做量测

    返回 dict:
      {
        "z": np.array([x,y]) or None,
        "u": ..., "weights": ..., "points": ...,
        "selected_mask": ..., "weight_sum": ..., "n_selected": ...
      }
    """
    pts = np.asarray(cluster_pts_world, dtype=float)
    if pts.shape[0] == 0:
        return {
            "z": None,
            "u": np.empty(0, dtype=float),
            "weights": np.empty(0, dtype=float),
            "points": pts,
            "selected_mask": np.zeros(0, dtype=bool),
            "weight_sum": 0.0,
            "n_selected": 0,
        }

    local_pts = world_to_local(pts, gx, gy, yaw_rad)
    local_y = local_pts[:, 1]
    near_axis = float(outward_sign) * local_y
    u = _safe_normalize(near_axis, near_is_max=True)
    weights = lookup_probability_from_u(u, prob_df)

    weight_threshold = float(params.get("WEIGHT_KEEP_THRESHOLD", 0.0))
    selected_mask = weights >= weight_threshold
    selected_pts = pts[selected_mask]
    selected_w = weights[selected_mask]

    if selected_pts.shape[0] == 0:
        # 如果阈值太严，把所有点回退进来
        selected_mask = np.ones(len(pts), dtype=bool)
        selected_pts = pts
        selected_w = weights

    measurement_mode = str(params.get("WEIGHTED_MEASUREMENT_MODE", "weighted_mean")).lower()
    if measurement_mode == "weighted_mean":
        z = weighted_mean_xy(
            selected_pts,
            selected_w,
            min_weight_sum=float(params.get("WEIGHT_MIN_SUM", 1e-6)),
        )
    elif measurement_mode == "weighted_median":
        z = weighted_median_xy(selected_pts, selected_w)
    else:
        raise ValueError(
            f"未知 WEIGHTED_MEASUREMENT_MODE={measurement_mode}，"
            "仅支持: weighted_mean / weighted_median"
        )

    return {
        "z": z,
        "u": u,
        "weights": weights,
        "points": pts,
        "selected_mask": selected_mask,
        "weight_sum": float(np.sum(selected_w)),
        "n_selected": int(np.sum(selected_mask)),
    }


# =========================
# 单目标与批处理评估
# =========================

def process_one_target_prob_weighted(
    fid,
    gt_row,
    meas_df,
    cluster_labels,
    tracks,
    fit_histories,
    prob_df,
    params,
):
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

    # 仅用于离线评估：用 GT-ROI 找主 cluster
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

    labels = np.asarray(cluster_labels)
    if len(labels) != len(meas_df):
        raise ValueError(f"Frame {fid}: cluster_labels 与 meas_df 长度不一致")

    roi_labels = labels[np.asarray(roi_mask, dtype=bool)]
    main_cluster = _majority_non_noise_label(roi_labels)

    raw_weighted_z = None
    weight_sum = 0.0
    n_cluster = 0
    n_selected = 0
    track_exists = 0
    fit_points = 0

    if main_cluster is not None:
        pts_world = meas_df[["X", "Y"]].values.astype(float)
        cluster_mask = labels == main_cluster
        cluster_pts_world = pts_world[cluster_mask]
        n_cluster = int(cluster_pts_world.shape[0])

        weighted_info = apply_prob_weights_and_measure(
            cluster_pts_world=cluster_pts_world,
            gx=gx,
            gy=gy,
            yaw_rad=yaw_rad,
            outward_sign=outward_sign,
            prob_df=prob_df,
            params=params,
        )
        raw_weighted_z = weighted_info["z"]
        weight_sum = float(weighted_info["weight_sum"])
        n_selected = int(weighted_info["n_selected"])
    else:
        weighted_info = {
            "z": None,
            "u": np.empty(0, dtype=float),
            "weights": np.empty(0, dtype=float),
            "points": np.empty((0, 2), dtype=float),
            "selected_mask": np.zeros(0, dtype=bool),
            "weight_sum": 0.0,
            "n_selected": 0,
        }

    estimation_mode = str(params.get("ESTIMATION_MODE", "raw")).lower()
    output_xy = None
    used_measurement = 0

    if estimation_mode == "raw":
        output_xy = raw_weighted_z
        used_measurement = 1 if raw_weighted_z is not None else 0

    elif estimation_mode == "kalman":
        output_xy, used_measurement = update_track(tracks, gid, raw_weighted_z, params)
        track_exists = 1 if gid in tracks else 0

    elif estimation_mode == "cv_fit":
        history = fit_histories[gid]
        if raw_weighted_z is not None:
            history.append((fid, raw_weighted_z.copy()))
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
            f"未知 ESTIMATION_MODE={estimation_mode}，仅支持: raw / kalman / cv_fit"
        )

    if output_xy is None:
        x_hat = np.nan
        y_hat = np.nan
        y_error = np.nan
        abs_y_error = np.nan
        ok = 0
    else:
        x_hat = float(output_xy[0])
        y_hat = float(output_xy[1])
        y_error = y_hat - y_gt
        abs_y_error = abs(y_error)
        ok = 1

    row = {
        "Frame": int(fid),
        "gid": gid,
        "model": model,
        "main_cluster": np.nan if main_cluster is None else int(main_cluster),
        "method": f"prob_weighted_{estimation_mode}",
        "n_roi": n_roi,
        "n_cluster": n_cluster,
        "n_selected": n_selected,
        "weight_sum": weight_sum,
        "used_measurement": used_measurement,
        "track_exists": track_exists,
        "fit_points": fit_points,
        "x_gt": float(x_gt),
        "y_gt": float(y_gt),
        "x_hat": x_hat,
        "y_hat": y_hat,
        "y_error": y_error,
        "abs_y_error": abs_y_error,
        "ok": ok,
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
        "x_gt": float(x_gt),
        "y_gt": float(y_gt),
        "x_hat": x_hat,
        "y_hat": y_hat,
        "ok": ok,
        "method": f"prob_weighted_{estimation_mode}",
        "fit_points": fit_points,
        "cluster_points": weighted_info["points"],
        "cluster_u": weighted_info["u"],
        "cluster_weights": weighted_info["weights"],
        "selected_mask": weighted_info["selected_mask"],
        "main_cluster": np.nan if main_cluster is None else int(main_cluster),
        "n_cluster": n_cluster,
        "n_selected": n_selected,
        "weight_sum": weight_sum,
    }

    return row, vis


def run_prob_weighted_analysis(radar_data, gt_df, frame_ids, cluster_label_map, prob_df, params):
    tracks = {}
    rows = []
    frame_cache = {}
    fit_histories = defaultdict(lambda: deque(maxlen=params.get("CVFIT_HISTORY_MAXLEN", 20)))

    for fid in frame_ids:
        meas_df = radar_data[fid].copy()
        gt_frame_df = gt_df[gt_df["Frame"].astype(int) == fid].copy()
        cluster_labels = np.asarray(cluster_label_map[fid])

        print(f"\n===== Frame {fid} =====")
        vis_targets = []

        for gt_row in gt_frame_df.itertuples(index=False):
            row, vis = process_one_target_prob_weighted(
                fid=fid,
                gt_row=gt_row,
                meas_df=meas_df,
                cluster_labels=cluster_labels,
                tracks=tracks,
                fit_histories=fit_histories,
                prob_df=prob_df,
                params=params,
            )
            if row is None:
                continue

            rows.append(row)
            vis_targets.append(vis)

            if np.isfinite(row["y_hat"]):
                print(
                    f"gid={row['gid']}, cluster={row['main_cluster']}, "
                    f"n_cluster={row['n_cluster']}, n_sel={row['n_selected']}, "
                    f"w_sum={row['weight_sum']:.3f}, "
                    f"y_gt={row['y_gt']:.3f}, y_hat={row['y_hat']:.3f}, err={row['y_error']:.3f}"
                )
            else:
                print(
                    f"gid={row['gid']}, cluster={row['main_cluster']}, "
                    f"n_cluster={row['n_cluster']}, n_sel={row['n_selected']}, "
                    f"w_sum={row['weight_sum']:.3f}, y_gt={row['y_gt']:.3f}, y_hat=NaN"
                )

        frame_cache[fid] = {
            "meas_df": meas_df,
            "gt_frame_df": gt_frame_df,
            "targets": vis_targets,
            "cluster_labels": cluster_labels,
        }

    return pd.DataFrame(rows), frame_cache


# =========================
# 汇总打印
# =========================

def print_summary(df, title="概率加权 cluster 量测结果"):
    print(f"\n===== {title} =====")

    if df.empty:
        print("没有结果。")
        return

    ok_df = df[df["ok"] == 1].copy()
    print(f"总记录数: {len(df)}")
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

    print("\n===== 选点统计 =====")
    print(f"平均主 cluster 点数 = {ok_df['n_cluster'].mean():.3f}")
    print(f"平均保留点数       = {ok_df['n_selected'].mean():.3f}")
    print(f"平均权重和         = {ok_df['weight_sum'].mean():.3f}")

    print("\n===== 每帧平均绝对误差 =====")
    frame_group = ok_df.groupby("Frame")["abs_y_error"].mean()
    for fid, val in frame_group.items():
        print(f"Frame {fid}: mean_abs_y_error = {val:.4f} m")
