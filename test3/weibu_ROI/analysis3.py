import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from roi_analysis import resolve_target_side_geometry, get_roi_points
from tracker_logic import update_track


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
    返回: np.array([x_hat, y_hat]) 或 None
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
# 全局纵轴 u
# =========================

def compute_u_from_global_y(cluster_pts_world, sensor_y=0.0):
    """
    仅使用全局纵轴 Y 定义 cluster 内部位置 u。

    规则:
      1) 不依赖 GT yaw / 局部系 / PCA
      2) 只看 cluster 中各点的 Y 值
      3) 比较 cluster 在 y_min 与 y_max 两端谁更靠近 sensor_y
      4) 更靠近传感器的一端定义为 u=0，另一端定义为 u=1
    """
    pts = np.asarray(cluster_pts_world, dtype=float)
    if pts.shape[0] == 0:
        return {
            "u": np.empty(0, dtype=float),
            "y_values": np.empty(0, dtype=float),
            "near_y_value": np.nan,
            "far_y_value": np.nan,
            "near_end_is_ymax": False,
        }

    y_values = pts[:, 1].astype(float)
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))

    dist_min = abs(y_min - float(sensor_y))
    dist_max = abs(y_max - float(sensor_y))
    near_end_is_ymax = dist_max < dist_min

    if near_end_is_ymax:
        u = _safe_normalize(y_values, near_is_max=True)
        near_y_value = y_max
        far_y_value = y_min
    else:
        u = _safe_normalize(y_values, near_is_max=False)
        near_y_value = y_min
        far_y_value = y_max

    return {
        "u": u,
        "y_values": y_values,
        "near_y_value": float(near_y_value),
        "far_y_value": float(far_y_value),
        "near_end_is_ymax": bool(near_end_is_ymax),
    }


# =========================
# Cluster 摘要与最近邻关联
# =========================

def build_cluster_summaries(meas_df, cluster_labels, center_mode="median"):
    """
    把每个非噪声 cluster 压成一个摘要:
      - label
      - 中心(cx, cy)
      - 点数
      - 原始点集
    """
    labels = np.asarray(cluster_labels)
    pts_world = meas_df[["X", "Y"]].values.astype(float)

    if len(labels) != len(pts_world):
        raise ValueError("cluster_labels 与 meas_df 长度不一致")

    cluster_summaries = []

    for label in sorted(np.unique(labels)):
        if int(label) < 0:
            continue

        mask = labels == label
        cluster_pts = pts_world[mask]
        if cluster_pts.shape[0] == 0:
            continue

        if str(center_mode).lower() == "mean":
            cx = float(np.mean(cluster_pts[:, 0]))
            cy = float(np.mean(cluster_pts[:, 1]))
        else:
            cx = float(np.median(cluster_pts[:, 0]))
            cy = float(np.median(cluster_pts[:, 1]))

        cluster_summaries.append({
            "label": int(label),
            "cx": cx,
            "cy": cy,
            "n_points": int(cluster_pts.shape[0]),
            "points": cluster_pts,
        })

    return cluster_summaries


def _get_reference_position(gt_row, assoc_reference="gt_center", predicted_positions=None):
    """
    获取目标关联时的参考位置:
      - gt_center: 当前帧 GT 中心
      - predicted: 轨迹预测位置（如果存在）
    """
    gid = int(gt_row.ID)

    if assoc_reference == "predicted" and predicted_positions is not None and gid in predicted_positions:
        px, py = predicted_positions[gid]
        return float(px), float(py)

    return float(gt_row.X), float(gt_row.Y)


def associate_targets_to_clusters_nearest(
    gt_frame_df,
    cluster_summaries,
    params,
    predicted_positions=None,
):
    """
    最近邻 + 门控 + 未占用排除 的目标-cluster关联。

    返回:
        association_map: {gid: cluster_label or None}
        assoc_debug_rows: 每个候选 pair 的距离与代价信息
    """
    assoc_reference = str(params.get("ASSOC_REFERENCE", "gt_center")).lower()

    x_gate = float(params.get("ASSOC_GATE_X", 3.0))
    y_gate = float(params.get("ASSOC_GATE_Y", 6.0))

    alpha_x = float(params.get("ASSOC_DIST_WEIGHT_X", 1.0))
    alpha_y = float(params.get("ASSOC_DIST_WEIGHT_Y", 2.0))

    rows = []
    target_ids = []

    gt_rows = list(gt_frame_df.itertuples(index=False))
    for gt_row in gt_rows:
        gid = int(gt_row.ID)
        rx, ry = _get_reference_position(
            gt_row=gt_row,
            assoc_reference=assoc_reference,
            predicted_positions=predicted_positions,
        )
        target_ids.append(gid)

        for clu in cluster_summaries:
            dx = float(clu["cx"] - rx)
            dy = float(clu["cy"] - ry)

            in_gate = (abs(dx) <= x_gate) and (abs(dy) <= y_gate)
            if not in_gate:
                continue

            cost = alpha_x * abs(dx) + alpha_y * abs(dy)

            rows.append({
                "gid": gid,
                "ref_x": rx,
                "ref_y": ry,
                "cluster_label": int(clu["label"]),
                "cluster_cx": float(clu["cx"]),
                "cluster_cy": float(clu["cy"]),
                "dx": dx,
                "dy": dy,
                "cost": float(cost),
            })

    rows = sorted(rows, key=lambda r: r["cost"])

    association_map = {gid: None for gid in target_ids}
    used_clusters = set()
    assigned_targets = set()

    for row in rows:
        gid = row["gid"]
        label = row["cluster_label"]

        if gid in assigned_targets:
            continue
        if label in used_clusters:
            continue

        association_map[gid] = label
        assigned_targets.add(gid)
        used_clusters.add(label)

    assoc_debug_rows = pd.DataFrame(rows)
    return association_map, assoc_debug_rows


def _build_predicted_position_map_from_tracks(tracks):
    """
    从已有 tracks 中取出当前预测参考位置。
    这里默认用 track.output_center 作为参考。
    """
    predicted = {}
    for gid, trk in tracks.items():
        center = getattr(trk, "output_center", None)
        if center is None:
            continue
        arr = np.asarray(center, dtype=float).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            predicted[int(gid)] = (float(arr[0]), float(arr[1]))
    return predicted


# =========================
# 概率表：生成 / 查询
# =========================

def generate_probability_table_global_y(
    radar_data,
    gt_df,
    frame_ids,
    cluster_label_map,
    params,
    num_u_bins=20,
):
    """
    用“最近邻主 cluster 关联 + 全局纵轴 Y 定义的 u”生成概率表。
    is_roi 仍来自 GT-ROI，u 来自主 cluster 内部的全局 Y 归一化位置。
    """
    point_rows = []
    sensor_y = float(params.get("SENSOR_Y", 0.0))
    center_mode = str(params.get("ASSOC_CLUSTER_CENTER_MODE", "median")).lower()

    for fid in frame_ids:
        meas_df = radar_data[fid].copy()
        gt_frame_df = gt_df[gt_df["Frame"].astype(int) == fid].copy()
        cluster_labels = np.asarray(cluster_label_map[fid])

        pts_world = meas_df[["X", "Y"]].values.astype(float)
        if len(cluster_labels) != len(pts_world):
            raise ValueError(f"Frame {fid}: cluster_labels 与 meas_df 长度不一致")

        cluster_summaries = build_cluster_summaries(
            meas_df=meas_df,
            cluster_labels=cluster_labels,
            center_mode=center_mode,
        )

        association_map, _assoc_debug = associate_targets_to_clusters_nearest(
            gt_frame_df=gt_frame_df,
            cluster_summaries=cluster_summaries,
            params=params,
            predicted_positions=None,
        )

        for gt_row in gt_frame_df.itertuples(index=False):
            gid = int(gt_row.ID)
            model = int(gt_row.model)

            gt_dim = params["GT_DIM"]
            if model not in gt_dim:
                continue

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

            _roi_pts, roi_mask = get_roi_points(
                meas_df=meas_df,
                gx=gx,
                gy=gy,
                yaw_rad=yaw_rad,
                side_info=side_info,
                outward_sign=outward_sign,
                outer_margin=params["ROI_OUTER"],
                inner_margin=params["ROI_INNER"],
            )

            main_cluster = association_map.get(gid, None)
            if main_cluster is None:
                continue

            cluster_mask = cluster_labels == int(main_cluster)
            cluster_pts_world = pts_world[cluster_mask]
            cluster_roi_mask = np.asarray(roi_mask, dtype=bool)[cluster_mask]

            yaxis_info = compute_u_from_global_y(
                cluster_pts_world=cluster_pts_world,
                sensor_y=sensor_y,
            )
            u = yaxis_info["u"]

            for idx in range(cluster_pts_world.shape[0]):
                point_rows.append({
                    "Frame": int(fid),
                    "gid": gid,
                    "model": model,
                    "main_cluster": int(main_cluster),
                    "is_roi": int(cluster_roi_mask[idx]),
                    "u": float(u[idx]),
                })

    point_df = pd.DataFrame(point_rows)

    if point_df.empty:
        prob_df = pd.DataFrame(columns=[
            "u_bin", "u_left", "u_right", "n_points", "n_roi_points", "p_roi_given_u"
        ])
        return point_df, prob_df

    edges = np.linspace(0.0, 1.0, int(num_u_bins) + 1)
    bin_ids = np.digitize(point_df["u"].values, edges[1:-1], right=False)
    point_df = point_df.copy()
    point_df["u_bin"] = bin_ids.astype(int)

    bin_rows = []
    for bin_idx in range(len(edges) - 1):
        mask = point_df["u_bin"].values == bin_idx
        n_points = int(np.sum(mask))
        n_roi_points = int(point_df.loc[mask, "is_roi"].sum()) if n_points > 0 else 0
        p_roi = (n_roi_points / n_points) if n_points > 0 else np.nan
        bin_rows.append({
            "u_bin": int(bin_idx),
            "u_left": float(edges[bin_idx]),
            "u_right": float(edges[bin_idx + 1]),
            "n_points": n_points,
            "n_roi_points": n_roi_points,
            "p_roi_given_u": p_roi,
        })

    prob_df = pd.DataFrame(bin_rows)
    return point_df, prob_df


def normalize_probability_table(prob_df, min_weight=0.0):
    required_cols = {"u_left", "u_right", "p_roi_given_u"}
    missing = required_cols.difference(prob_df.columns)
    if missing:
        raise ValueError(f"概率表缺少字段: {sorted(missing)}")

    out = prob_df[["u_left", "u_right", "p_roi_given_u"]].copy()
    out["u_left"] = out["u_left"].astype(float)
    out["u_right"] = out["u_right"].astype(float)
    out["p_roi_given_u"] = out["p_roi_given_u"].astype(float)
    out["p_roi_given_u"] = out["p_roi_given_u"].fillna(0.0).clip(lower=float(min_weight))
    out = out.sort_values(["u_left", "u_right"]).reset_index(drop=True)
    return out


def lookup_probability_from_u(u_values, prob_df):
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
        centers = 0.5 * (lefts + rights)
        miss_idx = np.where(~assigned)[0]
        for i in miss_idx:
            nearest = int(np.argmin(np.abs(centers - u[i])))
            weights[i] = probs[nearest]

    return weights


# =========================
# 概率加权量测
# =========================

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


def apply_prob_weights_and_measure_global_y(cluster_pts_world, prob_df, params):
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
            "near_y_value": np.nan,
            "far_y_value": np.nan,
            "near_end_is_ymax": False,
        }

    sensor_y = float(params.get("SENSOR_Y", 0.0))
    yaxis_info = compute_u_from_global_y(cluster_pts_world=pts, sensor_y=sensor_y)
    u = yaxis_info["u"]
    weights = lookup_probability_from_u(u, prob_df)

    weight_threshold = float(params.get("WEIGHT_KEEP_THRESHOLD", 0.0))
    selected_mask = weights >= weight_threshold
    selected_pts = pts[selected_mask]
    selected_w = weights[selected_mask]

    if selected_pts.shape[0] == 0:
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
            f"未知 WEIGHTED_MEASUREMENT_MODE={measurement_mode}，仅支持: weighted_mean / weighted_median"
        )

    return {
        "z": z,
        "u": u,
        "weights": weights,
        "points": pts,
        "selected_mask": selected_mask,
        "weight_sum": float(np.sum(selected_w)),
        "n_selected": int(np.sum(selected_mask)),
        "near_y_value": float(yaxis_info["near_y_value"]),
        "far_y_value": float(yaxis_info["far_y_value"]),
        "near_end_is_ymax": bool(yaxis_info["near_end_is_ymax"]),
    }


def _pick_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def extract_selected_measurement_points(
    fid,
    gid,
    model,
    main_cluster,
    meas_df,
    cluster_labels,
    weighted_info,
):
    """
    导出“真正参与量测生成”的点：
      - 在主 cluster 内
      - selected_mask == True

    如果该 cluster 没有点过阈值，则沿用原逻辑：
      selected_mask 会退化为全 True，
      即导出该 cluster 全部点。
    """
    if main_cluster is None:
        return pd.DataFrame()

    labels = np.asarray(cluster_labels)
    cluster_mask = labels == int(main_cluster)
    if np.sum(cluster_mask) == 0:
        return pd.DataFrame()

    cluster_df = meas_df.loc[cluster_mask].copy().reset_index(drop=True)

    selected_mask = np.asarray(weighted_info["selected_mask"], dtype=bool)
    weights = np.asarray(weighted_info["weights"], dtype=float)
    u = np.asarray(weighted_info["u"], dtype=float)

    if len(cluster_df) != len(selected_mask):
        raise ValueError(
            f"Frame {fid}, gid {gid}: cluster_df 长度与 selected_mask 长度不一致"
        )

    cluster_df["selected_for_measurement"] = selected_mask.astype(int)
    cluster_df["weight"] = weights
    cluster_df["u"] = u
    cluster_df["Frame"] = int(fid)
    cluster_df["gid"] = int(gid)
    cluster_df["model"] = int(model)
    cluster_df["main_cluster"] = int(main_cluster)

    speed_col = _pick_first_existing_column(
        cluster_df,
        ["speed", "Speed", "VEL", "Vel", "Velocity", "velocity", "v", "V"]
    )
    range_col = _pick_first_existing_column(
        cluster_df,
        ["range", "Range", "R", "range_m", "RangeM"]
    )
    angle_col = _pick_first_existing_column(
        cluster_df,
        ["angle", "Angle", "AZI", "Azi", "azimuth", "Azimuth"]
    )
    snr_col = _pick_first_existing_column(
        cluster_df,
        ["SNR", "snr"]
    )

    cluster_df["speed"] = cluster_df[speed_col] if speed_col is not None else np.nan
    cluster_df["range"] = cluster_df[range_col] if range_col is not None else np.nan
    cluster_df["angle"] = cluster_df[angle_col] if angle_col is not None else np.nan
    cluster_df["SNR"] = cluster_df[snr_col] if snr_col is not None else np.nan

    selected_df = cluster_df[cluster_df["selected_for_measurement"] == 1].copy()

    preferred_cols = [
        "Frame",
        "gid",
        "model",
        "main_cluster",
        "speed",
        "range",
        "angle",
        "SNR",
        "weight",
        "u",
        "X",
        "Y",
        "selected_for_measurement",
    ]
    other_cols = [c for c in selected_df.columns if c not in preferred_cols]
    selected_df = selected_df[preferred_cols + other_cols]

    return selected_df


# =========================
# 中心几何补偿
# =========================

def _get_center_comp_distance(model, length, params):
    """
    从 params 中读取中心补偿距离。
    支持三种模式:
      - constant
      - model_half_length
      - model_lookup
    """
    mode = str(params.get("CENTER_COMP_MODE", "constant")).lower()

    if mode == "constant":
        return float(params.get("CENTER_COMP_DISTANCE", 0.0))

    if mode == "model_half_length":
        alpha = float(params.get("CENTER_COMP_ALPHA", 1.0))
        return alpha * float(length) / 2.0

    if mode == "model_lookup":
        table = params.get("CENTER_COMP_BY_MODEL", {})
        if model not in table:
            raise ValueError(f"CENTER_COMP_BY_MODEL 中缺少 model={model} 的补偿值")
        return float(table[model])

    raise ValueError(
        f"未知 CENTER_COMP_MODE={mode}，仅支持: constant / model_half_length / model_lookup"
    )


def apply_center_compensation(x_hat, y_hat, near_end_is_ymax, model, length, params):
    """
    从当前近端代表点，沿全局Y方向补偿到中心点。
    x方向默认不补偿。
    """
    if x_hat is None or y_hat is None:
        return x_hat, y_hat, 0.0

    if not bool(params.get("ENABLE_CENTER_COMPENSATION", False)):
        return x_hat, y_hat, 0.0

    d = _get_center_comp_distance(model=model, length=length, params=params)

    if near_end_is_ymax:
        y_center = float(y_hat) - d
    else:
        y_center = float(y_hat) + d

    return float(x_hat), float(y_center), float(d)


# =========================
# 单目标处理
# =========================

def process_one_target_global_y_prob_weighted(
    fid,
    gt_row,
    meas_df,
    cluster_labels,
    main_cluster,
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

    # 误差真值使用 GT中心
    x_gt = gx
    y_gt = gy

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

    raw_weighted_z = None
    weight_sum = 0.0
    n_cluster = 0
    n_selected = 0
    track_exists = 0
    fit_points = 0

    if main_cluster is not None:
        pts_world = meas_df[["X", "Y"]].values.astype(float)
        cluster_mask = labels == int(main_cluster)
        cluster_pts_world = pts_world[cluster_mask]
        n_cluster = int(cluster_pts_world.shape[0])

        weighted_info = apply_prob_weights_and_measure_global_y(
            cluster_pts_world=cluster_pts_world,
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
            "near_y_value": np.nan,
            "far_y_value": np.nan,
            "near_end_is_ymax": False,
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
        raise ValueError(f"未知 ESTIMATION_MODE={estimation_mode}，仅支持: raw / kalman / cv_fit")

    if output_xy is None:
        x_hat = np.nan
        y_hat = np.nan
        center_comp_distance = 0.0
        x_error = np.nan
        y_error = np.nan
        abs_x_error = np.nan
        abs_y_error = np.nan
        xy_error = np.nan
        ok = 0
    else:
        x_hat = float(output_xy[0])
        y_hat = float(output_xy[1])

        # 先做中心几何补偿
        x_hat, y_hat, center_comp_distance = apply_center_compensation(
            x_hat=x_hat,
            y_hat=y_hat,
            near_end_is_ymax=bool(weighted_info["near_end_is_ymax"]),
            model=model,
            length=length,
            params=params,
        )

        x_error = x_hat - x_gt
        y_error = y_hat - y_gt
        abs_x_error = abs(x_error)
        abs_y_error = abs(y_error)
        xy_error = float(np.sqrt(x_error ** 2 + y_error ** 2))
        ok = 1

    row = {
        "Frame": int(fid),
        "gid": gid,
        "model": model,
        "main_cluster": np.nan if main_cluster is None else int(main_cluster),
        "method": f"global_y_prob_weighted_{estimation_mode}",
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

        "center_comp_distance": float(center_comp_distance),

        "x_error": x_error,
        "y_error": y_error,
        "abs_x_error": abs_x_error,
        "abs_y_error": abs_y_error,
        "xy_error": xy_error,

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

        "center_comp_distance": float(center_comp_distance),

        "x_error": x_error,
        "y_error": y_error,
        "abs_x_error": abs_x_error,
        "abs_y_error": abs_y_error,
        "xy_error": xy_error,

        "ok": ok,
        "method": f"global_y_prob_weighted_{estimation_mode}",
        "fit_points": fit_points,

        "cluster_points": weighted_info["points"],
        "cluster_u": weighted_info["u"],
        "cluster_weights": weighted_info["weights"],
        "selected_mask": weighted_info["selected_mask"],

        "main_cluster": np.nan if main_cluster is None else int(main_cluster),
        "n_cluster": n_cluster,
        "n_selected": n_selected,
        "weight_sum": weight_sum,
        "near_y_value": weighted_info["near_y_value"],
        "far_y_value": weighted_info["far_y_value"],
        "near_end_is_ymax": weighted_info["near_end_is_ymax"],
    }
    return row, vis


# =========================
# 批量评估
# =========================

def run_global_y_prob_weighted_analysis(radar_data, gt_df, frame_ids, cluster_label_map, prob_df, params):
    tracks = {}
    rows = []
    frame_cache = {}
    fit_histories = defaultdict(lambda: deque(maxlen=params.get("CVFIT_HISTORY_MAXLEN", 20)))
    selected_meas_rows = []

    center_mode = str(params.get("ASSOC_CLUSTER_CENTER_MODE", "median")).lower()

    for fid in frame_ids:
        meas_df = radar_data[fid].copy()
        gt_frame_df = gt_df[gt_df["Frame"].astype(int) == fid].copy()
        cluster_labels = np.asarray(cluster_label_map[fid])

        predicted_positions = None
        if str(params.get("ASSOC_REFERENCE", "gt_center")).lower() == "predicted":
            predicted_positions = _build_predicted_position_map_from_tracks(tracks)

        cluster_summaries = build_cluster_summaries(
            meas_df=meas_df,
            cluster_labels=cluster_labels,
            center_mode=center_mode,
        )

        association_map, _assoc_debug = associate_targets_to_clusters_nearest(
            gt_frame_df=gt_frame_df,
            cluster_summaries=cluster_summaries,
            params=params,
            predicted_positions=predicted_positions,
        )

        print(f"\n===== Frame {fid} =====")
        vis_targets = []

        for gt_row in gt_frame_df.itertuples(index=False):
            gid = int(gt_row.ID)
            model = int(gt_row.model)
            main_cluster = association_map.get(gid, None)

            row, vis = process_one_target_global_y_prob_weighted(
                fid=fid,
                gt_row=gt_row,
                meas_df=meas_df,
                cluster_labels=cluster_labels,
                main_cluster=main_cluster,
                tracks=tracks,
                fit_histories=fit_histories,
                prob_df=prob_df,
                params=params,
            )
            if row is None:
                continue

            rows.append(row)
            vis_targets.append(vis)

            # 导出“真正参与量测计算”的点
            if main_cluster is not None:
                weighted_info = {
                    "selected_mask": vis["selected_mask"],
                    "weights": vis["cluster_weights"],
                    "u": vis["cluster_u"],
                }

                selected_df = extract_selected_measurement_points(
                    fid=fid,
                    gid=gid,
                    model=model,
                    main_cluster=main_cluster,
                    meas_df=meas_df,
                    cluster_labels=cluster_labels,
                    weighted_info=weighted_info,
                )
                if not selected_df.empty:
                    selected_meas_rows.append(selected_df)

            if np.isfinite(row["xy_error"]):
                print(
                    f"gid={row['gid']}, cluster={row['main_cluster']}, "
                    f"n_cluster={row['n_cluster']}, n_sel={row['n_selected']}, "
                    f"w_sum={row['weight_sum']:.3f}, "
                    f"comp={row['center_comp_distance']:.3f}, "
                    f"x_gt={row['x_gt']:.3f}, x_hat={row['x_hat']:.3f}, x_err={row['x_error']:.3f}, "
                    f"y_gt={row['y_gt']:.3f}, y_hat={row['y_hat']:.3f}, y_err={row['y_error']:.3f}, "
                    f"xy_err={row['xy_error']:.3f}"
                )
            else:
                print(
                    f"gid={row['gid']}, cluster={row['main_cluster']}, "
                    f"n_cluster={row['n_cluster']}, n_sel={row['n_selected']}, "
                    f"w_sum={row['weight_sum']:.3f}, x_hat=NaN, y_hat=NaN"
                )

        frame_cache[fid] = {
            "meas_df": meas_df,
            "gt_frame_df": gt_frame_df,
            "targets": vis_targets,
            "cluster_labels": cluster_labels,
        }

    selected_meas_df = (
        pd.concat(selected_meas_rows, ignore_index=True)
        if len(selected_meas_rows) > 0
        else pd.DataFrame(columns=[
            "Frame", "gid", "model", "main_cluster",
            "speed", "range", "angle", "SNR",
            "weight", "u", "X", "Y", "selected_for_measurement"
        ])
    )

    return pd.DataFrame(rows), frame_cache, selected_meas_df


# =========================
# 打印统计
# =========================

def print_probability_summary(prob_df):
    print("\n===== 当前数据重新生成的概率表 =====")
    print(prob_df)
    if prob_df.empty:
        return

    valid = prob_df[prob_df["n_points"] > 0].copy()
    if valid.empty:
        return

    topk = valid.sort_values("p_roi_given_u", ascending=False).head(5)
    print("\n===== P(ROI|u) 最高的 5 个区间 =====")
    for row in topk.itertuples(index=False):
        print(
            f"u in [{row.u_left:.2f}, {row.u_right:.2f}): "
            f"n={row.n_points}, roi={row.n_roi_points}, P={row.p_roi_given_u:.4f}"
        )


def print_summary(df, params=None):
    estimation_mode = "raw" if params is None else params.get("ESTIMATION_MODE", "raw").lower()

    print(f"\n===== 全局纵轴概率加权结果（mode={estimation_mode}） =====")

    if df.empty:
        print("没有结果。")
        return

    ok_df = df[df["ok"] == 1].copy()
    meas_ok_df = df[(df["ok"] == 1) & (df["used_measurement"] == 1)].copy()
    pred_only_df = df[(df["ok"] == 1) & (df["used_measurement"] == 0)].copy()

    print(f"\n总记录数: {len(df)}")
    print(f"有效估计数(全部输出): {len(ok_df)}")
    print(f"有效估计数(量测更新帧): {len(meas_ok_df)}")
    print(f"有效估计数(纯预测帧): {len(pred_only_df)}")

    if len(ok_df) == 0:
        print("没有有效估计。")
        return

    _print_metric_block("全部有效输出统计（含预测帧）", ok_df)
    _print_metric_block("仅量测更新帧统计（used_measurement=1）", meas_ok_df)
    _print_metric_block("仅纯预测帧统计（used_measurement=0）", pred_only_df)

    print("\n===== 每帧误差统计（全部有效输出） =====")
    frame_group = ok_df.groupby("Frame").agg({
        "abs_x_error": "mean",
        "abs_y_error": "mean",
        "xy_error": "mean",
    })

    for fid, row in frame_group.iterrows():
        print(
            f"Frame {fid}: "
            f"mean_abs_x_error = {row['abs_x_error']:.4f} m, "
            f"mean_abs_y_error = {row['abs_y_error']:.4f} m, "
            f"mean_xy_error = {row['xy_error']:.4f} m"
        )


def _print_metric_block(title, stat_df):
    print(f"\n===== {title} =====")

    if stat_df.empty:
        print("没有结果。")
        return

    x_mae = stat_df["abs_x_error"].mean()
    x_medae = stat_df["abs_x_error"].median()
    x_rmse = np.sqrt(np.mean(stat_df["x_error"].values ** 2))
    x_bias = stat_df["x_error"].mean()

    y_mae = stat_df["abs_y_error"].mean()
    y_medae = stat_df["abs_y_error"].median()
    y_rmse = np.sqrt(np.mean(stat_df["y_error"].values ** 2))
    y_bias = stat_df["y_error"].mean()

    xy_mae = stat_df["xy_error"].mean()
    xy_medae = stat_df["xy_error"].median()
    xy_rmse = np.sqrt(np.mean(stat_df["xy_error"].values ** 2))

    acc_05 = np.mean(stat_df["xy_error"].values <= 0.5)
    acc_10 = np.mean(stat_df["xy_error"].values <= 1.0)
    acc_20 = np.mean(stat_df["xy_error"].values <= 2.0)

    print("\n----- X方向误差统计 -----")
    print(f"MAE   = {x_mae:.4f} m")
    print(f"MedAE = {x_medae:.4f} m")
    print(f"RMSE  = {x_rmse:.4f} m")
    print(f"Bias  = {x_bias:.4f} m")

    print("\n----- Y方向误差统计 -----")
    print(f"MAE   = {y_mae:.4f} m")
    print(f"MedAE = {y_medae:.4f} m")
    print(f"RMSE  = {y_rmse:.4f} m")
    print(f"Bias  = {y_bias:.4f} m")

    print("\n----- 二维整体定位误差统计 -----")
    print(f"XY-MAE   = {xy_mae:.4f} m")
    print(f"XY-MedAE = {xy_medae:.4f} m")
    print(f"XY-RMSE  = {xy_rmse:.4f} m")

    print("\n----- 二维定位命中率 -----")
    print(f"<= 0.5 m : {acc_05:.4f}")
    print(f"<= 1.0 m : {acc_10:.4f}")
    print(f"<= 2.0 m : {acc_20:.4f}")

    print("\n----- 选点统计 -----")
    print(f"平均 cluster 点数 = {stat_df['n_cluster'].mean():.4f}")
    print(f"平均保留点数     = {stat_df['n_selected'].mean():.4f}")
    print(f"平均权重和       = {stat_df['weight_sum'].mean():.4f}")
    print(f"平均中心补偿量   = {stat_df['center_comp_distance'].mean():.4f}")
