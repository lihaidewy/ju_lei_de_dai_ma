import numpy as np
import pandas as pd

from load_data2 import load_data
from plot_gt_main import load_gt_reference

import sys
from pathlib import Path

# 把项目根目录加入搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from eval_clusters2_multi_prior_v2 import eval_one_frame_target_level
from centers import compute_center_with_optional_velocity_filter


def load_all_data(cfg):
    gt_df = load_gt_reference(cfg.GT_PATH, H=6.0)
    radar_data = load_data(cfg.RADAR_PATH)
    return radar_data, gt_df


def get_frame_ids(radar_data, gt_df, cfg, args):
    radar_frames = sorted(radar_data.keys())
    gt_frames = sorted(gt_df["Frame"].unique())
    common_frames = sorted(set(radar_frames).intersection(gt_frames))

    if len(common_frames) == 0:
        raise ValueError(f"No common frames between {cfg.RADAR_PATH} and {cfg.GT_PATH}")

    if cfg.FRAMES_TO_SHOW is None:
        return common_frames[:args.max_frames]

    frame_ids = [fid for fid in cfg.FRAMES_TO_SHOW if fid in common_frames]
    if len(frame_ids) == 0:
        raise ValueError("None of FRAMES_TO_SHOW are in common_frames")
    return frame_ids


def build_gt_list_for_frame(gt_df: pd.DataFrame, fid: int):
    g = gt_df[gt_df["Frame"] == fid]
    return [
        {"id": int(r.ID), "x": float(r.X), "y": float(r.Y), "model": int(r.model)}
        for r in g.itertuples(index=False)
    ]


def cluster_one_frame(radar_data, fid: int, cfg):
    return cluster_frame_dbscan(
        radar_data,
        fid,
        eps_x=cfg.EPS_X,
        eps_y=cfg.EPS_Y,
        eps_v=cfg.EPS_V,
        min_pts=cfg.MIN_PTS,
    )


def build_cluster_centers(labels: np.ndarray, pts: np.ndarray, frame_item: dict, center_fn, bias_fn, cfg):
    """
    显式构建当前帧每个 cluster 的中心。
    返回：
        {cid: np.array([x, y])}
    """
    cluster_centers = {}

    for cid in np.unique(labels):
        if cid < 1:
            continue

        mask = (labels == cid)
        cpts = pts[mask]
        if cpts.size == 0:
            continue

        center = compute_center_with_optional_velocity_filter(
            cpts=cpts,
            frame_item=frame_item,
            mask=mask,
            center_fn=center_fn,
            cfg=cfg,
        )
        center = bias_fn(center, cfg)

        cluster_centers[int(cid)] = np.asarray(center, dtype=float)

    return cluster_centers


def temporal_filter_cluster_centers_with_matches(cluster_centers: dict, matches: list, tracker):
    """
    用当前帧匹配结果中的 gid 作为 track_id，
    对 cluster center 做时序滤波（EMA 或 Kalman 都可）。
    """
    filtered_centers = {}

    cid_to_gid = {}
    for mm in matches:
        cid_to_gid[int(mm["cid"])] = int(mm["gid"])

    for cid, center in cluster_centers.items():
        if cid in cid_to_gid:
            gid = cid_to_gid[cid]
            filtered_centers[cid] = tracker.update(gid, center)
        else:
            filtered_centers[cid] = np.asarray(center, dtype=float)

    return filtered_centers


def build_point_level_table_from_centers(
    fid: int,
    frame_item: pd.DataFrame,
    labels: np.ndarray,
    cluster_centers: dict,
    matches=None,
    track_assignments=None,
    raw_cluster_centers=None,
) -> pd.DataFrame:
    """
    用外部传入的 cluster_centers 构建点级表。
    保留原始表格列顺序，只在右侧追加结果列。

    新增:
    - track_id
    - Raw_Center_X / Raw_Center_Y
    """

    df = frame_item.copy().reset_index(drop=True)

    center_x = np.full(len(labels), np.nan, dtype=float)
    center_y = np.full(len(labels), np.nan, dtype=float)

    raw_center_x = np.full(len(labels), np.nan, dtype=float)
    raw_center_y = np.full(len(labels), np.nan, dtype=float)

    gid_arr = np.full(len(labels), np.nan, dtype=float)
    track_id_arr = np.full(len(labels), np.nan, dtype=float)

    cid_to_gid = {}
    if matches is not None:
        for mm in matches:
            cid_to_gid[int(mm["cid"])] = int(mm["gid"])

    for cid, center in cluster_centers.items():
        mask = (labels == cid)

        center_x[mask] = float(center[0])
        center_y[mask] = float(center[1])

        if cid in cid_to_gid:
            gid_arr[mask] = float(cid_to_gid[cid])

        if track_assignments is not None and cid in track_assignments:
            track_id_arr[mask] = float(track_assignments[cid])

        if raw_cluster_centers is not None and cid in raw_cluster_centers:
            raw_center_x[mask] = float(raw_cluster_centers[cid][0])
            raw_center_y[mask] = float(raw_cluster_centers[cid][1])

    # 只追加列
    df["Label"] = labels
    df["gid"] = gid_arr
    df["track_id"] = track_id_arr

    df["Raw_Center_X"] = raw_center_x
    df["Raw_Center_Y"] = raw_center_y

    df["Center_X"] = center_x
    df["Center_Y"] = center_y

    # gid / track_id 转 int（保留 NaN）
    for col in ["gid", "track_id"]:
        if col in df.columns:
            valid_mask = df[col].notna()
            if valid_mask.any():
                df.loc[valid_mask, col] = df.loc[valid_mask, col].astype(int)

    return df


def evaluate_with_given_centers(cluster_centers: dict, gt_list: list, dist_thr: float = 6.0):
    """
    用外部给定的 cluster centers 和 gt_list 重新做一个简单匹配评估。
    这里先做 greedy nearest matching，仅用于 EMA / Kalman 验证。
    """
    gts = []
    for g in gt_list:
        gts.append({
            "gid": int(g["id"]),
            "center": np.array([float(g["x"]), float(g["y"])], dtype=float),
            "model": int(g["model"]),
        })

    cids = sorted(cluster_centers.keys())
    used_g = set()
    matches = []
    center_errors = []
    dx_errors = []
    dy_errors = []

    # greedy by nearest distance
    for cid in cids:
        ccenter = np.asarray(cluster_centers[cid], dtype=float)

        best_j = None
        best_d = float("inf")

        for j, gt in enumerate(gts):
            if j in used_g:
                continue

            d = float(np.linalg.norm(ccenter - gt["center"]))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j is not None and best_d <= float(dist_thr):
            gt = gts[best_j]
            used_g.add(best_j)

            dx = float(gt["center"][0] - ccenter[0])
            dy = float(gt["center"][1] - ccenter[1])

            matches.append({
                "cid": int(cid),
                "gid": int(gt["gid"]),
                "center_dist": float(best_d),
                "iou": float("nan"),
                "dx": float(dx),
                "dy": float(dy),
            })
            center_errors.append(float(best_d))
            dx_errors.append(float(dx))
            dy_errors.append(float(dy))

    TP = len(matches)
    FP = len(cids) - TP
    FN = len(gts) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if len(center_errors) > 0:
        ce = np.asarray(center_errors, dtype=float)
        dxe = np.asarray(dx_errors, dtype=float)
        dye = np.asarray(dy_errors, dtype=float)

        mean_center_error = float(np.mean(ce))
        median_center_error = float(np.median(ce))
        p90_center_error = float(np.percentile(ce, 90))
        p95_center_error = float(np.percentile(ce, 95))
        acc_0p3m = float(np.mean(ce <= 0.3))
        acc_0p5m = float(np.mean(ce <= 0.5))

        mean_dx_error = float(np.mean(dxe))
        mean_dy_error = float(np.mean(dye))
        median_dx_error = float(np.median(dxe))
        median_dy_error = float(np.median(dye))
        std_dx_error = float(np.std(dxe))
        std_dy_error = float(np.std(dye))
    else:
        mean_center_error = float("nan")
        median_center_error = float("nan")
        p90_center_error = float("nan")
        p95_center_error = float("nan")
        acc_0p3m = float("nan")
        acc_0p5m = float("nan")
        mean_dx_error = float("nan")
        mean_dy_error = float("nan")
        median_dx_error = float("nan")
        median_dy_error = float("nan")
        std_dx_error = float("nan")
        std_dy_error = float("nan")

    model_counts = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}
    used_gid = set(m["gid"] for m in matches)

    for gt in gts:
        mm = int(gt["model"])
        if gt["gid"] in used_gid:
            model_counts[mm]["TP"] += 1
        else:
            model_counts[mm]["FN"] += 1

    return {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_center_error": float(mean_center_error),
        "median_center_error": float(median_center_error),
        "p90_center_error": float(p90_center_error),
        "p95_center_error": float(p95_center_error),
        "acc_0p3m": float(acc_0p3m),
        "acc_0p5m": float(acc_0p5m),
        "mean_dx_error": float(mean_dx_error),
        "mean_dy_error": float(mean_dy_error),
        "median_dx_error": float(median_dx_error),
        "median_dy_error": float(median_dy_error),
        "std_dx_error": float(std_dx_error),
        "std_dy_error": float(std_dy_error),
        "center_errors": center_errors,
        "dx_errors": dx_errors,
        "dy_errors": dy_errors,
        "matches": matches,
        "unmatched_clusters": [cid for cid in cids if cid not in {m["cid"] for m in matches}],
        "unmatched_gts": [gt["gid"] for gt in gts if gt["gid"] not in used_gid],
        "model_counts": model_counts,
    }


def build_point_level_table(fid: int, frame_item: dict, pts: np.ndarray, labels: np.ndarray,
                            center_fn, bias_fn, cfg) -> pd.DataFrame:
    center_x = np.full(len(labels), np.nan, dtype=float)
    center_y = np.full(len(labels), np.nan, dtype=float)

    for cid in np.unique(labels):
        if cid < 1:
            continue

        mask = (labels == cid)
        cpts = pts[mask]
        if cpts.size == 0:
            continue

        center = compute_center_with_optional_velocity_filter(
            cpts=cpts,
            frame_item=frame_item,
            mask=mask,
            center_fn=center_fn,
            cfg=cfg,
        )
        center = bias_fn(center, cfg)

        center_x[mask] = center[0]
        center_y[mask] = center[1]

    df = pd.DataFrame({
        "Frame": np.full(len(labels), fid, dtype=int),
        "X": frame_item["X"],
        "Y": frame_item["Y"],
        "V": frame_item["V"],
        "SNR": frame_item["SNR"],
        "Label": labels,
        "Center_X": center_x,
        "Center_Y": center_y,
    })

    for col in ["Range", "Angle", "Speed"]:
        if col in frame_item:
            df[col] = frame_item[col]

    return df


def evaluate_frame(pts, labels, gt_list, snr, fit_mode, cfg):
    return eval_one_frame_target_level(
        pts_xy=pts,
        labels=labels,
        gt_list=gt_list,
        dist_thr=cfg.DIST_THR,
        iou_thr=cfg.IOU_THR,
        use_fixed_box=True,
        fixed_box_priors=cfg.FIXED_BOX_PRIORS,
        fixed_box_fit_mode=fit_mode,
        fixed_box_yaw=cfg.FIXED_BOX_YAW,
        fixed_box_score_lambda=cfg.FIXED_BOX_SCORE_LAMBDA,
        fixed_box_inside_margin=cfg.FIXED_BOX_INSIDE_MARGIN,
        fixed_box_alpha_out=cfg.FIXED_BOX_ALPHA_OUT,
        fixed_box_beta_in=cfg.FIXED_BOX_BETA_IN,
        snr=snr,
        cluster_center_mode=cfg.CLUSTER_CENTER_MODE,
        center_bias_x=cfg.CENTER_BIAS_X,
        center_bias_y=0.0,
        use_range_bias_y=cfg.USE_RANGE_BIAS_Y,
        bias_y_near=cfg.BIAS_Y_NEAR,
        bias_y_far=cfg.BIAS_Y_FAR,
        bias_split_y=cfg.BIAS_SPLIT_Y,
    )


def process_one_frame(fid: int, radar_data, gt_df, fit_mode: str, cfg, center_fn, bias_fn, tracker=None):
    frame_item = radar_data[fid]

    x = frame_item["X"]
    y = frame_item["Y"]
    v = frame_item["V"]
    pts = np.column_stack([x, y])

    labels = cluster_one_frame(radar_data, fid, cfg)
    gt_list = build_gt_list_for_frame(gt_df, fid)

    # 1) 当前帧原始中心
    cluster_centers_raw = build_cluster_centers(
        labels, pts, frame_item, center_fn, bias_fn, cfg
    )

    # 2) 在线关联 + 时序滤波
    track_assignments = {}
    cluster_centers_filtered = cluster_centers_raw
    raw_centers_for_export = {
        cid: np.asarray(c, dtype=float) for cid, c in cluster_centers_raw.items()
    }

    if tracker is not None:
        cluster_centers_filtered, track_assignments, raw_centers_for_export = \
            temporal_filter_cluster_centers_online(
                cluster_centers=cluster_centers_raw,
                tracker=tracker,
            )

    # 3) raw / filtered 两套评估，方便对比收益
    metrics_raw = evaluate_with_given_centers(
        cluster_centers=cluster_centers_raw,
        gt_list=gt_list,
        dist_thr=cfg.DIST_THR,
    )

    metrics = evaluate_with_given_centers(
        cluster_centers=cluster_centers_filtered,
        gt_list=gt_list,
        dist_thr=cfg.DIST_THR,
    )

    # 4) 点级表默认导出 filtered center，同时带 raw center + track_id
    point_table = build_point_level_table_from_centers(
        fid=fid,
        frame_item=frame_item,
        labels=labels,
        cluster_centers=cluster_centers_filtered,
        matches=metrics.get("matches", []),
        track_assignments=track_assignments,
        raw_cluster_centers=raw_centers_for_export,
    )

    cache_item = {
        "pts": pts,
        "v": v,
        "labels": labels,
        "gt_list": gt_list,
        "metrics": metrics,
        "metrics_raw": metrics_raw,
        "cluster_centers": cluster_centers_filtered,
        "cluster_centers_raw": cluster_centers_raw,
        "track_assignments": track_assignments,
    }

    return {
        "point_table": point_table,
        "metrics": metrics,
        "metrics_raw": metrics_raw,
        "cache_item": cache_item,
        "gt_list": gt_list,
    }


def temporal_filter_cluster_centers_online(cluster_centers: dict, tracker):
    """
    真正的在线时序滤波：
    - 不依赖 GT
    - 先 cluster-to-track 关联
    - 再做 EMA / Kalman 平滑
    """
    filtered_centers, track_assignments, raw_centers = tracker.step(cluster_centers)
    return filtered_centers, track_assignments, raw_centers
