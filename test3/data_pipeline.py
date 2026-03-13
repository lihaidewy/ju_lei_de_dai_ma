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


def build_point_level_table_from_centers(fid: int, frame_item: dict, labels: np.ndarray, cluster_centers: dict) -> pd.DataFrame:
    """
    用外部传入的 cluster_centers 构建点集表。
    """
    center_x = np.full(len(labels), np.nan, dtype=float)
    center_y = np.full(len(labels), np.nan, dtype=float)

    for cid, center in cluster_centers.items():
        mask = (labels == cid)
        center_x[mask] = float(center[0])
        center_y[mask] = float(center[1])

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

def evaluate_with_given_centers(cluster_centers: dict, gt_list: list, dist_thr: float = 6.0):
    """
    用外部给定的 cluster centers 和 gt_list 重新做一个简单匹配评估。
    这里先做 greedy nearest matching，仅用于 EMA 验证。
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
    snr = frame_item["SNR"]

    pts = np.column_stack([x, y])
    labels = cluster_one_frame(radar_data, fid, cfg)
    gt_list = build_gt_list_for_frame(gt_df, fid)

    # 先做单帧评估（得到 matches，用于 gid 关联）
    raw_metrics = evaluate_frame(pts, labels, gt_list, snr, fit_mode, cfg)

    # 显式生成单帧 cluster centers
    cluster_centers = build_cluster_centers(labels, pts, frame_item, center_fn, bias_fn, cfg)

    # 如果启用 EMA，则用 gid 做平滑
    if tracker is not None:
        cluster_centers = temporal_filter_cluster_centers_with_matches(
            cluster_centers=cluster_centers,
            matches=raw_metrics.get("matches", []),
            tracker=tracker,
        )


    # 用平滑后的中心重建点表
    point_table = build_point_level_table_from_centers(fid, frame_item, labels, cluster_centers)

    # 用平滑后的中心重新评估
    metrics = evaluate_with_given_centers(
        cluster_centers=cluster_centers,
        gt_list=gt_list,
        dist_thr=cfg.DIST_THR,
    )

    cache_item = {
        "pts": pts,
        "v": v,
        "labels": labels,
        "gt_list": gt_list,
        "metrics": metrics,
        "cluster_centers": cluster_centers,
    }

    return {
        "point_table": point_table,
        "metrics": metrics,
        "cache_item": cache_item,
        "gt_list": gt_list,
    }

