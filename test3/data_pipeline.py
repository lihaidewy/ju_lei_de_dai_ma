import sys
from pathlib import Path

import numpy as np
import pandas as pd

from load_data2 import load_data
from plot_gt_main import load_gt_reference

# 把项目根目录加入搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mylib.cluster_frame_dbscan import cluster_frame_dbscan
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
        raise ValueError("No common frames between {} and {}".format(cfg.RADAR_PATH, cfg.GT_PATH))

    if cfg.FRAMES_TO_SHOW is None:
        return common_frames[: args.max_frames]

    frame_ids = [fid for fid in cfg.FRAMES_TO_SHOW if fid in common_frames]
    if len(frame_ids) == 0:
        raise ValueError("None of FRAMES_TO_SHOW are in common_frames")
    return frame_ids



def build_gt_list_for_frame(gt_df, fid):
    g = gt_df[gt_df["Frame"] == fid]
    return [
        {"id": int(r.ID), "x": float(r.X), "y": float(r.Y), "model": int(r.model)}
        for r in g.itertuples(index=False)
    ]



def build_gt_maps(gt_list):
    gt_map = {}
    gt_model_map = {}
    gt_pos_map = {}

    for g in gt_list:
        gid = int(g["id"])
        gx = float(g["x"])
        gy = float(g["y"])
        model = int(g["model"])

        gt_map[gid] = {"x": gx, "y": gy, "model": model}
        gt_model_map[gid] = model
        gt_pos_map[gid] = np.array([gx, gy], dtype=float)

    return gt_map, gt_model_map, gt_pos_map



def cluster_one_frame(radar_data, fid, cfg):
    return cluster_frame_dbscan(
        radar_data,
        fid,
        eps_x=cfg.EPS_X,
        eps_y=cfg.EPS_Y,
        eps_v=cfg.EPS_V,
        min_pts=cfg.MIN_PTS,
    )



def iter_valid_cluster_ids(labels):
    for cid in np.unique(labels):
        if int(cid) >= 1:
            yield int(cid)



def build_cluster_centers(labels, pts, frame_item, center_fn, bias_fn, cfg):
    """
    显式构建当前帧每个 cluster 的中心。
    返回：{cid: np.array([x, y])}
    """
    cluster_centers = {}

    for cid in iter_valid_cluster_ids(labels):
        mask = labels == cid
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
        cluster_centers[cid] = np.asarray(center, dtype=float)

    return cluster_centers



def temporal_filter_cluster_centers_with_matches(cluster_centers, matches, tracker):
    """
    用当前帧匹配结果中的 gid 作为 track_id，
    对 cluster center 做时序滤波（EMA 或 Kalman 都可）。
    """
    filtered_centers = {}
    cid_to_gid = build_cid_to_gid(matches)

    for cid, center in cluster_centers.items():
        if cid in cid_to_gid:
            gid = cid_to_gid[cid]
            filtered_centers[cid] = tracker.update(gid, center)
        else:
            filtered_centers[cid] = np.asarray(center, dtype=float)

    return filtered_centers



def build_cid_to_gid(matches):
    cid_to_gid = {}
    if matches is None:
        return cid_to_gid

    for mm in matches:
        cid_to_gid[int(mm["cid"])] = int(mm["gid"])

    return cid_to_gid



def _assign_cluster_arrays(labels, cluster_values, default_value=np.nan):
    out = np.full(len(labels), default_value, dtype=float)
    for cid, value in cluster_values.items():
        out[labels == int(cid)] = float(value)
    return out



def _append_center_columns(df, labels, cluster_centers, raw_cluster_centers):
    center_x_map = {}
    center_y_map = {}
    raw_center_x_map = {}
    raw_center_y_map = {}

    for cid, center in cluster_centers.items():
        center_x_map[int(cid)] = float(center[0])
        center_y_map[int(cid)] = float(center[1])

    if raw_cluster_centers is not None:
        for cid, center in raw_cluster_centers.items():
            raw_center_x_map[int(cid)] = float(center[0])
            raw_center_y_map[int(cid)] = float(center[1])

    df["Raw_Center_X"] = _assign_cluster_arrays(labels, raw_center_x_map)
    df["Raw_Center_Y"] = _assign_cluster_arrays(labels, raw_center_y_map)
    df["Center_X"] = _assign_cluster_arrays(labels, center_x_map)
    df["Center_Y"] = _assign_cluster_arrays(labels, center_y_map)



def _append_id_columns(df, labels, cid_to_gid, track_assignments):
    gid_map = {}
    track_id_map = {}

    for cid, gid in cid_to_gid.items():
        gid_map[int(cid)] = int(gid)

    if track_assignments is not None:
        for cid, track_id in track_assignments.items():
            track_id_map[int(cid)] = int(track_id)

    df["gid"] = _assign_cluster_arrays(labels, gid_map)
    df["track_id"] = _assign_cluster_arrays(labels, track_id_map)

    for col in ["gid", "track_id"]:
        valid_mask = df[col].notna()
        if valid_mask.any():
            df.loc[valid_mask, col] = df.loc[valid_mask, col].astype(int)



def build_point_level_table_from_centers(
    fid,
    frame_item,
    labels,
    cluster_centers,
    matches=None,
    track_assignments=None,
    raw_cluster_centers=None,
):
    """
    用外部传入的 cluster_centers 构建点级表。
    保留原始表格列顺序，只在右侧追加结果列。

    新增:
    - track_id
    - Raw_Center_X / Raw_Center_Y
    """
    df = frame_item.copy().reset_index(drop=True)
    df["Label"] = labels

    cid_to_gid = build_cid_to_gid(matches)
    _append_id_columns(df, labels, cid_to_gid, track_assignments)
    _append_center_columns(df, labels, cluster_centers, raw_cluster_centers)

    return df



def _build_eval_summary(matches, cids, gts):
    tp = len(matches)
    fp = len(cids) - tp
    fn = len(gts) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    center_errors = [float(m["center_dist"]) for m in matches]
    dx_errors = [float(m["dx"]) for m in matches]
    dy_errors = [float(m["dy"]) for m in matches]

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

    used_gid = set([m["gid"] for m in matches])
    model_counts = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}
    for gt in gts:
        model = int(gt["model"])
        if gt["gid"] in used_gid:
            model_counts[model]["TP"] += 1
        else:
            model_counts[model]["FN"] += 1

    matched_cids = set([m["cid"] for m in matches])

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
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
        "unmatched_clusters": [cid for cid in cids if cid not in matched_cids],
        "unmatched_gts": [gt["gid"] for gt in gts if gt["gid"] not in used_gid],
        "model_counts": model_counts,
    }



def evaluate_with_given_centers(cluster_centers, gt_list, dist_thr=6.0):
    """
    用外部给定的 cluster centers 和 gt_list 重新做一个简单匹配评估。
    这里先做 greedy nearest matching，仅用于 EMA / Kalman 验证。
    """
    gts = []
    for g in gt_list:
        gts.append(
            {
                "gid": int(g["id"]),
                "center": np.array([float(g["x"]), float(g["y"])], dtype=float),
                "model": int(g["model"]),
            }
        )

    cids = sorted(cluster_centers.keys())
    used_g = set()
    matches = []

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

            matches.append(
                {
                    "cid": int(cid),
                    "gid": int(gt["gid"]),
                    "center_dist": float(best_d),
                    "iou": float("nan"),
                    "dx": float(dx),
                    "dy": float(dy),
                }
            )

    return _build_eval_summary(matches, cids, gts)



def build_point_level_table(fid, frame_item, pts, labels, center_fn, bias_fn, cfg):
    cluster_centers = build_cluster_centers(labels, pts, frame_item, center_fn, bias_fn, cfg)
    return build_point_level_table_from_centers(
        fid=fid,
        frame_item=frame_item,
        labels=labels,
        cluster_centers=cluster_centers,
        matches=None,
        track_assignments=None,
        raw_cluster_centers=None,
    )



def process_one_frame(fid, radar_data, gt_df, fit_mode, cfg, center_fn, bias_fn, tracker=None):
    frame_item = radar_data[fid]

    pts = np.column_stack([frame_item["X"], frame_item["Y"]])
    v = frame_item["V"]
    labels = cluster_one_frame(radar_data, fid, cfg)
    gt_list = build_gt_list_for_frame(gt_df, fid)
    gt_map, gt_model_map, gt_pos_map = build_gt_maps(gt_list)

    cluster_centers_raw = build_cluster_centers(labels, pts, frame_item, center_fn, bias_fn, cfg)

    track_assignments = {}
    cluster_centers_filtered = cluster_centers_raw
    raw_centers_for_export = dict((cid, np.asarray(c, dtype=float)) for cid, c in cluster_centers_raw.items())

    if tracker is not None:
        cluster_centers_filtered, track_assignments, raw_centers_for_export = temporal_filter_cluster_centers_online(
            cluster_centers=cluster_centers_raw,
            tracker=tracker,
        )

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
        "gt_map": gt_map,
        "gt_model_map": gt_model_map,
        "gt_pos_map": gt_pos_map,
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
        "gt_map": gt_map,
        "gt_model_map": gt_model_map,
        "gt_pos_map": gt_pos_map,
    }



def temporal_filter_cluster_centers_online(cluster_centers, tracker):
    """
    真正的在线时序滤波：
    - 不依赖 GT
    - 先 cluster-to-track 关联
    - 再做 EMA / Kalman 平滑
    """
    filtered_centers, track_assignments, raw_centers = tracker.step(cluster_centers)
    return filtered_centers, track_assignments, raw_centers
