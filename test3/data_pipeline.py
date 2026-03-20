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

try:
    from centers import get_fixed_box_prior_candidates
except Exception:
    get_fixed_box_prior_candidates = None


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


def _fallback_get_fixed_box_prior_candidates(cfg, model_id=None):
    if model_id is not None:
        model_priors = getattr(cfg, "GT_MODEL_PRIORS", {})
        item = model_priors.get(int(model_id))
        if item is not None:
            return [(float(item["L"]), float(item["W"]))]

    priors = getattr(cfg, "FIXED_BOX_PRIORS", None)
    if priors:
        return [(float(l), float(w)) for l, w in priors]

    model_priors = getattr(cfg, "GT_MODEL_PRIORS", {})
    out = []
    for mid in sorted(model_priors.keys()):
        item = model_priors[mid]
        out.append((float(item["L"]), float(item["W"])))
    return out


def _get_prior_candidates(cfg, model_id=None):
    if get_fixed_box_prior_candidates is not None:
        try:
            return get_fixed_box_prior_candidates(cfg, model_id=model_id)
        except TypeError:
            return get_fixed_box_prior_candidates(cfg)
    return _fallback_get_fixed_box_prior_candidates(cfg, model_id=model_id)


def _guess_model_id_for_cluster(cpts, gt_list, cfg):
    """
    当前先用离线 nearest-GT 方式，为 fixed_box 选择模型先验。
    后续接视觉时，只需要把这里替换成视觉输出即可。
    """
    if not gt_list:
        return None

    center0 = np.mean(np.asarray(cpts, dtype=float), axis=0)

    best_model = None
    best_dist = float("inf")
    max_match_dist = float(getattr(cfg, "DIST_THR", 6.0)) * 2.0

    for g in gt_list:
        gt_center = np.array([float(g["x"]), float(g["y"])], dtype=float)
        d = float(np.linalg.norm(center0 - gt_center))
        if d < best_dist:
            best_dist = d
            best_model = int(g["model"])

    if best_dist <= max_match_dist:
        return best_model
    return None


def _box_score_axis_aligned(cpts, center, length, width, cfg):
    """
    简化版 fixed-box 打分：
    - yaw 固定为 0
    - X 对应宽度 W
    - Y 对应长度 L
    - 候选中心由不同 anchor 生成，再用 inside/outside 代价打分
    """
    cpts = np.asarray(cpts, dtype=float)
    center = np.asarray(center, dtype=float)

    half_w = 0.5 * float(width)
    half_l = 0.5 * float(length)
    margin = float(getattr(cfg, "FIXED_BOX_INSIDE_MARGIN", 0.2))
    alpha_out = float(getattr(cfg, "FIXED_BOX_ALPHA_OUT", 10.0))
    beta_in = float(getattr(cfg, "FIXED_BOX_BETA_IN", 2.0))
    score_lambda = float(getattr(cfg, "FIXED_BOX_SCORE_LAMBDA", 1.0))

    dx = np.abs(cpts[:, 0] - center[0]) - (half_w + margin)
    dy = np.abs(cpts[:, 1] - center[1]) - (half_l + margin)

    dx_pos = np.maximum(dx, 0.0)
    dy_pos = np.maximum(dy, 0.0)
    outside_penalty = np.sqrt(dx_pos ** 2 + dy_pos ** 2)

    inside_mask = (dx <= 0.0) & (dy <= 0.0)
    inside_ratio = float(np.mean(inside_mask)) if len(cpts) > 0 else 0.0

    score = score_lambda * alpha_out * float(np.mean(outside_penalty)) - beta_in * inside_ratio
    return score, inside_ratio, float(np.mean(outside_penalty))


def _fit_center_fixed_box_with_priors(cpts, prior_candidates, cfg):
    """
    简化版 fixed-box center:
    - 先枚举尺寸先验
    - 再枚举 3 种 anchor:
      1) bbox_center
      2) rear_anchor  : min_y + L/2
      3) front_anchor : max_y - L/2
    - 最终选择 score 最小的候选中心
    """
    cpts = np.asarray(cpts, dtype=float)
    x_med = float(np.median(cpts[:, 0]))
    x_box_center = 0.5 * (float(np.min(cpts[:, 0])) + float(np.max(cpts[:, 0])))
    y_min = float(np.min(cpts[:, 1]))
    y_max = float(np.max(cpts[:, 1]))
    y_box_center = 0.5 * (y_min + y_max)

    best = None

    for length, width in prior_candidates:
        candidate_centers = [
            np.array([x_box_center, y_box_center], dtype=float),
            np.array([x_med, y_min + 0.5 * float(length)], dtype=float),
            np.array([x_med, y_max - 0.5 * float(length)], dtype=float),
        ]

        for center in candidate_centers:
            score, inside_ratio, outside_mean = _box_score_axis_aligned(
                cpts=cpts,
                center=center,
                length=length,
                width=width,
                cfg=cfg,
            )
            item = {
                "center": np.asarray(center, dtype=float),
                "score": float(score),
                "inside_ratio": float(inside_ratio),
                "outside_mean": float(outside_mean),
                "prior_l": float(length),
                "prior_w": float(width),
            }
            if best is None or item["score"] < best["score"]:
                best = item

    if best is None:
        return {
            "center": np.mean(cpts, axis=0),
            "score": float("nan"),
            "inside_ratio": float("nan"),
            "outside_mean": float("nan"),
            "prior_l": float("nan"),
            "prior_w": float("nan"),
        }

    return best

def _fit_center_bottom_half_length_with_priors(cpts, prior_candidates):
    """
    bottom_half_length:
    - x 用 cluster 横向中位数
    - x 用 cluster 横向均值
    - y 用 cluster 最底部点 min_y + L/2
    - 如果有多个 prior，就逐个生成候选；当前先取第一个
      （通常 model prior 已经会把候选缩到 1 个）
    """
    cpts = np.asarray(cpts, dtype=float)

    # x_med = float(np.median(cpts[:, 0]))
    x_med = float(np.mean(cpts[:, 0]))
    y_bottom = float(np.min(cpts[:, 1]))

    if not prior_candidates:
        center = np.array([x_med, y_bottom], dtype=float)
        return {
            "center": center,
            "bottom_y": y_bottom,
            "prior_l": float("nan"),
            "prior_w": float("nan"),
        }

    length, width = prior_candidates[0]
    center = np.array([x_med, y_bottom + 0.5 * float(length)], dtype=float)
    return {
        "center": center,
        "bottom_y": y_bottom,
        "prior_l": float(length),
        "prior_w": float(width),
    }


def build_cluster_centers(labels, pts, frame_item, center_fn, bias_fn, cfg, gt_list=None):
    """
    显式构建当前帧每个 cluster 的中心。
    返回：
      cluster_centers: {cid: np.array([x, y])}
      cluster_meta:    {cid: {...}}
    """
    cluster_centers = {}
    cluster_meta = {}
    mode = (getattr(cfg, "CLUSTER_CENTER_MODE", "mean") or "mean").lower().strip()

    for cid in iter_valid_cluster_ids(labels):
        mask = labels == cid
        cpts = pts[mask]
        if cpts.size == 0:
            continue

        if mode == "fixed_box":
            model_id = None
            if bool(getattr(cfg, "FIXED_BOX_USE_MODEL_PRIOR", False)):
                model_id = _guess_model_id_for_cluster(cpts, gt_list or [], cfg)

            prior_candidates = _get_prior_candidates(cfg, model_id=model_id)
            if (not prior_candidates) and bool(getattr(cfg, "FIXED_BOX_FALLBACK_TO_ALL_PRIORS", True)):
                prior_candidates = _get_prior_candidates(cfg, model_id=None)

            fit_result = _fit_center_fixed_box_with_priors(cpts, prior_candidates, cfg)
            center = fit_result["center"]
            cluster_meta[int(cid)] = {
                "center_mode": "fixed_box",
                "selected_model_id": None if model_id is None else int(model_id),
                "prior_l": float(fit_result["prior_l"]),
                "prior_w": float(fit_result["prior_w"]),
                "fit_score": float(fit_result["score"]),
                "inside_ratio": float(fit_result["inside_ratio"]),
                "outside_mean": float(fit_result["outside_mean"]),
                "bottom_y": np.nan,
                "raw_bottom_center_y": np.nan,
            }

        elif mode == "bottom_half_length":
            model_id = None
            if bool(getattr(cfg, "FIXED_BOX_USE_MODEL_PRIOR", False)):
                model_id = _guess_model_id_for_cluster(cpts, gt_list or [], cfg)

            prior_candidates = _get_prior_candidates(cfg, model_id=model_id)
            if (not prior_candidates) and bool(getattr(cfg, "FIXED_BOX_FALLBACK_TO_ALL_PRIORS", True)):
                prior_candidates = _get_prior_candidates(cfg, model_id=None)

            fit_result = _fit_center_bottom_half_length_with_priors(cpts, prior_candidates)
            center = fit_result["center"]
            cluster_meta[int(cid)] = {
                "center_mode": "bottom_half_length",
                "selected_model_id": None if model_id is None else int(model_id),
                "prior_l": float(fit_result["prior_l"]),
                "prior_w": float(fit_result["prior_w"]),
                "fit_score": np.nan,
                "inside_ratio": np.nan,
                "outside_mean": np.nan,
                "bottom_y": float(fit_result["bottom_y"]),
                "raw_bottom_center_y": float(center[1]),
            }

        else:
            center = compute_center_with_optional_velocity_filter(
                cpts=cpts,
                frame_item=frame_item,
                mask=mask,
                center_fn=center_fn,
                cfg=cfg,
            )
            cluster_meta[int(cid)] = {
                "center_mode": mode,
                "selected_model_id": np.nan,
                "prior_l": np.nan,
                "prior_w": np.nan,
                "fit_score": np.nan,
                "inside_ratio": np.nan,
                "outside_mean": np.nan,
                "bottom_y": np.nan,
                "raw_bottom_center_y": np.nan,
            }

        center = bias_fn(center, cfg)
        cluster_centers[int(cid)] = np.asarray(center, dtype=float)

    return cluster_centers, cluster_meta


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


def _append_center_columns(df, labels, cluster_centers, raw_cluster_centers, cluster_meta=None):
    center_x_map = {}
    center_y_map = {}
    raw_center_x_map = {}
    raw_center_y_map = {}
    prior_l_map = {}
    prior_w_map = {}
    selected_model_map = {}
    fit_score_map = {}
    inside_ratio_map = {}
    bottom_y_map = {}
    raw_bottom_center_y_map = {}

    for cid, center in cluster_centers.items():
        center_x_map[int(cid)] = float(center[0])
        center_y_map[int(cid)] = float(center[1])

    if raw_cluster_centers is not None:
        for cid, center in raw_cluster_centers.items():
            raw_center_x_map[int(cid)] = float(center[0])
            raw_center_y_map[int(cid)] = float(center[1])

    if cluster_meta is not None:
        for cid, meta in cluster_meta.items():
            prior_l_map[int(cid)] = float(meta.get("prior_l", np.nan))
            prior_w_map[int(cid)] = float(meta.get("prior_w", np.nan))
            selected_model_map[int(cid)] = float(meta.get("selected_model_id", np.nan)) if meta.get("selected_model_id") is not None else np.nan
            fit_score_map[int(cid)] = float(meta.get("fit_score", np.nan))
            inside_ratio_map[int(cid)] = float(meta.get("inside_ratio", np.nan))
            bottom_y_map[int(cid)] = float(meta.get("bottom_y", np.nan))
            raw_bottom_center_y_map[int(cid)] = float(meta.get("raw_bottom_center_y", np.nan))

    df["Raw_Center_X"] = _assign_cluster_arrays(labels, raw_center_x_map)
    df["Raw_Center_Y"] = _assign_cluster_arrays(labels, raw_center_y_map)
    df["Center_X"] = _assign_cluster_arrays(labels, center_x_map)
    df["Center_Y"] = _assign_cluster_arrays(labels, center_y_map)
    df["Prior_L"] = _assign_cluster_arrays(labels, prior_l_map)
    df["Prior_W"] = _assign_cluster_arrays(labels, prior_w_map)
    df["Selected_Model_ID"] = _assign_cluster_arrays(labels, selected_model_map)
    df["FixedBox_Score"] = _assign_cluster_arrays(labels, fit_score_map)
    df["FixedBox_InsideRatio"] = _assign_cluster_arrays(labels, inside_ratio_map)
    df["Bottom_Y"] = _assign_cluster_arrays(labels, bottom_y_map)
    df["Raw_Bottom_Center_Y"] = _assign_cluster_arrays(labels, raw_bottom_center_y_map)

    for col in ["Selected_Model_ID"]:
        valid_mask = df[col].notna()
        if valid_mask.any():
            df.loc[valid_mask, col] = df.loc[valid_mask, col].astype(int)


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
    cluster_meta=None,
):
    df = frame_item.copy().reset_index(drop=True)
    df["Label"] = labels

    cid_to_gid = build_cid_to_gid(matches)
    _append_id_columns(df, labels, cid_to_gid, track_assignments)
    _append_center_columns(df, labels, cluster_centers, raw_cluster_centers, cluster_meta=cluster_meta)

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


def build_point_level_table(fid, frame_item, pts, labels, center_fn, bias_fn, cfg, gt_list=None):
    cluster_centers, cluster_meta = build_cluster_centers(
        labels=labels,
        pts=pts,
        frame_item=frame_item,
        center_fn=center_fn,
        bias_fn=bias_fn,
        cfg=cfg,
        gt_list=gt_list,
    )
    return build_point_level_table_from_centers(
        fid=fid,
        frame_item=frame_item,
        labels=labels,
        cluster_centers=cluster_centers,
        matches=None,
        track_assignments=None,
        raw_cluster_centers=None,
        cluster_meta=cluster_meta,
    )


def process_one_frame(fid, radar_data, gt_df, fit_mode, cfg, center_fn, bias_fn, tracker=None):
    frame_item = radar_data[fid]

    pts = np.column_stack([frame_item["X"], frame_item["Y"]])
    v = frame_item["V"]
    labels = cluster_one_frame(radar_data, fid, cfg)
    gt_list = build_gt_list_for_frame(gt_df, fid)
    gt_map, gt_model_map, gt_pos_map = build_gt_maps(gt_list)

    cluster_centers_raw, cluster_meta_raw = build_cluster_centers(
        labels=labels,
        pts=pts,
        frame_item=frame_item,
        center_fn=center_fn,
        bias_fn=bias_fn,
        cfg=cfg,
        gt_list=gt_list,
    )

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
        cluster_meta=cluster_meta_raw,
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
        "cluster_meta_raw": cluster_meta_raw,
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
    filtered_centers, track_assignments, raw_centers = tracker.step(cluster_centers)
    return filtered_centers, track_assignments, raw_centers
