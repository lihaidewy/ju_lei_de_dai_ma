from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _safe_float(x, default=np.nan):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _norm2(a, b):
    a = np.asarray(a, dtype=float).reshape(2)
    b = np.asarray(b, dtype=float).reshape(2)
    return float(np.linalg.norm(a - b))


def _build_match_map(metrics: dict) -> Dict[int, dict]:
    out = {}
    if metrics is None:
        return out
    for m in metrics.get("matches", []):
        cid = int(m["cid"])
        out[cid] = dict(m)
    return out


def collect_track_records(cache: Dict[int, dict], frame_ids: List[int]) -> pd.DataFrame:
    """
    逐帧展开为 track-centric 明细表。
    一行表示：某一帧里，一个已输出的 track_id 对应的 cluster 记录。
    """
    rows = []

    for frame_idx, fid in enumerate(frame_ids):
        item = cache.get(fid, {})
        cluster_centers = item.get("cluster_centers", {}) or {}
        cluster_centers_raw = item.get("cluster_centers_raw", {}) or {}
        cluster_meta_raw = item.get("cluster_meta_raw", {}) or {}
        track_assignments = item.get("track_assignments", {}) or {}
        metrics = item.get("metrics", {}) or {}
        metrics_raw = item.get("metrics_raw", {}) or {}
        gt_list = item.get("gt_list", []) or {}

        match_map = _build_match_map(metrics)
        match_map_raw = _build_match_map(metrics_raw)

        gt_map = {}
        for g in gt_list:
            gid = int(g["id"])
            gt_map[gid] = {
                "x": float(g["x"]),
                "y": float(g["y"]),
                "model": int(g["model"]),
            }

        # 只针对当前帧被正式输出的 track
        for cid, tid in track_assignments.items():
            cid = int(cid)
            tid = int(tid)

            if cid not in cluster_centers:
                continue

            filt_center = np.asarray(cluster_centers[cid], dtype=float).reshape(2)

            raw_center = cluster_centers_raw.get(cid, filt_center)
            raw_center = np.asarray(raw_center, dtype=float).reshape(2)

            meta = cluster_meta_raw.get(cid, {}) or {}
            mm = match_map.get(cid)
            mm_raw = match_map_raw.get(cid)

            gid = None
            gt_x = np.nan
            gt_y = np.nan
            gt_model = np.nan

            dx_filtered = np.nan
            dy_filtered = np.nan
            err_filtered = np.nan

            dx_raw = np.nan
            dy_raw = np.nan
            err_raw = np.nan

            if mm is not None:
                gid = int(mm["gid"])
                dx_filtered = _safe_float(mm.get("dx"))
                dy_filtered = _safe_float(mm.get("dy"))
                err_filtered = _safe_float(mm.get("center_dist"))

                if gid in gt_map:
                    gt_x = float(gt_map[gid]["x"])
                    gt_y = float(gt_map[gid]["y"])
                    gt_model = int(gt_map[gid]["model"])

            if mm_raw is not None:
                dx_raw = _safe_float(mm_raw.get("dx"))
                dy_raw = _safe_float(mm_raw.get("dy"))
                err_raw = _safe_float(mm_raw.get("center_dist"))

                if gid is None:
                    gid = _safe_int(mm_raw.get("gid"))

            rows.append(
                {
                    "Frame": int(fid),
                    "frame_index": int(frame_idx),
                    "track_id": int(tid),
                    "cid": int(cid),
                    "gid": _safe_int(gid),
                    "raw_x": float(raw_center[0]),
                    "raw_y": float(raw_center[1]),
                    "filtered_x": float(filt_center[0]),
                    "filtered_y": float(filt_center[1]),
                    "gt_x": _safe_float(gt_x),
                    "gt_y": _safe_float(gt_y),
                    "dx_raw": _safe_float(dx_raw),
                    "dy_raw": _safe_float(dy_raw),
                    "err_raw": _safe_float(err_raw),
                    "dx_filtered": _safe_float(dx_filtered),
                    "dy_filtered": _safe_float(dy_filtered),
                    "err_filtered": _safe_float(err_filtered),
                    "gt_model": _safe_float(gt_model),
                    "prior_l": _safe_float(meta.get("prior_l")),
                    "prior_w": _safe_float(meta.get("prior_w")),
                    "fit_score": _safe_float(meta.get("fit_score")),
                    "inside_ratio": _safe_float(meta.get("inside_ratio")),
                    "center_mode": meta.get("center_mode"),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df = df.sort_values(["track_id", "Frame", "cid"]).reset_index(drop=True)

    # 相邻帧跳变 / 平滑收益
    raw_jumps = []
    filt_jumps = []
    smoothing_gains = []
    frame_gaps = []
    gt_switches = []

    prev_by_tid = {}

    for row in df.itertuples(index=False):
        tid = int(row.track_id)

        raw_c = np.array([row.raw_x, row.raw_y], dtype=float)
        filt_c = np.array([row.filtered_x, row.filtered_y], dtype=float)
        gid = _safe_int(row.gid)

        if tid not in prev_by_tid:
            raw_jumps.append(np.nan)
            filt_jumps.append(np.nan)
            smoothing_gains.append(np.nan)
            frame_gaps.append(np.nan)
            gt_switches.append(False)
        else:
            prev = prev_by_tid[tid]
            prev_raw = prev["raw"]
            prev_filt = prev["filt"]
            prev_frame = prev["frame"]
            prev_gid = prev["gid"]

            rj = _norm2(raw_c, prev_raw)
            fj = _norm2(filt_c, prev_filt)

            raw_jumps.append(float(rj))
            filt_jumps.append(float(fj))
            smoothing_gains.append(float(rj - fj))
            frame_gaps.append(int(row.Frame) - int(prev_frame))
            gt_switches.append(
                (gid is not None) and (prev_gid is not None) and (gid != prev_gid)
            )

        prev_by_tid[tid] = {
            "raw": raw_c,
            "filt": filt_c,
            "frame": int(row.Frame),
            "gid": gid,
        }

    df["raw_jump"] = raw_jumps
    df["filtered_jump"] = filt_jumps
    df["smoothing_gain"] = smoothing_gains
    df["frame_gap"] = frame_gaps
    df["is_gt_switch"] = gt_switches

    # 是否轨迹首尾帧
    first_frame_map = df.groupby("track_id")["Frame"].min().to_dict()
    last_frame_map = df.groupby("track_id")["Frame"].max().to_dict()
    df["is_first_frame"] = df.apply(lambda r: int(r["Frame"]) == int(first_frame_map[int(r["track_id"])]), axis=1)
    df["is_last_frame"] = df.apply(lambda r: int(r["Frame"]) == int(last_frame_map[int(r["track_id"])]), axis=1)

    return df


def _main_gid_and_switches(sub_df: pd.DataFrame) -> Tuple[object, int]:
    gids = [g for g in sub_df["gid"].tolist() if pd.notna(g)]
    if len(gids) == 0:
        return np.nan, 0

    gids = [int(g) for g in gids]
    main_gid = pd.Series(gids).value_counts().idxmax()

    switches = 0
    prev = None
    for g in gids:
        if prev is not None and g != prev:
            switches += 1
        prev = g

    return int(main_gid), int(switches)


def build_track_summary(track_frame_df: pd.DataFrame) -> pd.DataFrame:
    if track_frame_df is None or len(track_frame_df) == 0:
        return pd.DataFrame()

    rows = []

    for tid, sub in track_frame_df.groupby("track_id", sort=True):
        sub = sub.sort_values("Frame").reset_index(drop=True)

        start_frame = int(sub["Frame"].min())
        end_frame = int(sub["Frame"].max())
        life_frames = int(end_frame - start_frame + 1)
        num_observed_frames = int(len(sub))
        num_gaps = int(np.sum(sub["frame_gap"].fillna(1).values > 1))
        num_gt_switches = int(np.sum(sub["is_gt_switch"].astype(bool)))

        path_length_raw = float(np.nansum(sub["raw_jump"].values)) if "raw_jump" in sub else np.nan
        path_length_filtered = float(np.nansum(sub["filtered_jump"].values)) if "filtered_jump" in sub else np.nan

        main_gid, assigned_gt_switches = _main_gid_and_switches(sub)

        rows.append(
            {
                "track_id": int(tid),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "life_frames": life_frames,
                "num_observed_frames": num_observed_frames,
                "num_missing_internal_gaps": num_gaps,
                "main_gid": main_gid,
                "assigned_gt_switches": int(assigned_gt_switches),
                "num_gt_switch_events": int(num_gt_switches),
                "mean_raw_jump": _safe_float(np.nanmean(sub["raw_jump"].values)),
                "mean_filtered_jump": _safe_float(np.nanmean(sub["filtered_jump"].values)),
                "mean_smoothing_gain": _safe_float(np.nanmean(sub["smoothing_gain"].values)),
                "path_length_raw": _safe_float(path_length_raw),
                "path_length_filtered": _safe_float(path_length_filtered),
                "mean_err_raw": _safe_float(np.nanmean(sub["err_raw"].values)),
                "mean_err_filtered": _safe_float(np.nanmean(sub["err_filtered"].values)),
                "median_err_filtered": _safe_float(np.nanmedian(sub["err_filtered"].values)),
                "p90_err_filtered": _safe_float(np.nanpercentile(sub["err_filtered"].values, 90))
                if np.any(np.isfinite(sub["err_filtered"].values))
                else np.nan,
                "max_err_filtered": _safe_float(np.nanmax(sub["err_filtered"].values))
                if np.any(np.isfinite(sub["err_filtered"].values))
                else np.nan,
                "mean_dx_filtered": _safe_float(np.nanmean(sub["dx_filtered"].values)),
                "mean_dy_filtered": _safe_float(np.nanmean(sub["dy_filtered"].values)),
                "mean_inside_ratio": _safe_float(np.nanmean(sub["inside_ratio"].values)),
                "center_mode": sub["center_mode"].dropna().iloc[0] if sub["center_mode"].notna().any() else None,
            }
        )

    out = pd.DataFrame(rows).sort_values(["life_frames", "track_id"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_track_diagnostic_summary(track_frame_df: pd.DataFrame, track_summary_df: pd.DataFrame) -> dict:
    if track_summary_df is None or len(track_summary_df) == 0:
        return {
            "num_tracks": 0,
            "num_single_frame_tracks": 0,
            "num_short_tracks_le_2": 0,
            "mean_track_life": np.nan,
            "median_track_life": np.nan,
            "mean_raw_jump": np.nan,
            "mean_filtered_jump": np.nan,
            "mean_smoothing_gain": np.nan,
            "mean_err_raw": np.nan,
            "mean_err_filtered": np.nan,
            "num_tracks_with_gt_switch": 0,
            "num_gt_with_multi_track_ids": 0,
        }

    gt_multi_track = 0
    if track_frame_df is not None and len(track_frame_df) > 0 and "gid" in track_frame_df.columns:
        valid = track_frame_df.dropna(subset=["gid"]).copy()
        if len(valid) > 0:
            tmp = valid.groupby("gid")["track_id"].nunique()
            gt_multi_track = int(np.sum(tmp > 1))

    return {
        "num_tracks": int(len(track_summary_df)),
        "num_single_frame_tracks": int(np.sum(track_summary_df["num_observed_frames"].values == 1)),
        "num_short_tracks_le_2": int(np.sum(track_summary_df["num_observed_frames"].values <= 2)),
        "mean_track_life": _safe_float(np.nanmean(track_summary_df["life_frames"].values)),
        "median_track_life": _safe_float(np.nanmedian(track_summary_df["life_frames"].values)),
        "mean_raw_jump": _safe_float(np.nanmean(track_summary_df["mean_raw_jump"].values)),
        "mean_filtered_jump": _safe_float(np.nanmean(track_summary_df["mean_filtered_jump"].values)),
        "mean_smoothing_gain": _safe_float(np.nanmean(track_summary_df["mean_smoothing_gain"].values)),
        "mean_err_raw": _safe_float(np.nanmean(track_summary_df["mean_err_raw"].values)),
        "mean_err_filtered": _safe_float(np.nanmean(track_summary_df["mean_err_filtered"].values)),
        "num_tracks_with_gt_switch": int(np.sum(track_summary_df["num_gt_switch_events"].values > 0)),
        "num_gt_with_multi_track_ids": int(gt_multi_track),
    }


def print_track_diagnostic_summary(summary: dict):
    print("\n================ Track Diagnostic Summary ================")
    print("num_tracks                :", summary.get("num_tracks"))
    print("num_single_frame_tracks   :", summary.get("num_single_frame_tracks"))
    print("num_short_tracks_le_2     :", summary.get("num_short_tracks_le_2"))
    print("mean_track_life           :", summary.get("mean_track_life"))
    print("median_track_life         :", summary.get("median_track_life"))
    print("mean_raw_jump             :", summary.get("mean_raw_jump"))
    print("mean_filtered_jump        :", summary.get("mean_filtered_jump"))
    print("mean_smoothing_gain       :", summary.get("mean_smoothing_gain"))
    print("mean_err_raw              :", summary.get("mean_err_raw"))
    print("mean_err_filtered         :", summary.get("mean_err_filtered"))
    print("num_tracks_with_gt_switch :", summary.get("num_tracks_with_gt_switch"))
    print("num_gt_with_multi_track_ids:", summary.get("num_gt_with_multi_track_ids"))
    print("=========================================================\n")


def export_track_diagnostics(
    track_frame_df: pd.DataFrame,
    track_summary_df: pd.DataFrame,
    summary: dict,
    frame_csv_path: str,
    track_csv_path: str,
    summary_csv_path: str,
):
    if track_frame_df is not None and len(track_frame_df) > 0:
        track_frame_df.to_csv(frame_csv_path, index=False, encoding="utf-8-sig")

    if track_summary_df is not None and len(track_summary_df) > 0:
        track_summary_df.to_csv(track_csv_path, index=False, encoding="utf-8-sig")

    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
