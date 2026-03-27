import argparse
import matplotlib.pyplot as plt

from animated_viewer import launch_animated_viewer
from centers import get_bias_function, get_center_function
from config import Config
from data_pipeline import get_frame_ids, load_all_data, process_one_frame
from debug_tracking import TrackingDebugTool
from exporters import export_point_table, export_tp_matches, export_tp_matches_excel
from online_tracker import OnlineTrackerManager
from stats_utils import init_stats, print_global_summary, update_range_bias_stats, update_stats
from viz_utils import render_frame
from trajectory_diagnostics import collect_track_records,build_track_summary,build_track_diagnostic_summary,print_track_diagnostic_summary,export_track_diagnostics


CENTER_MODE_CHOICES = [
    "mean",
    "median",
    "snr_mean",
    "trimmed_mean",
    "mean_x_median_y",
    "velocity_mean",
    "velocity_trimmed_mean",
    "fixed_box",
    "bottom_half_length",
]

BIAS_MODE_CHOICES = [
    "two_segment",
    "none",
]


def parse_args():
    defaults = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--fit_mode", type=str, default="center", choices=["center", "edge"])
    p.add_argument("--max_frames", type=int, default=defaults.MAX_FRAMES_TO_VIEW)

    p.add_argument("--center_mode", type=str, default=defaults.CLUSTER_CENTER_MODE, choices=CENTER_MODE_CHOICES)
    p.add_argument("--bias_mode", type=str, default=defaults.BIAS_MODE, choices=BIAS_MODE_CHOICES)
    p.add_argument("--trim_ratio", type=float, default=defaults.TRIMMED_MEAN_RATIO)

    return p.parse_args()


def build_tracker(cfg):
    if not getattr(cfg, "USE_ONLINE_TRACKER", False):
        return None

    method = getattr(cfg, "TRACKER_METHOD", "cv")
    assoc_metric = getattr(cfg, "TRACK_ASSOC_METRIC", "euclidean")

    use_vel_assoc = getattr(cfg, "TRACK_USE_VEL_ASSOC", True)
    assoc_w_pos = getattr(cfg, "TRACK_ASSOC_W_POS", 1.0)
    assoc_w_vel = getattr(cfg, "TRACK_ASSOC_W_VEL", 0.8)

    q_pos = cfg.KF_Q_POS
    q_vel = cfg.KF_Q_VEL
    q_acc = getattr(cfg, "KF_Q_ACC", 0.20)

    use_adaptive_r = getattr(cfg, "KF_USE_ADAPTIVE_R", False)
    adaptive_r_gain = getattr(cfg, "KF_ADAPTIVE_R_GAIN", 0.25)
    min_r_scale = getattr(cfg, "KF_MIN_R_SCALE", 0.75)
    max_r_scale = getattr(cfg, "KF_MAX_R_SCALE", 4.0)

    enable_output_ema = getattr(cfg, "TRACK_ENABLE_OUTPUT_EMA", False)
    output_ema_alpha = getattr(cfg, "TRACK_OUTPUT_EMA_ALPHA", 0.85)

    # ------------------------------------------------------------
    # 第一步：强约束 CV 基线
    # ------------------------------------------------------------
    if getattr(cfg, "TRACK_STRICT_CONSTANT_VELOCITY", False):
        method = "cv"
        assoc_metric = "euclidean"

        use_vel_assoc = False
        assoc_w_pos = 1.0
        assoc_w_vel = 0.0

        q_pos = 0.02
        q_vel = 0.02
        q_acc = 0.0

        use_adaptive_r = False
        adaptive_r_gain = 0.0
        min_r_scale = 1.0
        max_r_scale = 1.0

        enable_output_ema = False
        output_ema_alpha = 1.0

    return OnlineTrackerManager(
        method=getattr(cfg, "TRACKER_METHOD", "cv"),
        assoc_metric=getattr(cfg, "TRACK_ASSOC_METRIC", "euclidean"),
        assoc_dist_thr=cfg.TRACK_ASSOC_DIST_THR,
        assoc_mahal_thr=getattr(cfg, "TRACK_ASSOC_MAHAL_THR", 3.5),

        # velocity-aware association
        use_vel_assoc=getattr(cfg, "TRACK_USE_VEL_ASSOC", True),
        assoc_vel_thr=getattr(cfg, "TRACK_ASSOC_VEL_THR", 2.0),
        assoc_w_pos=getattr(cfg, "TRACK_ASSOC_W_POS", 1.0),
        assoc_w_vel=getattr(cfg, "TRACK_ASSOC_W_VEL", 0.8),
        track_vel_ema_alpha=getattr(cfg, "TRACK_VEL_EMA_ALPHA", 0.6),

        max_misses=cfg.TRACK_MAX_MISSES,
        min_hits_to_confirm=getattr(cfg, "TRACK_MIN_HITS_TO_CONFIRM", 2),
        max_tentative_misses=getattr(cfg, "TRACK_MAX_TENTATIVE_MISSES", 1),
        dt=cfg.KF_DT,
        q_pos=cfg.KF_Q_POS,
        q_vel=cfg.KF_Q_VEL,
        q_acc=getattr(cfg, "KF_Q_ACC", 0.20),
        r_pos=cfg.KF_R_POS,
        init_pos_var=getattr(cfg, "KF_INIT_POS_VAR", 4.0),
        init_vel_var=getattr(cfg, "KF_INIT_VEL_VAR", 9.0),
        init_acc_var=getattr(cfg, "KF_INIT_ACC_VAR", 16.0),

        use_adaptive_r=getattr(cfg, "KF_USE_ADAPTIVE_R", False),
        adaptive_r_gain=getattr(cfg, "KF_ADAPTIVE_R_GAIN", 0.25),
        min_r_scale=getattr(cfg, "KF_MIN_R_SCALE", 0.75),
        max_r_scale=getattr(cfg, "KF_MAX_R_SCALE", 4.0),

        enable_output_ema=getattr(cfg, "TRACK_ENABLE_OUTPUT_EMA", False),
        output_ema_alpha=getattr(cfg, "TRACK_OUTPUT_EMA_ALPHA", 0.85),

        # ------------------------------------------------------------
        # Step 2: quality-aware R
        # ------------------------------------------------------------
        use_quality_aware_r=getattr(cfg, "KF_USE_QUALITY_AWARE_R", False),
        quality_r_min_scale=getattr(cfg, "KF_QUALITY_R_MIN_SCALE", 0.75),
        quality_r_max_scale=getattr(cfg, "KF_QUALITY_R_MAX_SCALE", 4.0),

        quality_singleton_penalty=getattr(cfg, "KF_QUALITY_SINGLETON_PENALTY", 2.20),
        quality_two_points_penalty=getattr(cfg, "KF_QUALITY_TWO_POINTS_PENALTY", 1.50),
        quality_three_points_penalty=getattr(cfg, "KF_QUALITY_THREE_POINTS_PENALTY", 1.15),
        quality_many_points_reward=getattr(cfg, "KF_QUALITY_MANY_POINTS_REWARD", 0.90),
        quality_many_points_thr=getattr(cfg, "KF_QUALITY_MANY_POINTS_THR", 4),

        quality_ref_vr_std=getattr(cfg, "KF_QUALITY_REF_VR_STD", 0.80),
        quality_high_vr_std_penalty=getattr(cfg, "KF_QUALITY_HIGH_VR_STD_PENALTY", 1.20),
        quality_low_vr_std_thr=getattr(cfg, "KF_QUALITY_LOW_VR_STD_THR", 0.20),
        quality_low_vr_std_reward=getattr(cfg, "KF_QUALITY_LOW_VR_STD_REWARD", 0.95),
    )

def collect_tp_match_rows(fid, cluster_centers, gt_list, metrics):
    rows = []
    gt_map = {
        int(g["id"]): {
            "x": float(g["x"]),
            "y": float(g["y"]),
            "model": int(g["model"]),
        }
        for g in gt_list
    }

    for match in metrics.get("matches", []):
        cid = int(match["cid"])
        gid = int(match["gid"])
        if cid not in cluster_centers or gid not in gt_map:
            continue

        pred_center = cluster_centers[cid]
        gt_center = gt_map[gid]
        rows.append({
            "Frame": int(fid),
            "cid": int(cid),
            "gid": int(gid),
            "pred_x": float(pred_center[0]),
            "pred_y": float(pred_center[1]),
            "gt_x": float(gt_center["x"]),
            "gt_y": float(gt_center["y"]),
            "dx": float(match["dx"]),
            "dy": float(match["dy"]),
            "model": int(gt_center["model"]),
        })
    return rows


def enrich_point_table_with_gt_model(point_table, gt_list):
    gt_model_map = {int(g["id"]): int(g["model"]) for g in gt_list}
    if "gid" in point_table.columns:
        point_table["gt_model"] = point_table["gid"].map(gt_model_map)
    return point_table


def run_pipeline(frame_ids, radar_data, gt_df, args, cfg, center_fn, bias_fn, tracker, debug_tool):
    stats = init_stats()
    point_tables = []
    cache = {}
    tp_match_rows = []
    range_bins = [(0.0, 100.0), (100.0, 1e9)]
    range_bias_stats = {rb: [] for rb in range_bins}

    for fid in frame_ids:
        result = process_one_frame(
            fid=fid,
            radar_data=radar_data,
            gt_df=gt_df,
            fit_mode=args.fit_mode,
            cfg=cfg,
            center_fn=center_fn,
            bias_fn=bias_fn,
            tracker=tracker,
        )

        metrics_raw = result.get("metrics_raw")
        metrics_filtered = result.get("metrics")
        if metrics_raw is not None and metrics_filtered is not None:
            print(
                "[Frame %s] raw_mean_err=%.3f -> filtered_mean_err=%.3f"
                % (fid, metrics_raw["mean_center_error"], metrics_filtered["mean_center_error"])
            )

        if debug_tool is not None:
            debug_tool.update(
                fid,
                metrics_raw,
                metrics_filtered,
                result["cache_item"].get("cluster_centers", {}),
                result["cache_item"].get("track_assignments", {}),
                result.get("gt_list", []),
            )

        cluster_centers = result["cache_item"].get("cluster_centers", {})
        gt_list = result["gt_list"]
        tp_match_rows.extend(collect_tp_match_rows(fid, cluster_centers, gt_list, result["metrics"]))

        point_table = enrich_point_table_with_gt_model(result["point_table"], gt_list)
        point_tables.append(point_table)

        update_stats(stats, result["metrics"])
        update_range_bias_stats(range_bias_stats, range_bins, result["metrics"].get("matches", []), gt_list)
        cache[fid] = result["cache_item"]

    return stats, point_tables, cache, tp_match_rows, range_bins, range_bias_stats


def export_results(cfg, point_tables, tp_match_rows):
    export_point_table(point_tables, cfg.EXPORT_CSV_PATH, cfg.EXPORT_XLSX_PATH)
    export_tp_matches(tp_match_rows, cfg.TP_MATCH_CSV_PATH)
    export_tp_matches_excel(tp_match_rows, cfg.TP_MATCH_XLSX_PATH)


def launch_viewer(cache, frame_ids, cfg, fit_mode="center"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.08)
    state = {"i": 0}

    def on_key(event):
        key = (event.key or "").lower()
        if key == "n":
            render_frame(fig, axes, cache, frame_ids, len(frame_ids), state, state["i"] + 1, fit_mode, cfg)
        elif key == "p":
            render_frame(fig, axes, cache, frame_ids, len(frame_ids), state, state["i"] - 1, fit_mode, cfg)
        elif key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render_frame(fig, axes, cache, frame_ids, len(frame_ids), state, 0, fit_mode, cfg)
    return fig, axes


def main():
    args = parse_args()
    cfg = Config()

    cfg.CLUSTER_CENTER_MODE = args.center_mode
    cfg.BIAS_MODE = args.bias_mode
    cfg.TRIMMED_MEAN_RATIO = float(args.trim_ratio)

    radar_data, gt_df = load_all_data(cfg)
    frame_ids = get_frame_ids(radar_data, gt_df, cfg, args)

    print("Will view frames:", frame_ids[:10], "..." if len(frame_ids) > 10 else "")
    print("FIT MODE:", args.fit_mode)
    print("CENTER MODE:", cfg.CLUSTER_CENTER_MODE)
    print("BIAS MODE:", cfg.BIAS_MODE)
    print("TRIM RATIO:", cfg.TRIMMED_MEAN_RATIO)
    print("MODEL PRIORS:", cfg.GT_MODEL_PRIORS)
    print("VIEWER MODE:", cfg.VIEWER_MODE)

    print("KF_INIT_POS_VAR:", cfg.KF_INIT_POS_VAR)
    print("KF_INIT_VEL_VAR:", cfg.KF_INIT_VEL_VAR)


    center_fn = get_center_function(cfg.CLUSTER_CENTER_MODE)
    bias_fn = get_bias_function(cfg.BIAS_MODE)
    tracker = build_tracker(cfg)
    debug_tool = TrackingDebugTool() if cfg.ENABLE_TEMPORAL_DEBUG else None

    stats, point_tables, cache, tp_match_rows, range_bins, range_bias_stats = run_pipeline(
        frame_ids=frame_ids,
        radar_data=radar_data,
        gt_df=gt_df,
        args=args,
        cfg=cfg,
        center_fn=center_fn,
        bias_fn=bias_fn,
        tracker=tracker,
        debug_tool=debug_tool,
    )

    export_results(cfg, point_tables, tp_match_rows)
    if getattr(cfg, "ENABLE_TRACK_DIAGNOSTICS", False):
        track_frame_df = collect_track_records(cache, frame_ids)
        track_summary_df = build_track_summary(track_frame_df)
        track_diag_summary = build_track_diagnostic_summary(track_frame_df, track_summary_df)

        print_track_diagnostic_summary(track_diag_summary)

        export_track_diagnostics(
            track_frame_df=track_frame_df,
            track_summary_df=track_summary_df,
            summary=track_diag_summary,
            frame_csv_path=cfg.TRACK_DIAG_FRAME_CSV_PATH,
            track_csv_path=cfg.TRACK_DIAG_SUMMARY_CSV_PATH,
            summary_csv_path=cfg.TRACK_DIAG_GLOBAL_CSV_PATH,
        )
        
    if cfg.VIEWER_ENABLE:
        if str(getattr(cfg, "VIEWER_MODE", "static")).lower() == "animated":
            launch_animated_viewer(cache, frame_ids, cfg, fit_mode=args.fit_mode)
        else:
            launch_viewer(cache, frame_ids, cfg, fit_mode=args.fit_mode)

    print_global_summary(frame_ids, stats, range_bins, range_bias_stats)
    print("TRACKER METHOD:", cfg.TRACKER_METHOD)
    print("STRICT CV MODE:", getattr(cfg, "TRACK_STRICT_CONSTANT_VELOCITY", False))

    if debug_tool is not None:
        debug_tool.show()

    plt.show()


if __name__ == "__main__":
    main()
