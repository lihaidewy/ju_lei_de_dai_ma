import argparse
import matplotlib.pyplot as plt

from centers import get_bias_function, get_center_function
from config import Config
from data_pipeline import get_frame_ids, load_all_data, process_one_frame
from debug_tracking import TrackingDebugTool
from exporters import export_point_table, export_tp_matches, export_tp_matches_excel
from online_tracker import OnlineTrackerManager
from stats_utils import init_stats, print_global_summary, update_range_bias_stats, update_stats
from viz_utils import render_frame


CENTER_MODE_CHOICES = [
    "mean",
    "median",
    "snr_mean",
    "trimmed_mean",
    "mean_x_median_y",
    "velocity_mean",
    "velocity_trimmed_mean",
    "fixed_box",
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
    return OnlineTrackerManager(
        assoc_dist_thr=cfg.TRACK_ASSOC_DIST_THR,
        max_misses=cfg.TRACK_MAX_MISSES,
        dt=cfg.KF_DT,
        q_pos=cfg.KF_Q_POS,
        q_vel=cfg.KF_Q_VEL,
        r_pos=cfg.KF_R_POS,
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


def launch_viewer(cache, frame_ids, args, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.08)
    state = {"i": 0}

    def on_key(event):
        key = (event.key or "").lower()
        if key == "n":
            render_frame(fig, axes, cache, frame_ids, len(frame_ids), state, state["i"] + 1, args.fit_mode, cfg)
        elif key == "p":
            render_frame(fig, axes, cache, frame_ids, len(frame_ids), state, state["i"] - 1, args.fit_mode, cfg)
        elif key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render_frame(fig, axes, cache, frame_ids, len(frame_ids), state, 0, args.fit_mode, cfg)
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
    launch_viewer(cache, frame_ids, args, cfg)
    print_global_summary(frame_ids, stats, range_bins, range_bias_stats)

    if debug_tool is not None:
        debug_tool.show()

    plt.show()


if __name__ == "__main__":
    main()
