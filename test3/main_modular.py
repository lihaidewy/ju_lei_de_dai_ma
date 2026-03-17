import argparse
import matplotlib.pyplot as plt

from centers import apply_two_segment_bias, get_center_function
from config import Config
from data_pipeline import get_frame_ids, load_all_data, process_one_frame
from debug_tracking import TrackingDebugTool
from exporters import export_point_table, export_tp_matches, export_tp_matches_excel
from online_tracker import OnlineTrackerManager
from stats_utils import init_stats, print_global_summary, update_range_bias_stats, update_stats
from viz_utils import render_frame


def parse_args():
    defaults = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--fit_mode", type=str, default="center", choices=["center", "edge"])
    p.add_argument("--max_frames", type=int, default=defaults.MAX_FRAMES_TO_VIEW)
    p.add_argument(
        "--tracker_method",
        type=str,
        default=getattr(defaults, "TRACKER_METHOD", "cv"),
        choices=["cv", "ca", "cv_robust"],
        help="cv: 原方法; ca: 恒加速度+马氏门控; cv_robust: 匀速+鲁棒关联",
    )
    p.add_argument(
        "--assoc_metric",
        type=str,
        default=getattr(defaults, "TRACK_ASSOC_METRIC", "euclidean"),
        choices=["euclidean", "mahalanobis"],
        help="轨迹关联代价类型",
    )
    return p.parse_args()


def build_tracker(cfg, args):
    if not getattr(cfg, "USE_ONLINE_TRACKER", False):
        return None

    return OnlineTrackerManager(
        method=getattr(args, "tracker_method", getattr(cfg, "TRACKER_METHOD", "cv")),
        assoc_metric=getattr(args, "assoc_metric", getattr(cfg, "TRACK_ASSOC_METRIC", "euclidean")),
        assoc_dist_thr=cfg.TRACK_ASSOC_DIST_THR,
        assoc_mahal_thr=getattr(cfg, "TRACK_ASSOC_MAHAL_THR", 3.5),
        max_misses=cfg.TRACK_MAX_MISSES,
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
        output_ema_alpha=getattr(cfg, "TRACK_OUTPUT_EMA_ALPHA", 0.65),
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
                "[Frame {fid}] raw_mean_err={raw:.3f} -> filtered_mean_err={filt:.3f}".format(
                    fid=fid,
                    raw=metrics_raw["mean_center_error"],
                    filt=metrics_filtered["mean_center_error"],
                )
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

    radar_data, gt_df = load_all_data(cfg)
    frame_ids = get_frame_ids(radar_data, gt_df, cfg, args)

    print("Will view frames:", frame_ids[:10], "..." if len(frame_ids) > 10 else "")
    print("FIT MODE:", args.fit_mode)
    print("TRACKER METHOD:", getattr(args, "tracker_method", getattr(cfg, "TRACKER_METHOD", "cv")))
    print("ASSOC METRIC:", getattr(args, "assoc_metric", getattr(cfg, "TRACK_ASSOC_METRIC", "euclidean")))

    center_fn = get_center_function(cfg.CLUSTER_CENTER_MODE)
    bias_fn = apply_two_segment_bias
    tracker = build_tracker(cfg, args)
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
