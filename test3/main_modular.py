import argparse
import matplotlib.pyplot as plt

from config import Config
from centers import get_center_function, apply_two_segment_bias
from data_pipeline import load_all_data, get_frame_ids, process_one_frame
from exporters import export_point_table, export_tp_matches, export_tp_matches_excel
from stats_utils import init_stats, update_stats, update_range_bias_stats, print_global_summary
from viz_utils import render_frame
from online_tracker import OnlineTrackerManager
from debug_tracking import TrackingDebugTool


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fit_mode", type=str, default="center", choices=["center", "edge"])
    p.add_argument("--max_frames", type=int, default=Config.MAX_FRAMES_TO_VIEW)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    radar_data, gt_df = load_all_data(cfg)
    frame_ids = get_frame_ids(radar_data, gt_df, cfg, args)

    print("Will view frames:", frame_ids[:10], "..." if len(frame_ids) > 10 else "")
    print("FIT MODE:", args.fit_mode)

    center_fn = get_center_function(cfg.CLUSTER_CENTER_MODE)
    bias_fn = apply_two_segment_bias

    # ===== 在线 tracker =====
    tracker = None
    if getattr(cfg, "USE_ONLINE_TRACKER", False):
        tracker = OnlineTrackerManager(
            assoc_dist_thr=cfg.TRACK_ASSOC_DIST_THR,
            max_misses=cfg.TRACK_MAX_MISSES,
            dt=cfg.KF_DT,
            q_pos=cfg.KF_Q_POS,
            q_vel=cfg.KF_Q_VEL,
            r_pos=cfg.KF_R_POS,
        )


    stats = init_stats()
    point_tables = []
    cache = {}
    tp_match_rows = []

    # ===== tracking 调试工具 =====
    debug_tool = TrackingDebugTool()

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

        # ===== 每帧打印 raw vs filtered 误差 =====
        if "metrics_raw" in result and result["metrics_raw"] is not None:
            mr = result["metrics_raw"]
            mf = result["metrics"]
            print(
                f"[Frame {fid}] "
                f"raw_mean_err={mr['mean_center_error']:.3f} -> "
                f"filtered_mean_err={mf['mean_center_error']:.3f}"
            )

        # ===== tracking 调试记录 =====
        debug_tool.update(
            fid,
            result.get("metrics_raw"),
            result.get("metrics"),
            result["cache_item"].get("cluster_centers", {}),
            result["cache_item"].get("track_assignments", {}),
            result.get("gt_list", []),
        )

        cluster_centers = result["cache_item"].get("cluster_centers", {})
        gt_list = result["gt_list"]

        gt_map = {
            int(g["id"]): {
                "x": float(g["x"]),
                "y": float(g["y"]),
                "model": int(g["model"]),
            }
            for g in gt_list
        }

        for mm in result["metrics"].get("matches", []):
            cid = int(mm["cid"])
            gid = int(mm["gid"])

            if cid not in cluster_centers:
                continue
            if gid not in gt_map:
                continue

            pred_center = cluster_centers[cid]
            gt_center = gt_map[gid]

            tp_match_rows.append({
                "Frame": int(fid),
                "cid": int(cid),
                "gid": int(gid),
                "pred_x": float(pred_center[0]),
                "pred_y": float(pred_center[1]),
                "gt_x": float(gt_center["x"]),
                "gt_y": float(gt_center["y"]),
                "dx": float(mm["dx"]),
                "dy": float(mm["dy"]),
                "model": int(gt_center["model"]),
            })

        pt = result["point_table"]

        # 给每个点补 GT 模型
        gt_model_map = {int(g["id"]): int(g["model"]) for g in result["gt_list"]}
        if "gid" in pt.columns:
            pt["gt_model"] = pt["gid"].map(gt_model_map)

        point_tables.append(pt)

        update_stats(stats, result["metrics"])
        update_range_bias_stats(
            range_bias_stats,
            range_bins,
            result["metrics"].get("matches", []),
            result["gt_list"]
        )

        cache[fid] = result["cache_item"]

    export_point_table(point_tables, cfg.EXPORT_CSV_PATH, cfg.EXPORT_XLSX_PATH)
    export_tp_matches(tp_match_rows, "data/tp_matches_for_bias.csv")
    export_tp_matches_excel(tp_match_rows, "data/tp_matches_for_bias.xlsx")

    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.08)

    state = {"i": 0}

    def on_key(event):
        k = (event.key or "").lower()
        if k == "n":
            render_frame(fig, axes, cache, frame_ids, len(frame_ids), state, state["i"] + 1, args.fit_mode, cfg)
        elif k == "p":
            render_frame(fig, axes, cache, frame_ids, len(frame_ids), state, state["i"] - 1, args.fit_mode, cfg)
        elif k in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render_frame(fig, axes, cache, frame_ids, len(frame_ids), state, 0, args.fit_mode, cfg)

    print_global_summary(frame_ids, stats, range_bins, range_bias_stats)

    # ===== 调试图 =====
    debug_tool.show()

    plt.show()


if __name__ == "__main__":
    main()
