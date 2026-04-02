import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import Config
from data_pipeline import load_all_data, cluster_one_frame

from analysis import (
    analyze_cluster_roi_core_metrics,
    build_valid_frames,
    print_cluster_roi_core_metrics_summary,
    print_summary,
    run_analysis,
)
from visualization import launch_viewer


def attach_cluster_labels_to_frame_cache(radar_data, frame_ids, frame_cache, cfg):
    """
    在保留原有 run_analysis/frame_cache 结果的前提下，
    为每个 frame 额外补充 DBSCAN 聚类 labels，
    供 visualization.py 在右图进行着色显示。
    """
    for fid in frame_ids:
        labels = cluster_one_frame(radar_data, fid, cfg)
        frame_item = radar_data[fid]

        num_points = len(frame_item["X"])
        if len(labels) != num_points:
            raise ValueError(
                f"Frame {fid}: labels 数量({len(labels)})与点数量({num_points})不一致"
            )

        if fid not in frame_cache:
            frame_cache[fid] = {}

        frame_cache[fid]["cluster_labels"] = labels

    return frame_cache


def main():
    params = {
        "START_FRAME": 100,
        "END_FRAME": 399,

        "ROI_OUTER": 1.0,
        "ROI_INNER": 1.0,

        "GT_DIM": {
            0: {"L": 5.06, "W": 2.22},
            1: {"L": 4.32, "W": 2.19},
            2: {"L": 3.55, "W": 2.58},
        },

        # 统一开关：
        # "kalman" -> 原 Kalman CV
        # "raw"    -> 直接使用 ROI 原始量测
        # "cv_fit" -> 滑动窗口匀速拟合
        "ESTIMATION_MODE": "cv_fit",

        # Kalman 参数（仅在 ESTIMATION_MODE="kalman" 时生效）
        "KF_DT": 1.0,
        "KF_Q_POS": 0.01,
        "KF_Q_VEL": 0.01,
        "KF_R_POS": 1.50,
        "KF_INIT_POS_VAR": 4.0,
        "KF_INIT_VEL_VAR": 9.0,

        # ROI 量测要求
        "MIN_ROI_POINTS": 1,

        # 滑动窗口匀速拟合参数（仅在 ESTIMATION_MODE="cv_fit" 时生效）
        "CVFIT_WINDOW_SIZE": 5,
        "CVFIT_MIN_POINTS": 2,
        "CVFIT_HISTORY_MAXLEN": 20,

        "ENABLE_VIEWER": True,
        "SAVE_RESULT_CSV": False,
        "RESULT_CSV_PATH": "edge_midpoint_results.csv",

        # 核心指标输出
        "SAVE_CLUSTER_ROI_CORE_CSV": True,
        "CLUSTER_ROI_POINT_CSV_PATH": "cluster_roi_core_points.csv",
        "CLUSTER_ROI_PROB_CSV_PATH": "cluster_roi_probability_by_u.csv",
        "CLUSTER_ROI_WINDOW_CSV_PATH": "cluster_roi_window_eval.csv",
        "U_NUM_BINS": 20,
        "WINDOW_SUCCESS_THRESHOLD": 0.50,
        "CANDIDATE_WINDOWS": [
            {"name": "front_20", "intervals": [(0.00, 0.20)]},
            {"name": "front_30", "intervals": [(0.00, 0.30)]},
            {"name": "dual_peak", "intervals": [(0.05, 0.25), (0.50, 0.60)]},
        ],

        "VIEW_XMIN": -20,
        "VIEW_XMAX": 20,
        "VIEW_YMIN": 0,
        "VIEW_YMAX": 235,
        "VIEW_XTICK_STEP": 5,
        "VIEW_YTICK_STEP": 5,
    }

    cfg = Config()
    radar_data, gt_df = load_all_data(cfg)
    print(gt_df.columns.tolist())

    frame_ids = build_valid_frames(
        radar_data,
        gt_df,
        params["START_FRAME"],
        params["END_FRAME"],
    )
    if not frame_ids:
        raise ValueError(
            f"在 {params['START_FRAME']} 到 {params['END_FRAME']} 帧之间没有找到公共帧。"
        )

    print("有效帧:")
    print(frame_ids)

    result_df, frame_cache = run_analysis(radar_data, gt_df, frame_ids, params)
    print_summary(result_df, params)

    frame_cache = attach_cluster_labels_to_frame_cache(
        radar_data=radar_data,
        frame_ids=frame_ids,
        frame_cache=frame_cache,
        cfg=cfg,
    )

    point_df, prob_df, window_df = analyze_cluster_roi_core_metrics(
        frame_ids=frame_ids,
        frame_cache=frame_cache,
        num_u_bins=params.get("U_NUM_BINS", 20),
        candidate_windows=params.get("CANDIDATE_WINDOWS"),
        success_threshold=params.get("WINDOW_SUCCESS_THRESHOLD", 0.50),
    )
    print_cluster_roi_core_metrics_summary(
        prob_df=prob_df,
        window_df=window_df,
        success_threshold=params.get("WINDOW_SUCCESS_THRESHOLD", 0.50),
    )

    if params.get("SAVE_RESULT_CSV", False):
        csv_path = params.get("RESULT_CSV_PATH", "edge_midpoint_results.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n逐目标结果已保存到: {csv_path}")

    if params.get("SAVE_CLUSTER_ROI_CORE_CSV", True):
        point_path = params.get("CLUSTER_ROI_POINT_CSV_PATH", "cluster_roi_core_points.csv")
        prob_path = params.get("CLUSTER_ROI_PROB_CSV_PATH", "cluster_roi_probability_by_u.csv")
        window_path = params.get("CLUSTER_ROI_WINDOW_CSV_PATH", "cluster_roi_window_eval.csv")

        point_df.to_csv(point_path, index=False, encoding="utf-8-sig")
        prob_df.to_csv(prob_path, index=False, encoding="utf-8-sig")
        window_df.to_csv(window_path, index=False, encoding="utf-8-sig")

        print(f"\n核心点级数据已保存到: {point_path}")
        print(f"u 概率表已保存到: {prob_path}")
        print(f"候选窗口评估表已保存到: {window_path}")

    if params["ENABLE_VIEWER"]:
        launch_viewer(frame_ids, frame_cache, params)


if __name__ == "__main__":
    main()
