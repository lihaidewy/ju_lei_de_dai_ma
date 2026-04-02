import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import Config
from data_pipeline import load_all_data, cluster_one_frame
from analysis2 import (
    build_valid_frames,
    load_probability_lookup,
    print_summary,
    run_prob_weighted_analysis,
)
from visualization import launch_viewer
"""
这个版本的 main.py 专注于实现基于概率加权的聚类 ROI"""

def build_cluster_label_map(radar_data, frame_ids, cfg):
    cluster_label_map = {}
    for fid in frame_ids:
        labels = cluster_one_frame(radar_data, fid, cfg)
        frame_item = radar_data[fid]
        num_points = len(frame_item["X"])
        if len(labels) != num_points:
            raise ValueError(
                f"Frame {fid}: labels 数量({len(labels)})与点数量({num_points})不一致"
            )
        cluster_label_map[fid] = np.asarray(labels)
    return cluster_label_map


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

        # 概率加权后的时序模式:
        #   raw    : 当前帧概率加权量测直接输出
        #   kalman : 对概率加权量测做 Kalman
        #   cv_fit : 对概率加权量测做滑窗匀速拟合
        "ESTIMATION_MODE": "cv_fit",

        # Kalman 参数
        "KF_DT": 1.0,
        "KF_Q_POS": 0.01,
        "KF_Q_VEL": 0.01,
        "KF_R_POS": 1.50,
        "KF_INIT_POS_VAR": 4.0,
        "KF_INIT_VEL_VAR": 9.0,

        # CV fit 参数
        "CVFIT_WINDOW_SIZE": 5,
        "CVFIT_MIN_POINTS": 2,
        "CVFIT_HISTORY_MAXLEN": 20,

        # 概率表路径
        "PROBABILITY_CSV_PATH": "cluster_roi_probability_by_u.csv",

        # 点级量测生成方式:
        #   weighted_mean / weighted_median
        "WEIGHTED_MEASUREMENT_MODE": "weighted_mean",

        # 保留点阈值: 仅保留 weight >= threshold 的点。
        # 建议先从 0.0 开始，避免过早硬截断。
        "WEIGHT_KEEP_THRESHOLD": 0.0,

        # 加权均值的最小权重和
        "WEIGHT_MIN_SUM": 1e-6,

        # 概率表中的最小权重截断，避免 NaN 或过低值
        "MIN_LOOKUP_WEIGHT": 0.0,

        "SAVE_RESULT_CSV": True,
        "RESULT_CSV_PATH": "prob_weighted_cluster_results.csv",

        "ENABLE_VIEWER": True,
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
        radar_data=radar_data,
        gt_df=gt_df,
        start_frame=params["START_FRAME"],
        end_frame=params["END_FRAME"],
    )
    if not frame_ids:
        raise ValueError(
            f"在 {params['START_FRAME']} 到 {params['END_FRAME']} 帧之间没有找到公共帧。"
        )

    print("有效帧:")
    print(frame_ids)

    prob_df = load_probability_lookup(
        prob_csv_path=params["PROBABILITY_CSV_PATH"],
        min_weight=params.get("MIN_LOOKUP_WEIGHT", 0.0),
    )
    print("\n===== 已加载概率表 =====")
    print(prob_df)

    cluster_label_map = build_cluster_label_map(
        radar_data=radar_data,
        frame_ids=frame_ids,
        cfg=cfg,
    )

    result_df, frame_cache = run_prob_weighted_analysis(
        radar_data=radar_data,
        gt_df=gt_df,
        frame_ids=frame_ids,
        cluster_label_map=cluster_label_map,
        prob_df=prob_df,
        params=params,
    )
    print_summary(result_df)

    if params.get("SAVE_RESULT_CSV", False):
        csv_path = params.get("RESULT_CSV_PATH", "prob_weighted_cluster_results.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n逐目标结果已保存到: {csv_path}")

    if params.get("ENABLE_VIEWER", False):
        launch_viewer(frame_ids, frame_cache, params)


if __name__ == "__main__":
    main()
