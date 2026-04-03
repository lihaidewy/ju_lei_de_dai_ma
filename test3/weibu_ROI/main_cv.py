import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import Config
from data_pipeline import load_all_data, cluster_one_frame
from analysis_cv import (
    build_valid_frames,
    generate_probability_table_global_y,
    normalize_probability_table,
    print_probability_summary,
    print_summary,
    run_global_y_prob_weighted_analysis,
)
from visualization_cv import launch_viewer


def build_cluster_label_map(radar_data, frame_ids, cfg):
    """
    对每个有效帧做聚类，并保存 label。
    """
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


def split_frames_for_prob_and_eval(frame_ids, params):
    """
    把帧划分成：
    1) 概率表构建帧
    2) 最终评估帧
    """
    use_same = bool(params.get("USE_SAME_DATA_FOR_PROB_AND_EVAL", True))
    if use_same:
        return list(frame_ids), list(frame_ids)

    if len(frame_ids) < 2:
        raise ValueError("frame_ids 数量不足，无法分离构建概率表与评估。")

    prob_ratio = float(params.get("PROB_FRAME_RATIO", 0.5))
    prob_ratio = min(max(prob_ratio, 0.1), 0.9)
    cut = int(round(len(frame_ids) * prob_ratio))
    cut = min(max(cut, 1), len(frame_ids) - 1)

    prob_frame_ids = list(frame_ids[:cut])
    eval_frame_ids = list(frame_ids[cut:])
    return prob_frame_ids, eval_frame_ids


def main():
    params = {
        # =========================
        # 帧范围
        # =========================
        "START_FRAME": 100,
        "END_FRAME": 399,

        # =========================
        # ROI 参数
        # =========================
        "ROI_OUTER": 1.0,
        "ROI_INNER": 1.0,

        # =========================
        # GT 尺寸
        # =========================
        "GT_DIM": {
            0: {"L": 5.06, "W": 2.22},
            1: {"L": 4.32, "W": 2.19},
            2: {"L": 3.55, "W": 2.58},
        },

        # =========================
        # 概率表相关
        # =========================
        "GENERATE_PROBABILITY_TABLE_IN_MAIN": True,
        "USE_SAME_DATA_FOR_PROB_AND_EVAL": True,
        "PROB_FRAME_RATIO": 0.5,
        "U_BIN_COUNT": 20,
        "MIN_LOOKUP_WEIGHT": 0.0,

        "SAVE_PROBABILITY_CSV": True,
        "PROBABILITY_CSV_PATH": "cluster_roi_probability_by_u_global_y.csv",
        "SAVE_PROBABILITY_POINT_CSV": False,
        "PROBABILITY_POINT_CSV_PATH": "cluster_roi_core_points_global_y.csv",

        # =========================
        # 全局 Y 参考
        # =========================
        "SENSOR_Y": 0.0,

        # =========================
        # 时序模式
        # raw / cv_recursive
        # =========================
        "ESTIMATION_MODE": "cv_recursive",

        # =========================
        # 简单 CV 递推参数
        # =========================
        "CV_MAX_MISSES": 10,
        "CV_INIT_X_SIGN_EPS": 1.0,

        # x > 0: 远离
        "CV_INIT_VY_POS": 1.1,

        # x < 0: 接近
        "CV_INIT_VY_NEG": -1.1,

        # |x| 很小时
        "CV_INIT_VY_CENTER": 0.0,

        # =========================
        # 概率加权量测参数
        # =========================
        "WEIGHTED_MEASUREMENT_MODE": "weighted_median",
        "WEIGHT_KEEP_THRESHOLD": 0.4,
        "WEIGHT_MIN_SUM": 1e-6,

        # =========================
        # 结果导出
        # =========================
        "SAVE_RESULT_CSV": True,
        "RESULT_CSV_PATH": "global_y_prob_weighted_cluster_results_cv_recursive.csv",

        # =========================
        # 最近邻目标-cluster 关联参数
        # =========================
        "ASSOC_REFERENCE": "gt_center",
        "ASSOC_CLUSTER_CENTER_MODE": "median",
        "ASSOC_GATE_X": 2.0,
        "ASSOC_GATE_Y": 4.0,
        "ASSOC_DIST_WEIGHT_X": 1.0,
        "ASSOC_DIST_WEIGHT_Y": 2.0,

        # =========================
        # 中心几何补偿参数
        # =========================
        "ENABLE_CENTER_COMPENSATION": True,
        "CENTER_COMP_MODE": "constant",
        "CENTER_COMP_DISTANCE": 1.56 + 0.45,
        "CENTER_COMP_ALPHA": 1.0,
        "CENTER_COMP_BY_MODEL": {
            0: 1.75,
            1: 1.45,
            2: 1.25,
        },

        # =========================
        # 可视化参数
        # =========================
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

    print("GT columns:")
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

    # print("\n有效帧:")
    # print(frame_ids)

    cluster_label_map = build_cluster_label_map(
        radar_data=radar_data,
        frame_ids=frame_ids,
        cfg=cfg,
    )

    prob_frame_ids, eval_frame_ids = split_frames_for_prob_and_eval(frame_ids, params)

    print("\n===== 概率表构建帧 =====")
    print(prob_frame_ids)
    print("\n===== 评估帧 =====")
    print(eval_frame_ids)

    if not bool(params.get("GENERATE_PROBABILITY_TABLE_IN_MAIN", True)):
        raise ValueError(
            "当前版本设计为主函数内生成概率表，请将 GENERATE_PROBABILITY_TABLE_IN_MAIN 设为 True。"
        )

    point_df, raw_prob_df = generate_probability_table_global_y(
        radar_data=radar_data,
        gt_df=gt_df,
        frame_ids=prob_frame_ids,
        cluster_label_map=cluster_label_map,
        params=params,
        num_u_bins=int(params.get("U_BIN_COUNT", 20)),
    )

    prob_df = normalize_probability_table(
        raw_prob_df,
        min_weight=params.get("MIN_LOOKUP_WEIGHT", 0.0),
    )

    print_probability_summary(raw_prob_df)

    if params.get("SAVE_PROBABILITY_CSV", False):
        prob_path = params.get(
            "PROBABILITY_CSV_PATH",
            "cluster_roi_probability_by_u_global_y.csv",
        )
        raw_prob_df.to_csv(prob_path, index=False, encoding="utf-8-sig")
        print(f"\n概率表已保存到: {prob_path}")

    if params.get("SAVE_PROBABILITY_POINT_CSV", False):
        point_path = params.get(
            "PROBABILITY_POINT_CSV_PATH",
            "cluster_roi_core_points_global_y.csv",
        )
        point_df.to_csv(point_path, index=False, encoding="utf-8-sig")
        print(f"点级分布已保存到: {point_path}")

    result_df, frame_cache = run_global_y_prob_weighted_analysis(
        radar_data=radar_data,
        gt_df=gt_df,
        frame_ids=eval_frame_ids,
        cluster_label_map=cluster_label_map,
        prob_df=prob_df,
        params=params,
    )

    print_summary(result_df, params)

    if params.get("SAVE_RESULT_CSV", False):
        csv_path = params.get(
            "RESULT_CSV_PATH",
            "global_y_prob_weighted_cluster_results_cv_recursive.csv",
        )
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n逐目标结果已保存到: {csv_path}")

    if params.get("ENABLE_VIEWER", False):
        launch_viewer(eval_frame_ids, frame_cache, params)


if __name__ == "__main__":
    main()
