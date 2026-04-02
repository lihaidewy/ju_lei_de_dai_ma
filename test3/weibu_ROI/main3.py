import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import Config
from data_pipeline import load_all_data, cluster_one_frame
from analysis3 import (
    build_valid_frames,
    generate_probability_table_global_y,
    normalize_probability_table,
    print_probability_summary,
    print_summary,
    run_global_y_prob_weighted_analysis,
)
from visualization import launch_viewer


def build_cluster_label_map(radar_data, frame_ids, cfg):
    """"""
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
    """把帧划分为两部分:
    1) 用于生成概率表
    2) 用于最终评估
    两种模式:
    - USE_SAME_DATA_FOR_PROB_AND_EVAL = True
        同一批帧既用来建概率表，也用来评估
    - USE_SAME_DATA_FOR_PROB_AND_EVAL = False
        按 PROB_FRAME_RATIO 切分:
            前半部分 -> 构建概率表
            后半部分 -> 评估
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


def _robust_group_stats(values, fallback_median, fallback_var):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(fallback_median), float(fallback_var), 0

    median = float(np.median(values))
    if values.size >= 2:
        var = float(np.var(values))
    else:
        var = float(fallback_var)

    var = max(var, 1e-3)
    return median, var, int(values.size)


def estimate_directional_speed_priors_from_gt(gt_df, frame_ids, params):
    """
    从训练帧 GT 中估计按 x 分组的 vy 初始先验。
    规则:
      - x > +eps : 远离组
      - x < -eps : 接近组
      - |x| <= eps : 中间组

    统计方法:
      - 对同一 ID 的相邻帧，计算 vy = dY / dFrame
      - 用 vy 的中位数作为该组初始速度
      - 用 vy 的方差作为该组初始速度方差

    注意：
      这里不直接 print，避免输出被中间大量逐帧信息冲掉。
      统计结果写回 params，最后统一打印。
    """
    eps = float(params.get("KF_INIT_X_SIGN_EPS", 1.0))

    # 默认回退值
    default_speed_abs = float(params.get("KF_INIT_SPEED_ABS_FALLBACK", 1.1))
    default_var = float(params.get("KF_INIT_VEL_VAR", 9.0))

    use_df = gt_df[gt_df["Frame"].astype(int).isin(list(frame_ids))].copy()
    if use_df.empty:
        out = dict(params)
        out["KF_INIT_VY_POS"] = +default_speed_abs
        out["KF_INIT_VY_NEG"] = -default_speed_abs
        out["KF_INIT_VY_CENTER"] = 0.0
        out["KF_INIT_VEL_VAR_POS"] = default_var
        out["KF_INIT_VEL_VAR_NEG"] = default_var
        out["KF_INIT_VEL_VAR_CENTER"] = default_var
        out["KF_PRIOR_STAT_N_POS"] = 0
        out["KF_PRIOR_STAT_N_NEG"] = 0
        out["KF_PRIOR_STAT_N_CENTER"] = 0
        return out

    use_df = use_df.sort_values(["ID", "Frame"]).reset_index(drop=True)

    pos_vy = []
    neg_vy = []
    center_vy = []

    for gid, sub in use_df.groupby("ID"):
        sub = sub.sort_values("Frame")
        frames = sub["Frame"].astype(int).values
        xs = sub["X"].astype(float).values
        ys = sub["Y"].astype(float).values

        if len(sub) < 2:
            continue

        for i in range(len(sub) - 1):
            f0 = int(frames[i])
            f1 = int(frames[i + 1])
            df = f1 - f0
            if df <= 0:
                continue

            x0 = float(xs[i])
            dy = float(ys[i + 1] - ys[i])
            vy = dy / float(df)

            if x0 > eps:
                pos_vy.append(vy)
            elif x0 < -eps:
                neg_vy.append(vy)
            else:
                center_vy.append(vy)

    pos_med, pos_var, n_pos = _robust_group_stats(
        pos_vy, fallback_median=+default_speed_abs, fallback_var=default_var
    )
    neg_med, neg_var, n_neg = _robust_group_stats(
        neg_vy, fallback_median=-default_speed_abs, fallback_var=default_var
    )
    cen_med, cen_var, n_cen = _robust_group_stats(
        center_vy, fallback_median=0.0, fallback_var=default_var
    )

    # 按场景先验校正符号
    if pos_med <= 0:
        pos_med = +abs(pos_med) if abs(pos_med) > 1e-6 else +default_speed_abs
    if neg_med >= 0:
        neg_med = -abs(neg_med) if abs(neg_med) > 1e-6 else -default_speed_abs

    out = dict(params)
    out["KF_INIT_VY_POS"] = float(pos_med)
    out["KF_INIT_VY_NEG"] = float(neg_med)
    out["KF_INIT_VY_CENTER"] = float(cen_med)

    out["KF_INIT_VEL_VAR_POS"] = float(max(pos_var, 1e-3))
    out["KF_INIT_VEL_VAR_NEG"] = float(max(neg_var, 1e-3))
    out["KF_INIT_VEL_VAR_CENTER"] = float(max(cen_var, 1e-3))

    out["KF_PRIOR_STAT_N_POS"] = int(n_pos)
    out["KF_PRIOR_STAT_N_NEG"] = int(n_neg)
    out["KF_PRIOR_STAT_N_CENTER"] = int(n_cen)

    return out


def print_directional_speed_priors(params):
    print("\n===== 数据驱动速度先验统计 =====")
    print(
        f"x > +eps 组: "
        f"n={int(params.get('KF_PRIOR_STAT_N_POS', 0))}, "
        f"KF_INIT_VY_POS={float(params.get('KF_INIT_VY_POS', 0.0)):.4f}, "
        f"KF_INIT_VEL_VAR_POS={float(params.get('KF_INIT_VEL_VAR_POS', 0.0)):.4f}"
    )
    print(
        f"x < -eps 组: "
        f"n={int(params.get('KF_PRIOR_STAT_N_NEG', 0))}, "
        f"KF_INIT_VY_NEG={float(params.get('KF_INIT_VY_NEG', 0.0)):.4f}, "
        f"KF_INIT_VEL_VAR_NEG={float(params.get('KF_INIT_VEL_VAR_NEG', 0.0)):.4f}"
    )
    print(
        f"|x| <= eps 组: "
        f"n={int(params.get('KF_PRIOR_STAT_N_CENTER', 0))}, "
        f"KF_INIT_VY_CENTER={float(params.get('KF_INIT_VY_CENTER', 0.0)):.4f}, "
        f"KF_INIT_VEL_VAR_CENTER={float(params.get('KF_INIT_VEL_VAR_CENTER', 0.0)):.4f}"
    )


def main():
    """
    主流程:
    1. 读取数据
    2. 找到有效帧
    3. 对所有有效帧做聚类
    4. 用当前数据直接生成概率表 P(ROI | u)
    5. 用这张概率表做全局纵轴版概率加权量测
    6. 统计误差、保存结果、可视化
    """
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

        "GENERATE_PROBABILITY_TABLE_IN_MAIN": True,
        "USE_SAME_DATA_FOR_PROB_AND_EVAL": True,
        "PROB_FRAME_RATIO": 0.5,
        "U_BIN_COUNT": 20,

        "SAVE_PROBABILITY_CSV": True,
        "PROBABILITY_CSV_PATH": "cluster_roi_probability_by_u_global_y.csv",
        "SAVE_PROBABILITY_POINT_CSV": False,
        "PROBABILITY_POINT_CSV_PATH": "cluster_roi_core_points_global_y.csv",

        "SENSOR_Y": 0.0,

        # ========== 概率加权后的时序处理模式 ==========
        #   raw    : 当前帧概率加权量测直接输出
        #   kalman : 对概率加权量测做 Kalman
        #   cv_fit : 对概率加权量测做滑窗匀速拟合
        "ESTIMATION_MODE": "kalman",

        # Kalman 参数
        "KF_DT": 1.0,
        "KF_Q_POS": 0.01,
        "KF_Q_VEL": 0.01,
        "KF_R_POS": 1.50,
        "KF_INIT_POS_VAR": 4.0,

        # 全局默认回退值；真正运行时会由 GT 统计覆盖
        "KF_INIT_VEL_VAR": 9.0,
        "KF_INIT_SPEED_ABS_FALLBACK": 8.0,

        # x 分组阈值
        "KF_INIT_X_SIGN_EPS": 1.0,
        "KF_INIT_VX": 0.0,

        # 连续 miss 的最大保留帧数
        "KF_MAX_MISSES": 10,

        # cv_fit 参数
        "CVFIT_WINDOW_SIZE": 5,
        "CVFIT_MIN_POINTS": 2,
        "CVFIT_HISTORY_MAXLEN": 20,

        # ========== 点级量测生成方式 ==========
        "WEIGHTED_MEASUREMENT_MODE": "weighted_median",
        "WEIGHT_KEEP_THRESHOLD": 0.4,
        "WEIGHT_MIN_SUM": 1e-6,
        "MIN_LOOKUP_WEIGHT": 0.0,

        "SAVE_RESULT_CSV": True,
        "RESULT_CSV_PATH": "global_y_prob_weighted_cluster_results.csv",

        # 最近邻数据关联参数
        "ASSOC_REFERENCE": "gt_center",
        "ASSOC_CLUSTER_CENTER_MODE": "median",
        "ASSOC_GATE_X": 2.0,
        "ASSOC_GATE_Y": 4.0,
        "ASSOC_DIST_WEIGHT_X": 1.0,
        "ASSOC_DIST_WEIGHT_Y": 2.0,

        # 中心几何补偿参数
        "ENABLE_CENTER_COMPENSATION": True,
        "CENTER_COMP_MODE": "constant",
        "CENTER_COMP_DISTANCE": 1.56 + 0.45,
        "CENTER_COMP_ALPHA": 1.0,
        "CENTER_COMP_BY_MODEL": {
            0: 1.75,
            1: 1.45,
            2: 1.25,
        },

        # 可视化
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

    # 用训练帧 GT 自动估计速度先验
    params = estimate_directional_speed_priors_from_gt(
        gt_df=gt_df,
        frame_ids=prob_frame_ids,
        params=params,
    )

    if not bool(params.get("GENERATE_PROBABILITY_TABLE_IN_MAIN", True)):
        raise ValueError("当前版本设计为主函数内生成概率表，请将 GENERATE_PROBABILITY_TABLE_IN_MAIN 设为 True。")

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
        prob_path = params.get("PROBABILITY_CSV_PATH", "cluster_roi_probability_by_u_global_y.csv")
        raw_prob_df.to_csv(prob_path, index=False, encoding="utf-8-sig")
        print(f"\n概率表已保存到: {prob_path}")

    if params.get("SAVE_PROBABILITY_POINT_CSV", False):
        point_path = params.get("PROBABILITY_POINT_CSV_PATH", "cluster_roi_core_points_global_y.csv")
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

    # 把速度先验统计放到下面输出
    print_directional_speed_priors(params)
    print_summary(result_df, params)

    if params.get("SAVE_RESULT_CSV", False):
        csv_path = params.get("RESULT_CSV_PATH", "global_y_prob_weighted_cluster_results.csv")
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n逐目标结果已保存到: {csv_path}")

    if params.get("ENABLE_VIEWER", False):
        launch_viewer(eval_frame_ids, frame_cache, params)


if __name__ == "__main__":
    main()
