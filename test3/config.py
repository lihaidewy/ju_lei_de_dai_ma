from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from eval_clusters2_multi_prior_v2 import GT_DIM


@dataclass
class Config:
    RADAR_PATH: str = "data\\radar.csv"
    GT_PATH: str = "data\\reference3.csv"

    EPS_X: float = 1.5
    EPS_Y: float = 4.0
    EPS_V: float = 1.5
    MIN_PTS: int = 2

    DIST_THR: float = 6.0
    IOU_THR: float = 0.0

    MAX_FRAMES_TO_VIEW: int = 1200
    FRAMES_TO_SHOW: Optional[List[int]] = None

    FIXED_BOX_PRIORS: list = field(default_factory=lambda: [(GT_DIM[m]["L"], GT_DIM[m]["W"]) for m in sorted(GT_DIM.keys())])
    FIXED_BOX_YAW: float = 0.0
    FIXED_BOX_SCORE_LAMBDA: float = 1.0
    FIXED_BOX_INSIDE_MARGIN: float = 0.2
    FIXED_BOX_ALPHA_OUT: float = 10.0
    FIXED_BOX_BETA_IN: float = 2.0

    CLUSTER_CENTER_MODE: str = "mean" # mean0.909, median0.953, snr_mean0.912

    USE_VELOCITY_FILTER: bool = False
    VELOCITY_FILTER_THR: float = 1.2
    VELOCITY_FILTER_MIN_POINTS: int = 2

    CENTER_BIAS_X: float = 0.0
    USE_RANGE_BIAS_Y: bool = False
    BIAS_SPLIT_Y: float = 100.0
    BIAS_Y_NEAR: float = 1.149
    BIAS_Y_FAR: float = 1.586

    EXPORT_CSV_PATH: str = "data/radar_points_with_labels.csv"
    EXPORT_XLSX_PATH: str = "data/radar_points_with_labels.xlsx"
    TP_MATCH_CSV_PATH: str = "data/tp_matches_for_bias.csv"
    TP_MATCH_XLSX_PATH: str = "data/tp_matches_for_bias.xlsx"

    USE_ONLINE_TRACKER: bool = True
    # ------------------------------------------------------------------
    # Tracker switch
    #   "cv"         : 原来的匀速 Kalman + Euclidean association
    #   "ca"         : 更强的恒加速度 Kalman + Mahalanobis association
    #   "cv_robust"  : 匀速模型，但使用 Mahalanobis + 自适应测量噪声
    # ------------------------------------------------------------------
    TRACKER_METHOD = "ca"

    # Association
    TRACK_ASSOC_METRIC = "mahalanobis"   # euclidean / mahalanobis
    TRACK_ASSOC_DIST_THR = 4.0           # 欧氏距离阈值(米)
    TRACK_ASSOC_MAHAL_THR = 3.5          # 马氏距离阈值(sqrt(chi2))

    TRACK_MAX_MISSES = 20

    # Motion model
    KF_DT = 1.0

    # CV model
    KF_Q_POS = 0.50
    KF_Q_VEL = 0.50
    KF_R_POS = 2.0

    # CA model / robust options
    KF_Q_ACC = 0.20
    KF_INIT_POS_VAR = 4.0
    KF_INIT_VEL_VAR = 9.0
    KF_INIT_ACC_VAR = 16.0
    KF_USE_ADAPTIVE_R = True
    KF_ADAPTIVE_R_GAIN = 0.25
    KF_MIN_R_SCALE = 0.75
    KF_MAX_R_SCALE = 4.0

    # Optional output smoother:
    # 在滤波结果上再做一层轻量 EMA，通常能继续压一点抖动
    TRACK_ENABLE_OUTPUT_EMA = False
    TRACK_OUTPUT_EMA_ALPHA = 0.65

    ENABLE_TEMPORAL_DEBUG = True

