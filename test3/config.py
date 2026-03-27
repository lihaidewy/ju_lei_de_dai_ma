from dataclasses import dataclass, field
from typing import Dict, List, Optional


_GT_DIM = {
    0: {"L": 5.06, "W": 2.22},
    1: {"L": 4.32, "W": 2.19},
    2: {"L": 3.55, "W": 2.58},
}


@dataclass
class Config:
    RADAR_PATH: str = "data\\radar.csv"
    GT_PATH: str = "data\\reference3.csv"
    # RADAR_PATH: str = "data\S1m_point\S1m.csv"
    # GT_PATH: str = "data\S1m_point\\reference.csv"
    # RADAR_PATH: str = "data\S1s_point\S1s.csv"
    # GT_PATH: str = "data\S1s_point\\references.csv"

    EPS_X: float = 1.5
    EPS_Y: float = 4.0
    EPS_V: float = 1.5
    MIN_PTS: int = 2

    DIST_THR: float = 6.0
    IOU_THR: float = 0.0

    MAX_FRAMES_TO_VIEW: int = 1200
    FRAMES_TO_SHOW: Optional[List[int]] = None

    # ------------------------------------------------------------
    # 噪声点处理策略
    # ------------------------------------------------------------
    # True:
    #   DBSCAN 输出的 label==-1 不再视为噪声丢弃，
    #   而是把每个噪声点重标成一个独立目标(singleton cluster)。
    # False:
    #   保持原逻辑，label==-1 直接忽略。
    TREAT_NOISE_AS_SINGLETON_TARGETS: bool = True

    # 如果你只想把“单点噪声”当目标，而不是任意噪声簇当目标，
    # 这里保持 True 即可。当前 DBSCAN 的噪声通常就是单点。
    ONLY_PROMOTE_SINGLE_POINT_NOISE: bool = True

    # ------------------------------------------------------------
    # 当前先使用 3 个 GT/车型先验。
    # 后续如果接视觉模型，可以优先用视觉先验，再退回这里。
    # ------------------------------------------------------------
    GT_MODEL_PRIORS: Dict[int, Dict[str, float]] = field(
        default_factory=lambda: dict(_GT_DIM)
    )

    FIXED_BOX_PRIORS: list = field(
        default_factory=lambda: [
            (_GT_DIM[m]["L"], _GT_DIM[m]["W"]) for m in sorted(_GT_DIM.keys())
        ]
    )
    FIXED_BOX_USE_MODEL_PRIOR: bool = True
    FIXED_BOX_FALLBACK_TO_ALL_PRIORS: bool = True
    FIXED_BOX_YAW: float = 0.0
    FIXED_BOX_SCORE_LAMBDA: float = 1.0
    FIXED_BOX_INSIDE_MARGIN: float = 0.2
    FIXED_BOX_ALPHA_OUT: float = 10.0
    FIXED_BOX_BETA_IN: float = 2.0

    # ------------------------------------------------------------
    # Center mode:
    # mean / median / snr_mean / trimmed_mean / mean_x_median_y
    # velocity_mean / velocity_trimmed_mean / fixed_box / bottom_half_length
    # ------------------------------------------------------------
    CLUSTER_CENTER_MODE: str = "bottom_half_length"

    USE_VELOCITY_FILTER: bool = False
    VELOCITY_FILTER_THR: float = 1.2
    VELOCITY_FILTER_MIN_POINTS: int = 2
    TRIMMED_MEAN_RATIO: float = 0.10

    # Bias mode: two_segment / none
    BIAS_MODE: str = "two_segment"

    CENTER_BIAS_X: float = 0.0
    USE_RANGE_BIAS_Y: bool = True
    BIAS_SPLIT_Y: float = 100.0
    BIAS_Y_NEAR: float = 0.434
    BIAS_Y_FAR: float = 1.581

    # ------------------------------------------------------------
    # bottom_half_length 的稳健底点参数
    # - 在 y 最小的前 k 个点中，再取 mean / median
    # - 默认用 median，更稳健
    # ------------------------------------------------------------
    BOTTOM_ROBUST_K: int = 1
    BOTTOM_ROBUST_MIN_MODE: str = "median"   # "median" 或 "mean"

    EXPORT_CSV_PATH: str = "data/S1m_point_with_labels.csv"
    EXPORT_XLSX_PATH: str = "data/S1m_point_with_labels.xlsx"
    TP_MATCH_CSV_PATH: str = "data/S1m_point_for_bias.csv"
    TP_MATCH_XLSX_PATH: str = "data/S1m_point_for_bias.xlsx"

    USE_ONLINE_TRACKER: bool = True
    # ------------------------------------------------------------------
    # Tracker switch
    #   "cv"         : 原来的匀速 Kalman + Euclidean association
    #   "ca"         : 更强的恒加速度 Kalman + Mahalanobis association
    #   "cv_robust"  : 匀速模型，但使用 Mahalanobis + 自适应测量噪声
    # ------------------------------------------------------------------
    TRACKER_METHOD: str = "cv"

    # 绝对匀速策略开关
    TRACK_STRICT_CONSTANT_VELOCITY: bool = True

    # Association
    TRACK_ASSOC_METRIC: str = "euclidean"   # euclidean / mahalanobis
    TRACK_ASSOC_DIST_THR: float = 6.0
    TRACK_ASSOC_MAHAL_THR: float = 3.5

    # ------------------------------------------------------------
    # Velocity-aware association
    # - cluster side uses vr_median
    # - track side keeps a radial-speed EMA
    # - apply velocity gate first, then add soft velocity cost
    # ------------------------------------------------------------
    TRACK_USE_VEL_ASSOC: bool = False
    TRACK_ASSOC_VEL_THR: float = 2.0
    TRACK_ASSOC_W_POS: float = 1.0
    TRACK_ASSOC_W_VEL: float = 0.0
    TRACK_VEL_EMA_ALPHA: float = 0.6

    TRACK_MAX_MISSES: int = 20
    # ------------------------------------------------------------
    # Track confirmation:
    # - 新轨迹先 tentative
    # - 连续命中达到阈值后才 confirmed
    # - tentative 轨迹允许更快删除
    # ------------------------------------------------------------
    TRACK_MIN_HITS_TO_CONFIRM: int = 2
    TRACK_MAX_TENTATIVE_MISSES: int = 1

    # Motion model
    KF_DT: float = 1.0

    # CV model
    KF_Q_POS: float = 0.02
    KF_Q_VEL: float = 0.02
    KF_R_POS: float = 2.0

    # CA model / robust options
    KF_Q_ACC: float = 0.0
    KF_INIT_POS_VAR: float = 2.0
    KF_INIT_VEL_VAR: float = 25.0
    KF_INIT_ACC_VAR: float = 16.0
    KF_USE_ADAPTIVE_R: bool = False
    KF_ADAPTIVE_R_GAIN: float = 0.0
    KF_MIN_R_SCALE: float = 1.0
    KF_MAX_R_SCALE: float = 1.0

    # Optional output smoother:
    # 在滤波结果上再做一层轻量 EMA，通常能继续压一点抖动
    TRACK_ENABLE_OUTPUT_EMA: bool = False
    TRACK_OUTPUT_EMA_ALPHA: float = 0.85

    ENABLE_TEMPORAL_DEBUG: bool = True

    # ------------------------------------------------------------------
    # Viewer / Animation
    # ------------------------------------------------------------------
    VIEWER_ENABLE: bool = True
    VIEWER_MODE: str = "animated"   # "static" / "animated"

    # animation behavior
    ANIM_AUTOPLAY: bool = True
    ANIM_INTERVAL_MS: int = 250
    ANIM_TRAIL_LEN: int = 40
    ANIM_LOOP: bool = True

    # layer switches
    VIEW_SHOW_GT: bool = True
    VIEW_SHOW_GT_BOX: bool = True
    VIEW_SHOW_GT_ID: bool = True

    VIEW_SHOW_MEASUREMENTS_LEFT: bool = True
    VIEW_SHOW_MEASUREMENTS_RIGHT: bool = True

    VIEW_SHOW_CLUSTERS: bool = True
    VIEW_SHOW_CENTERS: bool = True
    VIEW_SHOW_CENTER_TEXT: bool = True
    VIEW_SHOW_TRACKS: bool = True
    VIEW_SHOW_TRACK_ID: bool = True
    VIEW_SHOW_MATCH_TEXT: bool = False
    VIEW_SHOW_NOISE: bool = False

    # display range
    VIEW_XLIM: tuple = (-30, 30)
    VIEW_YLIM: tuple = (0, 250)
    VIEW_XTICK_STEP: float = 5.0
    VIEW_YTICK_STEP: float = 5.0

    # ------------------------------------------------------------------
    # Animation export
    # ------------------------------------------------------------------
    EXPORT_ANIMATION: bool = False
    EXPORT_ANIMATION_FORMAT: str = "gif"   # "gif" / "mp4"
    EXPORT_ANIMATION_PATH: str = "data/result_fixed_box_bias_ema.gif"

    EXPORT_ANIMATION_FPS: int = 5
    EXPORT_ANIMATION_DPI: int = 120

    # True: 导出所有 frame_ids
    # False: 只导出当前 viewer 中的序列
    EXPORT_ANIMATION_ALL_FRAMES: bool = True

    # ------------------------------------------------------------------
    # Trajectory diagnostics
    # ------------------------------------------------------------------
    ENABLE_TRACK_DIAGNOSTICS: bool = True
    TRACK_DIAG_FRAME_CSV_PATH: str = "data/track_frame_records.csv"
    TRACK_DIAG_SUMMARY_CSV_PATH: str = "data/track_summary.csv"
    TRACK_DIAG_GLOBAL_CSV_PATH: str = "data/track_diag_global_summary.csv"
