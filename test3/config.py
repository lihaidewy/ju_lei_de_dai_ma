from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from eval_clusters2_multi_prior_v2 import GT_DIM


@dataclass
class Config:
    RADAR_PATH: str = "data/radar.csv"
    GT_PATH: str = "data/reference3.csv"

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

    CLUSTER_CENTER_MODE: str = "mean"

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
    TRACK_ASSOC_DIST_THR: float = 10.0
    TRACK_MAX_MISSES: int = 5

    KF_DT: float = 1.0
    KF_Q_POS: float = 0.50
    KF_Q_VEL: float = 0.50
    KF_R_POS: float = 2.0

    ENABLE_TEMPORAL_DEBUG: bool = True
