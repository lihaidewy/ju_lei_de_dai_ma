from eval_clusters2_multi_prior_v2 import GT_DIM


class Config:
    RADAR_PATH = "data\\radar.csv"
    GT_PATH = "data\\reference3.csv"

    # DBSCAN
    EPS_X = 1.5
    EPS_Y = 4.0
    EPS_V = 1.5
    MIN_PTS = 2

    # Eval
    DIST_THR = 6.0
    IOU_THR = 0.0

    # Frames
    MAX_FRAMES_TO_VIEW = 1200
    FRAMES_TO_SHOW = None

    # Fixed-box settings
    FIXED_BOX_PRIORS = [(GT_DIM[m]["L"], GT_DIM[m]["W"]) for m in sorted(GT_DIM.keys())]
    FIXED_BOX_YAW = 0.0
    FIXED_BOX_SCORE_LAMBDA = 1.0
    FIXED_BOX_INSIDE_MARGIN = 0.2
    FIXED_BOX_ALPHA_OUT = 10.0
    FIXED_BOX_BETA_IN = 2.0

    # Center strategy
    CLUSTER_CENTER_MODE = "mean"   # mean / median / snr_mean / fixed_box

    # Doppler velocity refinement
    USE_VELOCITY_FILTER = False
    VELOCITY_FILTER_THR = 1.2
    VELOCITY_FILTER_MIN_POINTS = 2

    # Bias correction
    CENTER_BIAS_X = 0.0
    USE_RANGE_BIAS_Y = False
    BIAS_SPLIT_Y = 100.0
    BIAS_Y_NEAR = 1.149
    BIAS_Y_FAR = 1.586

    # Export
    EXPORT_CSV_PATH = "data/radar_points_with_labels.csv"
    EXPORT_XLSX_PATH = "data/radar_points_with_labels.xlsx"

    # Online tracking
    USE_ONLINE_TRACKER = True
    TRACK_ASSOC_DIST_THR = 10.0
    TRACK_MAX_MISSES = 5

    # Kalman
    KF_DT = 1.0
    KF_Q_POS = 0.50
    KF_Q_VEL = 0.50
    KF_R_POS = 2.0

    # Debug
    ENABLE_TEMPORAL_DEBUG = True
