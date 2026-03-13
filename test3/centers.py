import numpy as np


def compute_center_mean(cpts: np.ndarray, snr=None) -> np.ndarray:
    return np.mean(cpts, axis=0)


def compute_center_median(cpts: np.ndarray, snr=None) -> np.ndarray:
    return np.median(cpts, axis=0)


def compute_center_snr_mean(cpts: np.ndarray, snr: np.ndarray) -> np.ndarray:
    if snr is None:
        raise ValueError("snr_mean requires snr input")
    w = np.sqrt(np.maximum(np.asarray(snr, dtype=float), 1e-6))
    w = w / np.sum(w)
    return np.sum(cpts * w[:, None], axis=0)

def compute_center_velocity_filtered_mean(cpts: np.ndarray, v_local: np.ndarray, cfg) -> np.ndarray:
    """
    第一版 Doppler refinement:
    1) 取 cluster 内速度中位数 v_ref
    2) 保留 |v_i - v_ref| <= thr 的点
    3) 对保留下来的点求 mean
    4) 如果保留点太少，则退回普通 mean
    """
    cpts = np.asarray(cpts, dtype=float)

    if v_local is None:
        return np.mean(cpts, axis=0)

    v_local = np.asarray(v_local, dtype=float)
    if v_local.shape[0] != cpts.shape[0]:
        return np.mean(cpts, axis=0)

    v_ref = float(np.median(v_local))
    dv = np.abs(v_local - v_ref)

    mask_good = dv <= float(cfg.VELOCITY_FILTER_THR)

    if np.sum(mask_good) >= int(cfg.VELOCITY_FILTER_MIN_POINTS):
        return np.mean(cpts[mask_good], axis=0)

    return np.mean(cpts, axis=0)

def get_center_function(mode: str):
    mode = (mode or "mean").lower().strip()

    if mode == "mean":
        return compute_center_mean
    if mode == "median":
        return compute_center_median
    if mode == "snr_mean":
        return compute_center_snr_mean

    raise ValueError(f"Unsupported center mode: {mode}")

def compute_center_with_optional_velocity_filter(cpts: np.ndarray, frame_item, mask, center_fn, cfg):
    """
    在原始 center_fn 基础上，可选加入 Doppler velocity filtering。
    当前仅对 mean 模式启用 velocity refinement。
    """
    v_local = None
    if frame_item is not None and "V" in frame_item:
        v_local = np.asarray(frame_item["V"][mask], dtype=float)

    if cfg.USE_VELOCITY_FILTER and cfg.CLUSTER_CENTER_MODE == "mean":
        return compute_center_velocity_filtered_mean(cpts, v_local, cfg)

    snr_local = None
    if frame_item is not None and "SNR" in frame_item:
        snr_local = np.asarray(frame_item["SNR"][mask], dtype=float)

    # snr_mean 需要 snr，其它模式忽略第二参数
    return center_fn(cpts, snr_local)


def apply_no_bias(center: np.ndarray, cfg) -> np.ndarray:
    return np.asarray(center, dtype=float).copy()


def apply_two_segment_bias(center: np.ndarray, cfg) -> np.ndarray:
    out = np.asarray(center, dtype=float).copy()

    bias_y = 0.0
    if cfg.USE_RANGE_BIAS_Y:
        if out[1] < cfg.BIAS_SPLIT_Y:
            bias_y = cfg.BIAS_Y_NEAR
        else:
            bias_y = cfg.BIAS_Y_FAR

    out = out + np.array([cfg.CENTER_BIAS_X, bias_y], dtype=float)
    return out
