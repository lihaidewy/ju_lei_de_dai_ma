import numpy as np


def compute_center_mean(cpts: np.ndarray, snr=None) -> np.ndarray:
    cpts = np.asarray(cpts, dtype=float)
    return np.mean(cpts, axis=0)


def compute_center_median(cpts: np.ndarray, snr=None) -> np.ndarray:
    cpts = np.asarray(cpts, dtype=float)
    return np.median(cpts, axis=0)


def compute_center_snr_mean(cpts: np.ndarray, snr: np.ndarray) -> np.ndarray:
    cpts = np.asarray(cpts, dtype=float)
    if snr is None:
        raise ValueError("snr_mean requires snr input")
    w = np.sqrt(np.maximum(np.asarray(snr, dtype=float), 1e-6))
    w = w / np.sum(w)
    return np.sum(cpts * w[:, None], axis=0)


def compute_center_trimmed_mean(cpts: np.ndarray, trim_ratio: float = 0.10, snr=None) -> np.ndarray:
    cpts = np.asarray(cpts, dtype=float)

    if cpts.shape[0] <= 2:
        return np.mean(cpts, axis=0)

    trim_ratio = float(trim_ratio)
    trim_ratio = max(0.0, min(trim_ratio, 0.45))

    n = cpts.shape[0]
    k = int(np.floor(n * trim_ratio))
    if 2 * k >= n:
        return np.mean(cpts, axis=0)

    x_sorted = np.sort(cpts[:, 0])
    y_sorted = np.sort(cpts[:, 1])

    x_keep = x_sorted[k:n - k]
    y_keep = y_sorted[k:n - k]

    return np.array([np.mean(x_keep), np.mean(y_keep)], dtype=float)


def compute_center_mean_x_median_y(cpts: np.ndarray, snr=None) -> np.ndarray:
    cpts = np.asarray(cpts, dtype=float)
    return np.array([
        np.mean(cpts[:, 0]),
        np.median(cpts[:, 1]),
    ], dtype=float)


def compute_center_velocity_filtered_mean(cpts: np.ndarray, v_local: np.ndarray, cfg) -> np.ndarray:
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


def compute_center_velocity_filtered_trimmed_mean(cpts: np.ndarray, v_local: np.ndarray, cfg) -> np.ndarray:
    cpts = np.asarray(cpts, dtype=float)

    if v_local is None:
        return compute_center_trimmed_mean(cpts, trim_ratio=cfg.TRIMMED_MEAN_RATIO)

    v_local = np.asarray(v_local, dtype=float)
    if v_local.shape[0] != cpts.shape[0]:
        return compute_center_trimmed_mean(cpts, trim_ratio=cfg.TRIMMED_MEAN_RATIO)

    v_ref = float(np.median(v_local))
    dv = np.abs(v_local - v_ref)
    mask_good = dv <= float(cfg.VELOCITY_FILTER_THR)

    if np.sum(mask_good) >= int(cfg.VELOCITY_FILTER_MIN_POINTS):
        return compute_center_trimmed_mean(cpts[mask_good], trim_ratio=cfg.TRIMMED_MEAN_RATIO)

    return compute_center_trimmed_mean(cpts, trim_ratio=cfg.TRIMMED_MEAN_RATIO)


def get_center_function(mode):
    mode = str(mode).strip().lower()

    if mode == "mean":
        return compute_center_mean
    elif mode == "median":
        return compute_center_median
    elif mode == "snr_mean":
        return compute_center_mean
    elif mode == "trimmed_mean":
        return compute_center_mean
    elif mode == "mean_x_median_y":
        return compute_center_mean_x_median_y
    elif mode == "velocity_mean":
        return compute_center_mean
    elif mode == "velocity_trimmed_mean":
        return compute_center_mean
    elif mode == "fixed_box":
        return compute_center_mean
    elif mode == "bottom_half_length":
        return compute_center_mean
    else:
        raise ValueError("Unsupported center mode: %s" % mode)



def compute_center_with_optional_velocity_filter(cpts: np.ndarray, frame_item, mask, center_fn, cfg):
    cpts = np.asarray(cpts, dtype=float)
    mode = (cfg.CLUSTER_CENTER_MODE or "mean").lower().strip()

    v_local = None
    if frame_item is not None and "V" in frame_item:
        v_local = np.asarray(frame_item["V"][mask], dtype=float)

    snr_local = None
    if frame_item is not None and "SNR" in frame_item:
        snr_local = np.asarray(frame_item["SNR"][mask], dtype=float)

    if mode == "velocity_mean":
        return compute_center_velocity_filtered_mean(cpts, v_local, cfg)

    if mode == "velocity_trimmed_mean":
        return compute_center_velocity_filtered_trimmed_mean(cpts, v_local, cfg)

    if mode == "trimmed_mean":
        return compute_center_trimmed_mean(cpts, trim_ratio=cfg.TRIMMED_MEAN_RATIO)

    if mode == "mean_x_median_y":
        return compute_center_mean_x_median_y(cpts)

    if cfg.USE_VELOCITY_FILTER and mode == "mean":
        return compute_center_velocity_filtered_mean(cpts, v_local, cfg)

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


def get_bias_function(mode: str):
    mode = (mode or "two_segment").lower().strip()

    if mode in ("none", "no_bias"):
        return apply_no_bias
    if mode in ("two_segment", "two_segment_bias"):
        return apply_two_segment_bias

    raise ValueError("Unsupported bias mode: %s" % mode)


def get_fixed_box_prior_by_model(model_id: int, cfg):
    """
    根据模型 id 取单个尺寸先验。
    后续如果接视觉先验，可以在 data_pipeline 里优先从视觉结果拿尺寸，
    没有视觉时再退回这里的 model prior。
    """
    model_id = int(model_id)
    priors = getattr(cfg, "GT_MODEL_PRIORS", {})
    item = priors.get(model_id)
    if item is None:
        return None
    return float(item["L"]), float(item["W"])


def get_fixed_box_prior_candidates(cfg, model_id=None):
    """
    返回 fixed-box 可用的尺寸候选。
    - 如果给了 model_id，优先返回该模型对应的单个先验
    - 否则返回全局候选列表
    """
    if model_id is not None:
        one = get_fixed_box_prior_by_model(model_id, cfg)
        if one is not None:
            return [one]

    priors = getattr(cfg, "FIXED_BOX_PRIORS", None)
    if priors:
        return [(float(l), float(w)) for l, w in priors]

    model_priors = getattr(cfg, "GT_MODEL_PRIORS", {})
    out = []
    for mid in sorted(model_priors.keys()):
        item = model_priors[mid]
        out.append((float(item["L"]), float(item["W"])))
    return out
