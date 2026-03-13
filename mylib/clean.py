import numpy as np

def weighted_mean(X, w):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    return (X * w[:, None]).sum(axis=0)

def weighted_cov_2d(X, w, mean=None):
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    if mean is None:
        mean = weighted_mean(X, w)
    Xc = X - mean
    return (Xc * w[:, None]).T @ Xc

def weighted_quantile(values, quantiles, sample_weight=None):
    values = np.asarray(values, dtype=float)
    quantiles = np.atleast_1d(quantiles).astype(float)

    if sample_weight is None:
        out = np.quantile(values, quantiles)
        return out if len(out) > 1 else float(out[0])

    w = np.asarray(sample_weight, dtype=float)
    w = np.clip(w, 1e-12, None)

    sorter = np.argsort(values)
    v = values[sorter]
    w = w[sorter]
    cw = np.cumsum(w)
    cw /= cw[-1]
    out = np.interp(quantiles, cw, v)
    return out if len(out) > 1 else float(out[0])

def clean_cluster_by_velocity(idx, v, k=3.0, v_min_band=0.2):
    """簇内速度一致性清洗（稳健中位数+MAD）"""
    vc = np.asarray(v)[idx]
    v0 = np.median(vc)
    mad = np.median(np.abs(vc - v0))
    sigma = 1.4826 * mad
    band = max(v_min_band, k * sigma)
    keep = np.abs(vc - v0) <= band
    return idx[keep], {"v0": float(v0), "mad": float(mad), "band": float(band)}

def estimate_axis_and_endpoints(pts_xy, snr, q_low=0.10, q_high=0.90, min_points=3, length_range_m=(2.5, 7.0)):
    """
    加权PCA + 投影分位端点
    pts_xy: (N,2)
    snr: (N,) 用于权重（log1p）
    返回 dict 或 None
    """
    pts = np.asarray(pts_xy, dtype=float)
    if pts.shape[0] < min_points:
        return None

    snr = np.asarray(snr, dtype=float)
    w = np.log1p(np.clip(snr, 0, None))  # 推荐：避免极大snr支配
    if np.all(w <= 0):
        w = np.ones_like(w)

    c = weighted_mean(pts, w)
    C = weighted_cov_2d(pts, w, mean=c)
    evals, evecs = np.linalg.eigh(C)
    u = evecs[:, np.argmax(evals)]
    u = u / (np.linalg.norm(u) + 1e-12)

    s = (pts - c) @ u
    s_low, s_high = weighted_quantile(s, [q_low, q_high], w)

    # 长度约束（可选，但很推荐）
    length = float(s_high - s_low)
    if length_range_m is not None:
        Lmin, Lmax = length_range_m
        mid = 0.5 * (s_low + s_high)
        if length < Lmin:
            s_low, s_high = mid - 0.5 * Lmin, mid + 0.5 * Lmin
            length = Lmin
        elif length > Lmax:
            s_low, s_high = mid - 0.5 * Lmax, mid + 0.5 * Lmax
            length = Lmax

    rear = c + s_low * u
    front = c + s_high * u

    return {"center": c, "u": u, "front": front, "rear": rear, "length": length}
