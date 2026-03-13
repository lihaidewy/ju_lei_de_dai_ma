import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def rear_prob_and_center_for_cluster(X, Y, V, SNR,
                                    q0=0.05,
                                    sigma_s=1.0,
                                    sigma_v=0.7,
                                    snr0=5.0, snr_scale=3.0, snr_gamma=0.2,
                                    tau=1.0):
    pts = np.column_stack([X, Y])

    # --- 1) SNR soft-weighted center (avoid hard threshold, your SNR has negatives) ---
    w = sigmoid((SNR - snr0) / snr_scale) + 1e-3
    c = (pts * w[:, None]).sum(axis=0) / w.sum()

    norm = np.linalg.norm(c)
    u = c / norm if norm > 1e-6 else np.array([1.0, 0.0])

    # --- 2) LOS projection and boundary ---
    s = pts @ u
    s0 = np.quantile(s, q0)
    d = np.maximum(0.0, s - s0)  # only penalize farther-than-boundary

    P_geo = np.exp(-0.5 * (d / sigma_s) ** 2)

    # --- 3) velocity consistency (optional but cheap) ---
    v_c = np.median(V)
    P_v = np.exp(-0.5 * ((V - v_c) / sigma_v) ** 2)

    # --- 4) SNR reliability (soft) ---
    P_snr = sigmoid((SNR - snr0) / snr_scale)

    P_rear = P_geo * P_v * (P_snr ** snr_gamma)

    # --- 5) rear-center extraction (support line + near-boundary points) ---
    mask = s <= (s0 + tau)
    if mask.sum() < 2:
        idx = np.argsort(-P_rear)[:min(2, len(P_rear))]
        mask = np.zeros(len(P_rear), dtype=bool)
        mask[idx] = True

    t = np.array([-u[1], u[0]])
    q = pts @ t
    q_cand = q[mask]

    if len(q_cand) == 0:
        rear_center = pts[np.argmax(P_rear)]
        width = np.nan
    else:
        if len(q_cand) >= 5:
            qL, qR = np.quantile(q_cand, [0.2, 0.8])
        else:
            qL, qR = q_cand.min(), q_cand.max()

        rear_center = u * s0 + t * ((qL + qR) / 2.0)
        width = (qR - qL)

    return P_rear, rear_center, {"cluster_center": c, "u": u, "s0": s0, "v_median": v_c, "width": width}
