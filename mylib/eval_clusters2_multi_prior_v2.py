import numpy as np

# =========================
# Target-level evaluation for clustering vs GT vehicles
# Noise label is -1 (clusters are >= 1)
# =========================

GT_DIM = {
    0: {"L": 5.06, "W": 2.22},
    1: {"L": 4.32, "W": 2.19},
    2: {"L": 3.55, "W": 2.58},
}

# -------------------------
# Helpers
# -------------------------

def _huber(r: float, delta: float = 0.5) -> float:
    r = abs(float(r))
    if r <= delta:
        return 0.5 * r * r
    return delta * (r - 0.5 * delta)

def fixed_box_xyxy_from_center(cx: float, cy: float, L: float, W: float) -> np.ndarray:
    return np.array([cx - L / 2, cy - W / 2, cx + L / 2, cy + W / 2], dtype=float)

# =============================================================================
# Mode A) center-loss
# =============================================================================

def _point_to_rect_center_dist(u: float, v: float, L: float, W: float) -> float:
    a = max(L * 0.5, 1e-6)
    b = max(W * 0.5, 1e-6)
    return float(((u / a) ** 2 + (v / b) ** 2) ** 0.5)

def fit_center_fixed_yaw_center_loss(
    points_xy: np.ndarray,
    L: float,
    W: float,
    yaw: float = 0.0,
    steps: int = 50,
    step_size: float = 0.5,
    huber_delta: float = 0.5,
):
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return (np.nan, np.nan), np.inf

    c = np.cos(yaw)
    s = np.sin(yaw)
    R_T = np.array([[c, s],
                    [-s, c]], dtype=float)

    center = pts.mean(axis=0).astype(float)

    def loss_at(cxy: np.ndarray) -> float:
        q = (pts - cxy) @ R_T.T
        tot = 0.0
        for u, v in q:
            d = _point_to_rect_center_dist(u, v, L, W)
            tot += _huber(d, delta=huber_delta)
        return tot / max(1, q.shape[0])

    best = loss_at(center)
    step = float(step_size)

    for _ in range(int(steps)):
        improved = False
        for dx, dy in [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]:
            c2 = center + np.array([dx, dy], dtype=float)
            v2 = loss_at(c2)
            if v2 < best:
                center, best = c2, v2
                improved = True
        if not improved:
            step *= 0.7
            if step < 0.05:
                break

    return (float(center[0]), float(center[1])), float(best)

def choose_best_fixed_box_prior_center_loss(
    points_xy: np.ndarray,
    priors: list,
    yaw: float = 0.0,
    steps: int = 50,
    step_size: float = 0.5,
    huber_delta: float = 0.5,
    score_lambda: float = 1.0,
):
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return None

    c = np.cos(yaw)
    s = np.sin(yaw)
    R_T = np.array([[c, s],
                    [-s, c]], dtype=float)

    best = None
    for k, (L, W) in enumerate(priors):
        (cx, cy), loss = fit_center_fixed_yaw_center_loss(
            pts, L=L, W=W, yaw=yaw,
            steps=steps, step_size=step_size, huber_delta=huber_delta
        )

        q = (pts - np.array([cx, cy], dtype=float)) @ R_T.T
        a = L * 0.5
        b = W * 0.5
        inside = (np.abs(q[:, 0]) <= a) & (np.abs(q[:, 1]) <= b)
        inside_ratio = float(np.mean(inside)) if inside.size > 0 else 0.0

        denom = max(float(L + W), 1e-6)
        score = float(loss / denom + score_lambda * (1.0 - inside_ratio))

        cand = {
            "prior_id": int(k),
            "L": float(L),
            "W": float(W),
            "center": (float(cx), float(cy)),
            "box": fixed_box_xyxy_from_center(cx, cy, L, W),
            "loss": float(loss),
            "inside_ratio": float(inside_ratio),
            "score": float(score),
        }

        if best is None or cand["score"] < best["score"]:
            best = cand

    return best

# =============================================================================
# Mode B) edge-loss
# =============================================================================

def _dist_outside_rect(u: float, v: float, L: float, W: float) -> float:
    a = max(L * 0.5, 1e-6)
    b = max(W * 0.5, 1e-6)
    du = max(0.0, abs(u) - a)
    dv = max(0.0, abs(v) - b)
    return float((du * du + dv * dv) ** 0.5)

def _dist_to_nearest_edge_inside(u: float, v: float, L: float, W: float) -> float:
    a = max(L * 0.5, 1e-6)
    b = max(W * 0.5, 1e-6)
    return float(min(a - abs(u), b - abs(v)))

def fit_center_fixed_yaw_edge_loss(
    points_xy: np.ndarray,
    L: float,
    W: float,
    yaw: float = 0.0,
    steps: int = 60,
    step_size: float = 0.5,
    huber_delta: float = 0.5,
    inside_margin: float = 0.5,
    alpha_out: float = 10.0,
    beta_in: float = 1.0,
):
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return (np.nan, np.nan), np.inf

    c = np.cos(yaw)
    s = np.sin(yaw)
    R_T = np.array([[c, s], [-s, c]], dtype=float)

    center = pts.mean(axis=0).astype(float)

    def loss_at(cxy: np.ndarray) -> float:
        q = (pts - cxy) @ R_T.T
        out_list = []
        in_list = []
        for u, v in q:
            d_out = _dist_outside_rect(u, v, L, W)
            out_list.append(_huber(d_out, delta=huber_delta))

            if d_out <= 1e-9:
                d_in = _dist_to_nearest_edge_inside(u, v, L, W)
                in_list.append(_huber(max(0.0, d_in - inside_margin), delta=huber_delta))

        out_mean = float(np.mean(out_list)) if len(out_list) else 0.0
        in_mean = float(np.mean(in_list)) if len(in_list) else 0.0
        return alpha_out * out_mean + beta_in * in_mean

    best = loss_at(center)
    step = float(step_size)

    for _ in range(int(steps)):
        improved = False
        for dx, dy in [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]:
            c2 = center + np.array([dx, dy], dtype=float)
            v2 = loss_at(c2)
            if v2 < best:
                center, best = c2, v2
                improved = True
        if not improved:
            step *= 0.7
            if step < 0.05:
                break

    return (float(center[0]), float(center[1])), float(best)

def choose_best_fixed_box_prior_edge_loss(
    points_xy: np.ndarray,
    priors: list,
    yaw: float = 0.0,
    steps: int = 60,
    step_size: float = 0.5,
    huber_delta: float = 0.5,
    score_lambda: float = 1.0,
    inside_margin: float = 0.5,
    alpha_out: float = 10.0,
    beta_in: float = 1.0,
):
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return None

    c = np.cos(yaw)
    s = np.sin(yaw)
    R_T = np.array([[c, s], [-s, c]], dtype=float)

    best = None
    for k, (L, W) in enumerate(priors):
        (cx, cy), loss = fit_center_fixed_yaw_edge_loss(
            pts, L=L, W=W, yaw=yaw,
            steps=steps, step_size=step_size, huber_delta=huber_delta,
            inside_margin=inside_margin, alpha_out=alpha_out, beta_in=beta_in
        )

        q = (pts - np.array([cx, cy], dtype=float)) @ R_T.T
        a = L * 0.5
        b = W * 0.5
        inside = (np.abs(q[:, 0]) <= a) & (np.abs(q[:, 1]) <= b)
        inside_ratio = float(np.mean(inside)) if inside.size > 0 else 0.0

        denom = max(float(L + W), 1e-6)
        score = float(loss / denom + score_lambda * (1.0 - inside_ratio))

        cand = {
            "prior_id": int(k),
            "L": float(L),
            "W": float(W),
            "center": (float(cx), float(cy)),
            "box": fixed_box_xyxy_from_center(cx, cy, L, W),
            "loss": float(loss),
            "inside_ratio": float(inside_ratio),
            "score": float(score),
        }

        if best is None or cand["score"] < best["score"]:
            best = cand

    return best

def choose_best_fixed_box_prior_mode(
    points_xy: np.ndarray,
    priors: list,
    fit_mode: str = "center",
    yaw: float = 0.0,
    steps: int = 50,
    step_size: float = 0.5,
    huber_delta: float = 0.5,
    score_lambda: float = 1.0,
    inside_margin: float = 0.5,
    alpha_out: float = 10.0,
    beta_in: float = 1.0,
):
    fit_mode = (fit_mode or "center").lower().strip()
    if fit_mode in ("center", "center_loss", "pull_center"):
        return choose_best_fixed_box_prior_center_loss(
            points_xy, priors=priors, yaw=yaw,
            steps=steps, step_size=step_size, huber_delta=huber_delta,
            score_lambda=score_lambda
        )
    if fit_mode in ("edge", "edge_loss", "boundary"):
        return choose_best_fixed_box_prior_edge_loss(
            points_xy, priors=priors, yaw=yaw,
            steps=max(steps, 60), step_size=step_size, huber_delta=huber_delta,
            score_lambda=score_lambda,
            inside_margin=inside_margin, alpha_out=alpha_out, beta_in=beta_in
        )
    raise ValueError(f"Unknown fit_mode={fit_mode}. Use 'center' or 'edge'.")

# -------------------------
# Boxes & IoU (AABB)
# -------------------------

def gt_box_xyxy(gt_center: np.ndarray, model: int) -> np.ndarray:
    L = float(GT_DIM[int(model)]["L"])
    W = float(GT_DIM[int(model)]["W"])
    cx, cy = float(gt_center[0]), float(gt_center[1])
    return fixed_box_xyxy_from_center(cx, cy, L, W)

def aabb_from_points(pts_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_xy, float)
    if pts.size == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=float)
    xmin = float(np.min(pts[:, 0]))
    xmax = float(np.max(pts[:, 0]))
    ymin = float(np.min(pts[:, 1]))
    ymax = float(np.max(pts[:, 1]))
    return np.array([xmin, ymin, xmax, ymax], dtype=float)

def iou_aabb(b1: np.ndarray, b2: np.ndarray) -> float:
    b1 = np.asarray(b1, float)
    b2 = np.asarray(b2, float)
    if np.any(np.isnan(b1)) or np.any(np.isnan(b2)):
        return 0.0
    xA = max(float(b1[0]), float(b2[0]))
    yA = max(float(b1[1]), float(b2[1]))
    xB = min(float(b1[2]), float(b2[2]))
    yB = min(float(b1[3]), float(b2[3]))
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    area1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    area2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    union = area1 + area2 - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)

# -------------------------
# Main evaluation function
# -------------------------

def eval_one_frame_target_level(
    pts_xy: np.ndarray,
    labels: np.ndarray,
    gt_list,
    dist_thr: float = 4.0,
    iou_thr: float = 0.10,
    cost_iou_w: float = 2.0,
    fp_assign_dist: float = 8.0,
    use_fixed_box: bool = False,
    fixed_box_priors=None,
    fixed_box_fit_mode: str = "center",   # NEW
    fixed_box_yaw: float = 0.0,
    fixed_box_steps: int = 50,
    fixed_box_step_size: float = 0.5,
    fixed_box_huber_delta: float = 0.5,
    fixed_box_score_lambda: float = 1.0,
    # edge-loss extras
    fixed_box_inside_margin: float = 0.5,
    fixed_box_alpha_out: float = 10.0,
    fixed_box_beta_in: float = 1.0,
):
    """
    Evaluate clustering result against GT per frame at target level.

    IMPORTANT:
      - The only difference between fit_mode=center/edge is how cluster box/center is produced.
      - The matching / metrics are identical (same cost, same gating, same TP/FP/FN, etc.).
    """

    pts = np.asarray(pts_xy, float)
    labels = np.asarray(labels)

    if fixed_box_priors is None:
        fixed_box_priors = [
            (GT_DIM[0]["L"], GT_DIM[0]["W"]),
            (GT_DIM[1]["L"], GT_DIM[1]["W"]),
            (GT_DIM[2]["L"], GT_DIM[2]["W"]),
        ]

    model_counts = {0: {"TP": 0, "FP": 0, "FN": 0},
                    1: {"TP": 0, "FP": 0, "FN": 0},
                    2: {"TP": 0, "FP": 0, "FN": 0}}

    # Build cluster list
    clusters = []
    for cid in np.unique(labels):
        if cid < 1:
            continue
        mask = (labels == cid)
        cpts = pts[mask]
        if cpts.size == 0:
            continue

        if use_fixed_box:
            best = choose_best_fixed_box_prior_mode(
                cpts,
                priors=fixed_box_priors,
                fit_mode=fixed_box_fit_mode,
                yaw=fixed_box_yaw,
                steps=fixed_box_steps,
                step_size=fixed_box_step_size,
                huber_delta=fixed_box_huber_delta,
                score_lambda=fixed_box_score_lambda,
                inside_margin=fixed_box_inside_margin,
                alpha_out=fixed_box_alpha_out,
                beta_in=fixed_box_beta_in,
            )
            center = np.array(best["center"], dtype=float)
            box = np.asarray(best["box"], dtype=float)
            prior_id = int(best["prior_id"])
        else:
            center = cpts.mean(axis=0)
            box = aabb_from_points(cpts)
            prior_id = -1

        clusters.append({
            "cid": int(cid),
            "center": center,
            "box": box,
            "prior_id": prior_id,
        })

    # Build GT list
    gts = []
    for g in gt_list:
        gid = int(g["id"])
        model = int(g["model"])
        c = np.array([float(g["x"]), float(g["y"])], dtype=float)
        b = gt_box_xyxy(c, model)
        gts.append({"gid": gid, "model": model, "center": c, "box": b})

    nC = len(clusters)
    nG = len(gts)

    if nC == 0 and nG == 0:
        return {
            "TP": 0, "FP": 0, "FN": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "mean_center_error": float("nan"),
            "center_errors": [],
            "matches": [],
            "unmatched_clusters": [],
            "unmatched_gts": [],
            "model_counts": model_counts,
        }

    if nG == 0:
        FP = nC
        P = 0.0 if FP > 0 else 1.0
        R = 1.0
        F1 = 0.0
        return {
            "TP": 0, "FP": FP, "FN": 0,
            "precision": P, "recall": R, "f1": F1,
            "mean_center_error": float("nan"),
            "center_errors": [],
            "matches": [],
            "unmatched_clusters": [c["cid"] for c in clusters],
            "unmatched_gts": [],
            "model_counts": model_counts,
        }

    if nC == 0:
        for gt in gts:
            model_counts[int(gt["model"])]["FN"] += 1
        return {
            "TP": 0, "FP": 0, "FN": nG,
            "precision": 1.0, "recall": 0.0, "f1": 0.0,
            "mean_center_error": float("nan"),
            "center_errors": [],
            "matches": [],
            "unmatched_clusters": [],
            "unmatched_gts": [gt["gid"] for gt in gts],
            "model_counts": model_counts,
        }

    # =========================
    # GATED matching
    # =========================
    BIG = 1e9
    cost = np.full((nC, nG), BIG, dtype=float)
    dist_mat = np.zeros((nC, nG), dtype=float)
    iou_mat = np.zeros((nC, nG), dtype=float)
    feasible = np.zeros((nC, nG), dtype=bool)

    for i, c in enumerate(clusters):
        for j, gt in enumerate(gts):
            d = float(np.linalg.norm(c["center"] - gt["center"]))
            iou = float(iou_aabb(c["box"], gt["box"]))
            dist_mat[i, j] = d
            iou_mat[i, j] = iou

            ok = (d <= float(dist_thr)) and (iou >= float(iou_thr))
            feasible[i, j] = ok
            if ok:
                cost[i, j] = d + float(cost_iou_w) * (1.0 - iou)

    # Hungarian if available else greedy
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
    except Exception:
        flat = []
        for i in range(nC):
            for j in range(nG):
                if feasible[i, j]:
                    flat.append((float(cost[i, j]), i, j))
        flat.sort(key=lambda x: x[0])
        used_i = set()
        used_j = set()
        pairs = []
        for _, i, j in flat:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            pairs.append((i, j))

    matches = []
    used_c = set()
    used_g = set()
    center_errors = []

    for i, j in pairs:
        if cost[i, j] >= BIG * 0.5:
            continue
        d = float(dist_mat[i, j])
        iou = float(iou_mat[i, j])
        cid = clusters[i]["cid"]
        gid = gts[j]["gid"]
        matches.append({
            "cid": int(cid),
            "gid": int(gid),
            "center_dist": float(d),
            "iou": float(iou),
        })
        used_c.add(i)
        used_g.add(j)
        center_errors.append(float(d))

    TP = len(matches)
    FP = nC - TP
    FN = nG - TP

    P = TP / (TP + FP) if (TP + FP) > 0 else 1.0
    R = TP / (TP + FN) if (TP + FN) > 0 else 1.0
    F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    mean_center_error = float(np.mean(center_errors)) if len(center_errors) > 0 else float("nan")

    unmatched_clusters = [clusters[i]["cid"] for i in range(nC) if i not in used_c]
    unmatched_gts = [gts[j]["gid"] for j in range(nG) if j not in used_g]

    used_gid = set([m["gid"] for m in matches])
    for gt in gts:
        mm = int(gt["model"])
        if int(gt["gid"]) in used_gid:
            model_counts[mm]["TP"] += 1
        else:
            model_counts[mm]["FN"] += 1

    for i in range(nC):
        if i in used_c:
            continue
        cc = clusters[i]["center"]
        best_j = None
        best_d = float("inf")
        for j in range(nG):
            d = float(np.linalg.norm(cc - gts[j]["center"]))
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d <= float(fp_assign_dist):
            mm = int(gts[best_j]["model"])
            model_counts[mm]["FP"] += 1

    return {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "precision": float(P),
        "recall": float(R),
        "f1": float(F1),
        "mean_center_error": float(mean_center_error),
        "center_errors": center_errors,
        "matches": matches,
        "unmatched_clusters": unmatched_clusters,
        "unmatched_gts": unmatched_gts,
        "model_counts": model_counts,
    }
