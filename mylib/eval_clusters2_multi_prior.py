import numpy as np

# =========================
# Target-level evaluation for clustering vs GT vehicles
# Noise label is -1 (clusters are >= 1)
# =========================

# GT dimensions: model = ID % 3
GT_DIM = {
    0: {"L": 5.06, "W": 2.22},
    1: {"L": 4.32, "W": 2.19},
    2: {"L": 3.55, "W": 2.58},
}


# -------------------------
# Fixed-size box center fitting (yaw fixed, default yaw=0)
# Used to make evaluation consistent with visualization when use_fixed_box=True.
# Multi-prior: try 3 (L,W) priors and choose best.
# -------------------------

def _is_inside_rect(u: float, v: float, L: float, W: float) -> bool:
    a = L * 0.5
    b = W * 0.5
    return (abs(u) <= a) and (abs(v) <= b)


def _point_to_rect_center_dist(u: float, v: float, L: float, W: float) -> float:
    """Normalized distance of a point (u,v) to rectangle center (0,0).
    Encourages points to be close to center rather than boundary.

    Returns sqrt((u/(L/2))^2 + (v/(W/2))^2).
    """
    a = max(L * 0.5, 1e-6)
    b = max(W * 0.5, 1e-6)
    return float(((u / a) ** 2 + (v / b) ** 2) ** 0.5)


def _huber(r: float, delta: float = 0.5) -> float:
    r = abs(float(r))
    if r <= delta:
        return 0.5 * r * r
    return delta * (r - 0.5 * delta)


def fit_center_fixed_yaw(
    points_xy: np.ndarray,
    L: float,
    W: float,
    yaw: float = 0.0,
    steps: int = 50,
    step_size: float = 0.5,
    huber_delta: float = 0.5,
):
    """Fit rectangle center (cx, cy) with fixed L/W and fixed yaw using coordinate descent."""
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return (np.nan, np.nan), np.inf

    c = np.cos(yaw)
    s = np.sin(yaw)
    # world -> box frame
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


def fixed_box_xyxy_from_center(cx: float, cy: float, L: float, W: float) -> np.ndarray:
    """Axis-aligned box [xmin,ymin,xmax,ymax] from center and L/W (yaw assumed 0 for IoU)."""
    return np.array([cx - L / 2, cy - W / 2, cx + L / 2, cy + W / 2], dtype=float)


def choose_best_fixed_box_prior(
    points_xy: np.ndarray,
    priors: list,
    yaw: float = 0.0,
    steps: int = 50,
    step_size: float = 0.5,
    huber_delta: float = 0.5,
    score_lambda: float = 1.0,
):
    """Try multiple (L,W) priors and choose the best prior for this cluster.

    Score = (loss / (L+W)) + score_lambda * (1 - inside_ratio)

    Notes:
      - loss comes from center-distance objective (points closer to center is better)
      - inside_ratio keeps the box reasonable (prevents selecting obviously wrong size)
    """
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return {
            "prior_id": -1,
            "L": np.nan,
            "W": np.nan,
            "center": (np.nan, np.nan),
            "box": np.array([np.nan, np.nan, np.nan, np.nan], dtype=float),
            "loss": np.inf,
            "inside_ratio": 0.0,
            "score": np.inf,
        }

    c = np.cos(yaw)
    s = np.sin(yaw)
    R_T = np.array([[c, s],
                    [-s, c]], dtype=float)

    best = None
    for k, (L, W) in enumerate(priors):
        (cx, cy), loss = fit_center_fixed_yaw(
            pts, L=L, W=W, yaw=yaw,
            steps=steps, step_size=step_size, huber_delta=huber_delta
        )

        q = (pts - np.array([cx, cy], dtype=float)) @ R_T.T
        a = L * 0.5
        b = W * 0.5
        inside = (np.abs(q[:, 0]) <= a) & (np.abs(q[:, 1]) <= b)
        inside_ratio = float(np.mean(inside)) if inside.size > 0 else 0.0

        # size-fair scoring: normalize loss by (L+W)
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


# -------------------------
# Boxes & IoU (AABB)
# -------------------------

def gt_box_xyxy(gt_center: np.ndarray, model: int) -> np.ndarray:
    L = float(GT_DIM[int(model)]["L"])
    W = float(GT_DIM[int(model)]["W"])
    cx, cy = float(gt_center[0]), float(gt_center[1])
    return np.array([cx - L / 2, cy - W / 2, cx + L / 2, cy + W / 2], dtype=float)


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
    fixed_box_priors=None,              # list[(L,W)] or None -> use 3 GT priors
    fixed_box_yaw: float = 0.0,
    fixed_box_steps: int = 50,
    fixed_box_step_size: float = 0.5,
    fixed_box_huber_delta: float = 0.5,
    fixed_box_score_lambda: float = 1.0,
):
    """
    Evaluate clustering result against GT per frame at target level.
    Returns metrics dict with:
      TP/FP/FN, precision/recall/f1, mean_center_error,
      matches, unmatched_clusters, unmatched_gts, center_errors,
      model_counts (per GT model: TP/FP/FN)

    When use_fixed_box=True:
      - For each cluster, try multiple priors (L,W), pick best,
        and use that box/center for both matching and IoU.
    """

    pts = np.asarray(pts_xy, float)
    labels = np.asarray(labels)

    # default priors: 3 models
    if fixed_box_priors is None:
        fixed_box_priors = [
            (GT_DIM[0]["L"], GT_DIM[0]["W"]),
            (GT_DIM[1]["L"], GT_DIM[1]["W"]),
            (GT_DIM[2]["L"], GT_DIM[2]["W"]),
        ]

    # model counts init
    model_counts = {0: {"TP": 0, "FP": 0, "FN": 0},
                    1: {"TP": 0, "FP": 0, "FN": 0},
                    2: {"TP": 0, "FP": 0, "FN": 0}}

    # Build cluster list
    clusters = []
    # IMPORTANT: This assumes noise <= 0 and clusters >= 1 (consistent with your pipeline)
    for cid in np.unique(labels):
        if cid < 1:
            continue
        mask = (labels == cid)
        cpts = pts[mask]
        if cpts.size == 0:
            continue

        if use_fixed_box:
            best = choose_best_fixed_box_prior(
                cpts,
                priors=fixed_box_priors,
                yaw=fixed_box_yaw,
                steps=fixed_box_steps,
                step_size=fixed_box_step_size,
                huber_delta=fixed_box_huber_delta,
                score_lambda=fixed_box_score_lambda,
            )
            center = np.array(best["center"], dtype=float)
            box = best["box"]
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

    # Edge cases
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
        # all clusters are FP
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
        # all GT are FN
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
    # GATED matching:
    # only allow edges that satisfy thresholds; otherwise set huge cost.
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
        # greedy ONLY on feasible edges
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

    # Accept matches: keep only feasible (cost not huge)
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

    # per-model TP/FN (based on GT matched or not)
    used_gid = set([m["gid"] for m in matches])
    for gt in gts:
        mm = int(gt["model"])
        if int(gt["gid"]) in used_gid:
            model_counts[mm]["TP"] += 1
        else:
            model_counts[mm]["FN"] += 1

    # per-model FP assignment (by nearest GT center, within fp_assign_dist)
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
