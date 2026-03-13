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
#------------------------新增------------------------
def _point_to_rect_boundary_dist(u, v, L, W):
    """
    当前坐标系：
    u -> X（横向）
    v -> Y（前向）

    车辆框定义：
    - X方向宽 W
    - Y方向长 L
    """
    a = W * 0.5   # X方向半宽
    b = L * 0.5   # Y方向半长
    au = abs(u)
    av = abs(v)
    dx = au - a
    dy = av - b
    if dx <= 0 and dy <= 0:
        return min(a - au, b - av)
    return float(np.hypot(max(dx, 0.0), max(dy, 0.0)))


def _huber(r, delta=0.5):
    r = abs(float(r))
    if r <= delta:
        return 0.5 * r * r
    return delta * (r - 0.5 * delta)

def fit_center_fixed_yaw(points_xy, L, W, yaw=0.0, steps=50, step_size=0.5, huber_delta=0.5):
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return (np.nan, np.nan), np.inf

    c = np.cos(yaw); s = np.sin(yaw)
    R_T = np.array([[ c, s],
                    [-s, c]], dtype=float)  # world -> box frame

    center = pts.mean(axis=0).astype(float)

    def loss_at(cxy):
        q = (pts - cxy) @ R_T.T
        tot = 0.0
        for u, v in q:
            d = _point_to_rect_boundary_dist(u, v, L, W)
            tot += _huber(d, delta=huber_delta)
        return tot / max(1, q.shape[0])

    best = loss_at(center)
    step = float(step_size)

    for _ in range(int(steps)):
        improved = False
        for dx, dy in [(step,0), (-step,0), (0,step), (0,-step)]:
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
    """
    当前坐标系：
    X = 横向
    Y = 前向

    返回 [xmin, ymin, xmax, ymax]
    其中：
    - X方向宽 W
    - Y方向长 L
    """
    return np.array([cx - W/2, cy - L/2, cx + W/2, cy + L/2], dtype=float)

#---------------------------------------------------------------------------------------------
def gt_box_xyxy(x: float, y: float, model: int) -> np.ndarray:
    """
    当前坐标系：
    X = 横向
    Y = 前向

    GT车辆框：
    - 宽 W 沿 X
    - 长 L 沿 Y

    返回 [xmin, ymin, xmax, ymax]
    """
    L = GT_DIM[int(model)]["L"]
    W = GT_DIM[int(model)]["W"]
    return np.array([x - W/2, y - L/2, x + W/2, y + L/2], dtype=float)

def aabb_from_points(pts_xy: np.ndarray) -> np.ndarray:
    mn = pts_xy.min(axis=0)
    mx = pts_xy.max(axis=0)
    return np.array([mn[0], mn[1], mx[0], mx[1]], dtype=float)

def iou_aabb(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    a1 = max(0.0, b1[2]-b1[0]) * max(0.0, b1[3]-b1[1])
    a2 = max(0.0, b2[2]-b2[0]) * max(0.0, b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 1e-9 else 0.0

def _hungarian_or_greedy(cost: np.ndarray):
    """
    Returns matched row indices and col indices.
    Uses scipy Hungarian if available, otherwise greedy by smallest cost.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return r, c
    except Exception:
        M, K = cost.shape
        flat = [(cost[i, j], i, j) for i in range(M) for j in range(K)]
        flat.sort(key=lambda x: x[0])
        used_r, used_c = set(), set()
        pairs = []
        for _, i, j in flat:
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j)
            pairs.append((i, j))
        if not pairs:
            return np.array([], dtype=int), np.array([], dtype=int)
        r = np.array([p[0] for p in pairs], dtype=int)
        c = np.array([p[1] for p in pairs], dtype=int)
        return r, c

def eval_one_frame_target_level(
    pts_xy: np.ndarray,
    labels: np.ndarray,
    gt_list,
    dist_thr: float = 4.0,
    iou_thr: float = 0.10,
    cost_iou_w: float = 2.0,
    fp_assign_dist: float = 8.0,
    use_fixed_box: bool = False,
    fixed_box_L: float = 4.5,
    fixed_box_W: float = 2.0,
):
    """
    Target-level evaluation (cluster <-> GT vehicle).

    Inputs:
      pts_xy: (N,2)
      labels: (N,) noise = -1, clusters >= 1
      gt_list: [{'id':int,'x':float,'y':float,'model':int}, ...]

    Matching:
      Hungarian (or greedy fallback) on cost = center_dist + cost_iou_w*(1 - IoU),
      then accept match if center_dist<=dist_thr and IoU>=iou_thr.

    Returns dict:
      TP, FP, FN
      precision, recall, f1
      mean_center_error, center_errors
      matches: [(cid, gid, dist, iou), ...]
      model_counts: {0/1/2: {'TP','FP','FN'}}
    """
    pts_xy = np.asarray(pts_xy, float)
    labels = np.asarray(labels).reshape(-1)

    # ---- clusters (labels >= 1) ----
    cids = [int(c) for c in np.unique(labels) if c >= 1]
    clusters = []
    for cid in cids:
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        cpts = pts_xy[idx]
        if use_fixed_box:
            (cx_fit, cy_fit), _ = fit_center_fixed_yaw(
                cpts, L=fixed_box_L, W=fixed_box_W,
                yaw=0.0, steps=50, step_size=0.5, huber_delta=0.5
            )
            center = np.array([cx_fit, cy_fit], dtype=float)
            box = fixed_box_xyxy_from_center(cx_fit, cy_fit, fixed_box_L, fixed_box_W)
        else:
            center = cpts.mean(axis=0)
            box = aabb_from_points(cpts)
        clusters.append({"cid": cid, "center": center, "box": box})

    # ---- gts ----
    gts = []
    for g in gt_list:
        gid = int(g["id"])
        cx = float(g["x"]); cy = float(g["y"])
        model = int(g["model"])
        gts.append({
            "gid": gid,
            "model": model,
            "center": np.array([cx, cy], dtype=float),
            "box": gt_box_xyxy(cx, cy, model),
        })

    # Edge cases
    if len(clusters) == 0 and len(gts) == 0:
        return {
            "TP": 0, "FP": 0, "FN": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "mean_center_error": float("nan"),
            "center_errors": [],
            "matches": [],
            "model_counts": {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]},
        }
    if len(gts) == 0:
        return {
            "TP": 0, "FP": len(clusters), "FN": 0,
            "precision": 0.0, "recall": 1.0, "f1": 0.0,
            "mean_center_error": float("nan"),
            "center_errors": [],
            "matches": [],
            "model_counts": {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]},
        }
    if len(clusters) == 0:
        mc = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}
        for gt in gts:
            mc[gt["model"]]["FN"] += 1
        return {
            "TP": 0, "FP": 0, "FN": len(gts),
            "precision": 1.0, "recall": 0.0, "f1": 0.0,
            "mean_center_error": float("nan"),
            "center_errors": [],
            "matches": [],
            "model_counts": mc,
        }

    # ---- cost matrices ----
    M, K = len(clusters), len(gts)
    dist_mat = np.zeros((M, K), float)
    iou_mat = np.zeros((M, K), float)
    cost = np.zeros((M, K), float)

    for i, c in enumerate(clusters):
        for j, g in enumerate(gts):
            d = float(np.linalg.norm(c["center"] - g["center"]))
            iou = float(iou_aabb(c["box"], g["box"]))
            dist_mat[i, j] = d
            iou_mat[i, j] = iou
            cost[i, j] = d + cost_iou_w * (1.0 - iou)

    rr, cc = _hungarian_or_greedy(cost)

    # ---- accept matches with thresholds ----
    matches = []
    used_cid = set()
    used_gid = set()
    center_errors = []

    for i, j in zip(rr, cc):
        d = float(dist_mat[i, j])
        iou = float(iou_mat[i, j])
        if (d <= dist_thr) and (iou >= iou_thr):
            cid = clusters[i]["cid"]
            gid = gts[j]["gid"]
            matches.append((cid, gid, d, iou))
            used_cid.add(cid)
            used_gid.add(gid)
            center_errors.append(d)

    TP = len(matches)
    FP = len(clusters) - TP
    FN = len(gts) - TP

    precision = TP / (TP + FP) if (TP + FP) else 1.0
    recall = TP / (TP + FN) if (TP + FN) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_center_error = float(np.mean(center_errors)) if center_errors else float("nan")

    # ---- model stats ----
    model_counts = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}

    # TP/FN are GT-side
    for gt in gts:
        if gt["gid"] in used_gid:
            model_counts[gt["model"]]["TP"] += 1
        else:
            model_counts[gt["model"]]["FN"] += 1

    # FP are cluster-side: assign to nearest GT within fp_assign_dist
    gt_centers = np.vstack([gt["center"] for gt in gts])
    gt_models = np.array([gt["model"] for gt in gts], dtype=int)

    for c in clusters:
        if c["cid"] in used_cid:
            continue
        dists = np.linalg.norm(gt_centers - c["center"], axis=1)
        j = int(np.argmin(dists))
        if float(dists[j]) <= fp_assign_dist:
            model_counts[int(gt_models[j])]["FP"] += 1
    
    unmatched_clusters = [c["cid"] for c in clusters if c["cid"] not in used_cid]
    unmatched_gts = [gt["gid"] for gt in gts if gt["gid"] not in used_gid]

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mean_center_error": mean_center_error,
        "center_errors": center_errors,
        "matches": matches,
        "unmatched_clusters": unmatched_clusters,
        "unmatched_gts": unmatched_gts,
        "model_counts": model_counts,
    }
