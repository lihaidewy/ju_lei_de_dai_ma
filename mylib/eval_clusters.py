import numpy as np

# --------- GT尺寸：model = ID % 3 ----------
DIM = {
    0: {"L": 5.06, "W": 2.22},
    1: {"L": 4.32, "W": 2.19},
    2: {"L": 3.55, "W": 2.58},
}

def gt_box_xyxy(x, y, model):
    """车辆朝向固定：L沿+X, W沿+Y，返回AABB: [xmin,ymin,xmax,ymax]"""
    L = DIM[int(model)]["L"]
    W = DIM[int(model)]["W"]
    return np.array([x - L/2, y - W/2, x + L/2, y + W/2], dtype=float)

def aabb_from_points(pts_xy):
    mn = pts_xy.min(axis=0)
    mx = pts_xy.max(axis=0)
    return np.array([mn[0], mn[1], mx[0], mx[1]], dtype=float)

def iou_aabb(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    a1 = max(0.0, b1[2]-b1[0]) * max(0.0, b1[3]-b1[1])
    a2 = max(0.0, b2[2]-b2[0]) * max(0.0, b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 1e-9 else 0.0

def points_in_aabb(pts_xy, box):
    return ((pts_xy[:,0] >= box[0]) & (pts_xy[:,0] <= box[2]) &
            (pts_xy[:,1] >= box[1]) & (pts_xy[:,1] <= box[3]))

def _hungarian_or_greedy(cost):
    """优先用匈牙利；没scipy就用贪心。返回匹配 (row_indices, col_indices)"""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return r, c
    except Exception:
        # greedy: 逐个选最小cost，避免重复
        M, K = cost.shape
        used_r, used_c = set(), set()
        pairs = []
        flat = [(cost[i,j], i, j) for i in range(M) for j in range(K)]
        flat.sort(key=lambda x: x[0])
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

def eval_one_frame(pts_xy, labels, gt_list,
                   dist_thr=4.0, iou_thr=0.10,
                   cost_iou_w=2.0,
                   assign_fp_dist=8.0):
    """
    pts_xy: (N,2)
    labels: (N,)  噪声=-1 / <=0
    gt_list: [{'id':int,'x':float,'y':float,'model':int}, ...]
    输出：TP/FP/FN、mean_center_error、每车点级F1、并给出 model 统计用的计数
    """
    pts_xy = np.asarray(pts_xy, float)
    labels = np.asarray(labels).reshape(-1)

    # --- clusters ---
    cids = [c for c in np.unique(labels) if c > 0]
    clusters = []
    for cid in cids:
        idx = np.where(labels == cid)[0]
        cpts = pts_xy[idx]
        center = cpts.mean(axis=0)
        box = aabb_from_points(cpts)
        clusters.append({"cid": int(cid), "idx": idx, "center": center, "box": box})

    # --- gts ---
    gts = []
    for g in gt_list:
        center = np.array([g["x"], g["y"]], dtype=float)
        box = gt_box_xyxy(g["x"], g["y"], g["model"])
        gts.append({"gid": int(g["id"]), "model": int(g["model"]), "center": center, "box": box})

    # 边界
    if len(clusters) == 0 and len(gts) == 0:
        return {"TP":0,"FP":0,"FN":0,"precision":1.0,"recall":1.0,"f1":1.0,
                "mean_center_error":np.nan,"matches":[],
                "per_vehicle_point_f1":{}, "model_counts":{}}
    if len(gts) == 0:
        return {"TP":0,"FP":len(clusters),"FN":0,"precision":0.0,"recall":1.0,"f1":0.0,
                "mean_center_error":np.nan,"matches":[],
                "per_vehicle_point_f1":{}, "model_counts":{}}
    if len(clusters) == 0:
        perv = {gt["gid"]:0.0 for gt in gts}
        model_counts = {m:{"TP":0,"FP":0,"FN":sum(gt["model"]==m for gt in gts)} for m in [0,1,2]}
        return {"TP":0,"FP":0,"FN":len(gts),"precision":1.0,"recall":0.0,"f1":0.0,
                "mean_center_error":np.nan,"matches":[],
                "per_vehicle_point_f1":perv, "model_counts":model_counts}

    # --- cost matrix (distance + iou) ---
    M, K = len(clusters), len(gts)
    dist_mat = np.zeros((M, K), float)
    iou_mat  = np.zeros((M, K), float)
    cost     = np.zeros((M, K), float)

    for i, c in enumerate(clusters):
        for j, g in enumerate(gts):
            d = np.linalg.norm(c["center"] - g["center"])
            iou = iou_aabb(c["box"], g["box"])
            dist_mat[i,j] = d
            iou_mat[i,j]  = iou
            cost[i,j] = d + cost_iou_w * (1.0 - iou)

    rr, cc = _hungarian_or_greedy(cost)

    matches = []
    used_cid = set()
    used_gid = set()
    center_errors = []

    for i, j in zip(rr, cc):
        d = dist_mat[i,j]
        iou = iou_mat[i,j]
        if (d <= dist_thr) and (iou >= iou_thr):
            cid = clusters[i]["cid"]
            gid = gts[j]["gid"]
            matches.append((cid, gid, float(d), float(iou)))
            used_cid.add(cid)
            used_gid.add(gid)
            center_errors.append(float(d))

    TP = len(matches)
    FP = len(clusters) - TP
    FN = len(gts) - TP
    precision = TP / (TP + FP) if (TP+FP) else 1.0
    recall    = TP / (TP + FN) if (TP+FN) else 1.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    mean_center_error = float(np.mean(center_errors)) if center_errors else np.nan

    # --- 点级F1（每车）：用“车矩形内点” vs “匹配簇点” ---
    gid2cid = {gid: cid for (cid, gid, _, _) in matches}
    per_vehicle_point_f1 = {}

    for g in gts:
        gid = g["gid"]
        in_gt = points_in_aabb(pts_xy, g["box"])
        if gid not in gid2cid:
            per_vehicle_point_f1[gid] = 0.0
            continue

        cid = gid2cid[gid]
        in_c = (labels == cid)

        tp_pts = int((in_gt & in_c).sum())
        fp_pts = int((~in_gt & in_c).sum())
        fn_pts = int((in_gt & (~in_c)).sum())  # 包含噪声和其他簇

        p = tp_pts / (tp_pts + fp_pts) if (tp_pts + fp_pts) else 1.0
        r = tp_pts / (tp_pts + fn_pts) if (tp_pts + fn_pts) else 0.0
        f = 2*p*r/(p+r) if (p+r) else 0.0
        per_vehicle_point_f1[gid] = float(f)

    # --- model统计：把“匹配到的GT”计TP，没匹配到的GT计FN；
    #     FP：未匹配簇按“最近GT(<=assign_fp_dist)”归属到那个model，否则算unknown(不进model FP)
    model_counts = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0,1,2]}
    gid2model = {g["gid"]: g["model"] for g in gts}
    # TP/FN by model (GT侧)
    for g in gts:
        m = g["model"]
        if g["gid"] in used_gid:
            model_counts[m]["TP"] += 1
        else:
            model_counts[m]["FN"] += 1

    # FP by model (cluster侧，最近GT归属)
    gt_centers = np.vstack([g["center"] for g in gts])
    gt_models  = np.array([g["model"] for g in gts], dtype=int)
    for c in clusters:
        if c["cid"] in used_cid:
            continue
        dists = np.linalg.norm(gt_centers - c["center"], axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= assign_fp_dist:
            model_counts[int(gt_models[j])]["FP"] += 1

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mean_center_error": mean_center_error,
        "matches": matches,
        "per_vehicle_point_f1": per_vehicle_point_f1,
        "model_counts": model_counts
    }
