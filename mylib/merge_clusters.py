import numpy as np

def _aabb_iou(b1, b2):
    xA = max(float(b1[0]), float(b2[0]))
    yA = max(float(b1[1]), float(b2[1]))
    xB = min(float(b1[2]), float(b2[2]))
    yB = min(float(b1[3]), float(b2[3]))
    iw = max(0.0, xB - xA)
    ih = max(0.0, yB - yA)
    inter = iw * ih
    a1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    a2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    u = a1 + a2 - inter
    return 0.0 if u <= 1e-9 else float(inter / u)

class _DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def _prior_fit(choose_best_fixed_box_prior, cpts, priors, yaw=0.0, score_lambda=1.0):
    """
    兼容两种返回：
      1) dict: {"center":(cx,cy), "L":..., "W":..., "prior_id":..., "score":...}
      2) tuple: ((cx,cy), pid, (L,W), score)
    """
    best = choose_best_fixed_box_prior(cpts, priors=priors, yaw=yaw, score_lambda=score_lambda)

    if isinstance(best, dict):
        cx, cy = best["center"]
        L, W = best["L"], best["W"]
        pid = best.get("prior_id", -1)
        score = best.get("score", np.nan)
        return float(cx), float(cy), float(L), float(W), int(pid), float(score)

    # tuple style
    (cx, cy), pid, lw, score = best
    L, W = lw
    return float(cx), float(cy), float(L), float(W), int(pid), float(score)

def merge_overlapping_clusters_fixed_box(
    pts_xy: np.ndarray,
    v: np.ndarray,
    labels: np.ndarray,
    choose_best_fixed_box_prior,
    priors,
    yaw=0.0,
    score_lambda=1.0,
    # 合并判据（默认值建议从保守开始）
    iou_thr_merge=0.12,
    center_dist_thr_merge=5.0,
    v_diff_thr_merge=2.0,
):
    """
    输入：原始 labels（noise=-1，cluster>=1）
    输出：labels_merged（连续1..K），merge_log（便于你打印/可视化检查）
    """

    pts = np.asarray(pts_xy, float)
    labels = np.asarray(labels)

    if v is None:
        v = np.zeros((pts.shape[0],), float)
    else:
        v = np.asarray(v, float)

    # 1) 收集原始 clusters，并用固定框拟合得到 box/center
    cids = sorted([int(c) for c in np.unique(labels) if int(c) >= 1])
    if len(cids) <= 1:
        return labels.copy(), []  # 无需合并

    clusters = []
    for cid in cids:
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        cpts = pts[idx]
        cx, cy, L, W, pid, score = _prior_fit(
            choose_best_fixed_box_prior, cpts, priors, yaw=yaw, score_lambda=score_lambda
        )
        box = np.array([cx - L/2, cy - W/2, cx + L/2, cy + W/2], float)
        v_mean = float(np.mean(v[idx])) if idx.size else 0.0
        clusters.append({
            "cid": cid,
            "idx": idx,
            "center": np.array([cx, cy], float),
            "box": box,
            "v_mean": v_mean,
            "pid": pid,
            "score": score,
            "L": L,
            "W": W,
        })

    n = len(clusters)
    dsu = _DSU(n)

    # 2) 两两检测重叠并 union
    for i in range(n):
        for j in range(i + 1, n):
            iou = _aabb_iou(clusters[i]["box"], clusters[j]["box"])
            if iou <= iou_thr_merge:
                continue

            dc = float(np.linalg.norm(clusters[i]["center"] - clusters[j]["center"]))
            if dc > center_dist_thr_merge:
                continue

            dv = abs(clusters[i]["v_mean"] - clusters[j]["v_mean"])
            if dv > v_diff_thr_merge:
                continue

            dsu.union(i, j)

    # 3) 根据并查集形成合并组
    groups = {}
    for i in range(n):
        r = dsu.find(i)
        groups.setdefault(r, []).append(i)

    # 4) 生成合并后的 labels（连续编号）
    new_labels = labels.copy()
    # 先清空所有 cluster label（>=1），保留 noise<=0
    new_labels[new_labels >= 1] = 0

    merge_log = []
    new_cid = 1

    for members in groups.values():
        merged_idx = np.concatenate([clusters[k]["idx"] for k in members], axis=0)
        merged_pts = pts[merged_idx]

        # 对合并后的点集重新选 prior / 重拟合中心（很重要）
        cx, cy, L, W, pid, score = _prior_fit(
            choose_best_fixed_box_prior, merged_pts, priors, yaw=yaw, score_lambda=score_lambda
        )
        new_labels[merged_idx] = new_cid

        merge_log.append({
            "new_cid": new_cid,
            "old_cids": [clusters[k]["cid"] for k in members],
            "pid": pid,
            "center": (cx, cy),
            "L": L,
            "W": W,
            "score": score,
        })
        new_cid += 1

    return new_labels, merge_log
