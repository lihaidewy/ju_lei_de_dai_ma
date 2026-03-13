import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================
# GT尺寸：model = ID % 3
# =========================
_GT_DIM = {
    0: {"L": 5.06, "W": 2.22},
    1: {"L": 4.32, "W": 2.19},
    2: {"L": 3.55, "W": 2.58},
}

# -----------------------多先验固定 L/W 的矩形拟合（中心吸引型损失）---------------------------------

def _is_inside_rect(u: float, v: float, L: float, W: float) -> bool:
    a = L * 0.5
    b = W * 0.5
    return (abs(u) <= a) and (abs(v) <= b)


def _point_to_rect_center_dist(u: float, v: float, L: float, W: float) -> float:
    """点(u,v)到矩形中心的归一化距离（鼓励点靠近中心，而非边界）
    返回 sqrt((u/(L/2))^2 + (v/(W/2))^2)
    """
    a = max(L * 0.5, 1e-6)
    b = max(W * 0.5, 1e-6)
    return float(((u / a) ** 2 + (v / b) ** 2) ** 0.5)


def _huber(r, delta=0.5):
    r = abs(float(r))
    if r <= delta:
        return 0.5 * r * r
    return delta * (r - 0.5 * delta)


def rect_xyxy_from_center(cx, cy, L, W):
    return (cx - L/2, cy - W/2, L, W)


def fit_center_fixed_yaw(points_xy, L, W, yaw=0.0, steps=50, step_size=0.5, huber_delta=0.5):
    """固定尺寸+固定yaw，拟合中心，使点更靠近中心（坐标下降）"""
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return (np.nan, np.nan), np.inf

    c = np.cos(yaw); s = np.sin(yaw)
    R_T = np.array([[ c, s],
                    [-s, c]], float)  # world -> box frame

    center = pts.mean(axis=0).astype(float)

    def loss_at(cxy):
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
        for dx, dy in [(step,0), (-step,0), (0,step), (0,-step)]:
            c2 = center + np.array([dx, dy], float)
            v2 = loss_at(c2)
            if v2 < best:
                center, best = c2, v2
                improved = True
        if not improved:
            step *= 0.7
            if step < 0.05:
                break

    return (float(center[0]), float(center[1])), float(best)


def choose_best_fixed_box_prior(
    points_xy,
    priors,
    yaw=0.0,
    steps=50,
    step_size=0.5,
    huber_delta=0.5,
    score_lambda=1.0,
):
    """每簇对多个(L,W)先验试拟合，选score最小的一个。
    score = (loss/(L+W)) + score_lambda*(1-inside_ratio)
    """
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return None

    c = np.cos(yaw); s = np.sin(yaw)
    R_T = np.array([[ c, s],
                    [-s, c]], float)

    best = None
    for k, (L, W) in enumerate(priors):
        (cx, cy), loss = fit_center_fixed_yaw(
            pts, L, W, yaw=yaw, steps=steps, step_size=step_size, huber_delta=huber_delta
        )
        q = (pts - np.array([cx, cy], float)) @ R_T.T
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
            "loss": float(loss),
            "inside_ratio": float(inside_ratio),
            "score": float(score),
        }
        if best is None or cand["score"] < best["score"]:
            best = cand

    return best

# =========================
# 贴边型 loss：框外强惩罚 + 框内离边太远才惩罚（更稳）
# =========================

def _dist_outside_rect(u: float, v: float, L: float, W: float) -> float:
    """distance to rectangle if point is outside; 0 if inside"""
    a = max(L * 0.5, 1e-6)
    b = max(W * 0.5, 1e-6)
    du = max(0.0, abs(u) - a)
    dv = max(0.0, abs(v) - b)
    return float((du * du + dv * dv) ** 0.5)

def _dist_to_nearest_edge_inside(u: float, v: float, L: float, W: float) -> float:
    """for inside points: min distance to rectangle boundary (positive if inside)"""
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
    inside_margin: float = 0.5,   # 框内点离边界>margin 才开始惩罚
    alpha_out: float = 10.0,      # 框外惩罚权重（要大）
    beta_in: float = 1.0,         # 框内“太靠中心”惩罚权重（要小）
):
    """
    用坐标下降拟合 center。
    loss = alpha_out * E[huber(d_out)] + beta_in * E[huber(max(0, d_in - inside_margin))]
    """
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return (np.nan, np.nan), np.inf

    c = np.cos(yaw)
    s = np.sin(yaw)
    R_T = np.array([[c, s], [-s, c]], dtype=float)  # world -> box frame

    center = pts.mean(axis=0).astype(float)

    def loss_at(cxy: np.ndarray) -> float:
        q = (pts - cxy) @ R_T.T
        out_list = []
        in_list = []
        for u, v in q:
            d_out = _dist_outside_rect(u, v, L, W)
            out_list.append(_huber(d_out, delta=huber_delta))

            if d_out <= 1e-9:  # inside
                d_in = _dist_to_nearest_edge_inside(u, v, L, W)
                # 只惩罚“离边界太远”的部分
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
    score_lambda: float = 1.0,     # 继续保留 inside_ratio 约束
    inside_margin: float = 0.5,
    alpha_out: float = 10.0,
    beta_in: float = 1.0,
):
    """
    多先验选择：拟合中心使用“贴边型 loss”，同时仍用 inside_ratio 做合理性约束。
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
            "box": rect_xyxy_from_center(cx, cy, L, W),
            "loss": float(loss),
            "inside_ratio": float(inside_ratio),
            "score": float(score),
        }
        if best is None or cand["score"] < best["score"]:
            best = cand

    return best


# =========================
# AABB IoU + 合并（考虑速度一致）
# =========================

def _iou_aabb_xyxy(b1: np.ndarray, b2: np.ndarray) -> float:
    b1 = np.asarray(b1, float); b2 = np.asarray(b2, float)
    if np.any(np.isnan(b1)) or np.any(np.isnan(b2)):
        return 0.0
    xA = max(float(b1[0]), float(b2[0]))
    yA = max(float(b1[1]), float(b2[1]))
    xB = min(float(b1[2]), float(b2[2]))
    yB = min(float(b1[3]), float(b2[3]))
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    area1 = max(0.0, float(b1[2] - b1[0])) * max(0.0, float(b1[3] - b1[1]))
    area2 = max(0.0, float(b2[2] - b2[0])) * max(0.0, float(b2[3] - b2[1]))
    union = area1 + area2 - inter
    if union <= 1e-9:
        return 0.0
    return float(inter / union)

class _UnionFind:
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
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def merge_overlapping_clusters_by_speed(
    cluster_items: list,
    priors: list,
    yaw: float = 0.0,
    merge_iou_thr: float = 0.35,
    merge_v_thr: float = 1.0,
    # 重新拟合用贴边loss的参数
    edge_steps: int = 60,
    edge_step_size: float = 0.5,
    edge_huber_delta: float = 0.5,
    edge_inside_margin: float = 0.5,
    edge_alpha_out: float = 10.0,
    edge_beta_in: float = 1.0,
    score_lambda: float = 1.0,
):
    """
    cluster_items: 每个元素至少包含
      {
        "cid": int,
        "points": (N,2),
        "mean_v": float,
        "best": { "center","box","prior_id","score"... }  # 初次拟合结果
      }
    返回：合并后的同结构 list（合并组会新生成 cid 为负数或组合id）
    """
    n = len(cluster_items)
    if n <= 1:
        return cluster_items

    uf = _UnionFind(n)

    # 先根据 IoU + 速度一致性建连通关系
    for i in range(n):
        bi = cluster_items[i]["best"]["box"]
        vi = float(cluster_items[i]["mean_v"])
        for j in range(i+1, n):
            bj = cluster_items[j]["best"]["box"]
            vj = float(cluster_items[j]["mean_v"])
            if abs(vi - vj) > merge_v_thr:
                continue
            iou = _iou_aabb_xyxy(bi, bj)
            if iou >= merge_iou_thr:
                uf.union(i, j)

    groups = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    merged = []
    for _, idxs in groups.items():
        if len(idxs) == 1:
            merged.append(cluster_items[idxs[0]])
            continue

        # 合并点集
        pts = np.concatenate([cluster_items[k]["points"] for k in idxs], axis=0)
        vs = np.concatenate([np.full((len(cluster_items[k]["points"]),), float(cluster_items[k]["mean_v"]))
                             for k in idxs], axis=0)  # 仅用于估计均速（简单版）
        mean_v = float(np.mean(vs)) if vs.size else float(np.mean([cluster_items[k]["mean_v"] for k in idxs]))

        # 重新选 prior（贴边loss）
        best = choose_best_fixed_box_prior_edge_loss(
            pts, priors=priors, yaw=yaw,
            steps=edge_steps, step_size=edge_step_size, huber_delta=edge_huber_delta,
            score_lambda=score_lambda,
            inside_margin=edge_inside_margin,
            alpha_out=edge_alpha_out,
            beta_in=edge_beta_in,
        )

        merged.append({
            "cid": -int(sum([cluster_items[k]["cid"] for k in idxs])),  # 生成一个合并cid（你也可以换成别的规则）
            "points": pts,
            "mean_v": mean_v,
            "best": best,
            "merged_from": [cluster_items[k]["cid"] for k in idxs],
        })

    return merged


# ----------------------- GT box ---------------------------------

def _gt_boxes_from_list(gt_list):
    """gt_list: [{'id','x','y','model'},...] -> list[(gid, (xmin,ymin,w,h), model)]"""
    out = []
    for g in gt_list:
        gid = int(g["id"])
        cx = float(g["x"])
        cy = float(g["y"])
        model = int(g["model"])
        L = float(_GT_DIM[model]["L"])
        W = float(_GT_DIM[model]["W"])
        out.append((gid, (cx - L/2, cy - W/2, L, W), model))
    return out


# ----------------------- Plot main ---------------------------------

def plot_raw_and_clusters(
    pts_xy,
    labels,
    v=None,
    gt_list=None,
    star_indices=None,
    fig=None,
    axes=None,
    title="",
    use_fixed_box=True,
    fixed_box_priors=None,        # list[(L,W)] or None -> use 3 GT priors
    fixed_box_yaw=0.0,
    fixed_box_steps=50,
    fixed_box_step_size=0.5,
    fixed_box_huber_delta=0.5,
    fixed_box_score_lambda=1.0,
):
    """
    Left: raw points (+GT boxes)
    Right: clustered points (+per-cluster fixed-box fit when use_fixed_box=True)
    """

    pts = np.asarray(pts_xy, float)
    labels = np.asarray(labels)
    if v is None:
        v = np.zeros((pts.shape[0],), float)
    else:
        v = np.asarray(v, float)

    if fixed_box_priors is None:
        fixed_box_priors = [
            (_GT_DIM[0]["L"], _GT_DIM[0]["W"]),
            (_GT_DIM[1]["L"], _GT_DIM[1]["W"]),
            (_GT_DIM[2]["L"], _GT_DIM[2]["W"]),
        ]

    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    ax_raw, ax_clu = axes
    ax_raw.cla()
    ax_clu.cla()

    ax_raw.set_title("raw")
    ax_clu.set_title("clusters")

    # 1) raw plot colored by v
    sc = ax_raw.scatter(pts[:, 0], pts[:, 1], c=v, s=10)
    ax_raw.set_xlabel("X")
    ax_raw.set_ylabel("Y")
    ax_raw.grid(True)

    # colorbar reuse
    if hasattr(fig, "_raw_cbar") and fig._raw_cbar is not None:
        try:
            fig._raw_cbar.remove()
        except Exception:
            pass
        fig._raw_cbar = None
    fig._raw_cbar = fig.colorbar(sc, ax=ax_raw, fraction=0.046, pad=0.04)
    fig._raw_cbar.set_label("V")

    # draw GT boxes on raw
    if gt_list is not None and len(gt_list) > 0:
        for gid, (xmin, ymin, w, h), model in _gt_boxes_from_list(gt_list):
            ax_raw.add_patch(Rectangle((xmin, ymin), w, h,
                                       fill=False, linestyle="--", linewidth=1.5))
            ax_raw.text(xmin, ymin + h, f"GT:{gid} M:{model}", fontsize=8)

    # 2) clusters plot
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mask_noise = labels <= 0
    if np.any(mask_noise):
        ax_clu.scatter(pts[mask_noise, 0], pts[mask_noise, 1], s=8, alpha=0.4)

    u_ids = np.unique(labels[labels > 0])
    for cid in u_ids:
        m = labels == cid
        cpts = pts[m]
        if cpts.size == 0:
            continue

        col = colors[(int(cid) - 1) % len(colors)]
        ax_clu.scatter(cpts[:, 0], cpts[:, 1], s=10)

        # cluster mean velocity label
        mean_v = float(np.mean(v[m])) if np.any(m) else 0.0

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
            cx_fit, cy_fit = best["center"]
            L, W = best["L"], best["W"]
            prior_id = best["prior_id"]

            xmin, ymin, w, h = rect_xyxy_from_center(cx_fit, cy_fit, L, W)
            ax_clu.add_patch(Rectangle((xmin, ymin), w, h, fill=False, linewidth=1.8))
            ax_clu.scatter([cx_fit], [cy_fit], marker="x", s=35)

            ax_clu.text(xmin, ymin + h,
                        f"ID:{int(cid)}  V:{mean_v:.1f}  P:{prior_id}",
                        fontsize=8)
        else:
            # fallback: no box / or you can draw AABB if desired
            cx, cy = float(np.mean(cpts[:, 0])), float(np.mean(cpts[:, 1]))
            ax_clu.text(cx, cy, f"ID:{int(cid)}  V:{mean_v:.1f}", fontsize=8)

    # optional stars (draw once on cluster axis)
    if star_indices is not None and len(star_indices) > 0:
        star_indices = np.asarray(star_indices, int)
        star_indices = star_indices[(star_indices >= 0) & (star_indices < pts.shape[0])]
        if star_indices.size > 0:
            star_pts = pts[star_indices]
            ax_clu.scatter(star_pts[:, 0], star_pts[:, 1], marker="*", s=90)

    ax_clu.set_xlabel("X")
    ax_clu.grid(True)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes
