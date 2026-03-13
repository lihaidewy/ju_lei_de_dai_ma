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
