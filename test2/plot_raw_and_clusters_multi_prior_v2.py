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

# -----------------------
# helpers
# -----------------------

def fixed_box_xyxy_from_center(cx: float, cy: float, L: float, W: float) -> np.ndarray:
    """
    当前坐标系：
    X = 横向
    Y = 前向

    因此固定朝向下：
    - 宽 W 沿 X
    - 长 L 沿 Y

    返回 [xmin, ymin, xmax, ymax]
    """
    return np.array([cx - W / 2, cy - L / 2, cx + W / 2, cy + L / 2], dtype=float)


def rect_xywh_from_xyxy(xyxy: np.ndarray):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return x1, y1, (x2 - x1), (y2 - y1)


def _huber(r, delta=0.5):
    r = abs(float(r))
    if r <= delta:
        return 0.5 * r * r
    return delta * (r - 0.5 * delta)


# =============================================================================
# Mode A) 中心吸引型 loss（points closer to center）
# =============================================================================

def _point_to_rect_center_dist(u: float, v: float, L: float, W: float) -> float:
    """
    当前坐标系：
    u -> X(横向)
    v -> Y(前向)

    所以：
    - X方向宽 W
    - Y方向长 L
    """
    a = max(W * 0.5, 1e-6)  # X方向半宽
    b = max(L * 0.5, 1e-6)  # Y方向半长
    return float(((u / a) ** 2 + (v / b) ** 2) ** 0.5)


def fit_center_fixed_yaw_center_loss(points_xy, L, W, yaw=0.0, steps=50, step_size=0.5, huber_delta=0.5):
    """
    固定尺寸 + 固定yaw，拟合中心，使点更靠近矩形中心（坐标下降）

    当前坐标系：
    - X：横向
    - Y：前向

    在当前实现中：
    - yaw=0 时，局部坐标 u/v 与世界 X/Y 对齐
    - 通过几何定义实现：u方向对应宽W，v方向对应长L
    """
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return (np.nan, np.nan), np.inf

    c = np.cos(yaw)
    s = np.sin(yaw)
    R_T = np.array([[c, s],
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
        for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
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


def choose_best_fixed_box_prior_center_loss(
    points_xy,
    priors,
    yaw=0.0,
    steps=50,
    step_size=0.5,
    huber_delta=0.5,
    score_lambda=1.0,
):
    """
    多先验选择：中心吸引型 loss
    score = loss/(L+W) + score_lambda*(1-inside_ratio)
    """
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return None

    c = np.cos(yaw)
    s = np.sin(yaw)
    R_T = np.array([[c, s],
                    [-s, c]], float)

    best = None
    for k, (L, W) in enumerate(priors):
        (cx, cy), loss = fit_center_fixed_yaw_center_loss(
            pts, L, W, yaw=yaw, steps=steps, step_size=step_size, huber_delta=huber_delta
        )

        q = (pts - np.array([cx, cy], float)) @ R_T.T

        # 当前坐标系下：
        # q[:,0] -> X方向，对应宽W
        # q[:,1] -> Y方向，对应长L
        a = W * 0.5
        b = L * 0.5
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
# Mode B) 贴边型 loss（框外强惩罚 + 框内离边太远才惩罚）
# =============================================================================

def _dist_outside_rect(u: float, v: float, L: float, W: float) -> float:
    """
    点在矩形外时，到矩形的距离；若在内部则为0

    当前坐标系：
    - u -> X(横向)，对应宽W
    - v -> Y(前向)，对应长L
    """
    a = max(W * 0.5, 1e-6)  # X方向半宽
    b = max(L * 0.5, 1e-6)  # Y方向半长
    du = max(0.0, abs(u) - a)
    dv = max(0.0, abs(v) - b)
    return float((du * du + dv * dv) ** 0.5)


def _dist_to_nearest_edge_inside(u: float, v: float, L: float, W: float) -> float:
    """
    点在矩形内部时，到最近边界的距离

    当前坐标系：
    - u -> X(横向)，对应宽W
    - v -> Y(前向)，对应长L
    """
    a = max(W * 0.5, 1e-6)  # X方向半宽
    b = max(L * 0.5, 1e-6)  # Y方向半长
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
    """
    loss = alpha_out * E[huber(d_out)] + beta_in * E[huber(max(0, d_in - inside_margin))]

    当前坐标系：
    - X：横向
    - Y：前向
    - yaw=0 时通过宽高映射实现：
      X方向宽W，Y方向长L
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
    """多先验选择：贴边型 loss + inside_ratio 约束；返回 box 为 xyxy"""
    pts = np.asarray(points_xy, float)
    if pts.size == 0:
        return None

    c = np.cos(yaw)
    s = np.sin(yaw)
    R_T = np.array([[c, s], [-s, c]], dtype=float)

    best = None
    for k, (L, W) in enumerate(priors):
        (cx, cy), loss = fit_center_fixed_yaw_edge_loss(
            pts,
            L=L,
            W=W,
            yaw=yaw,
            steps=steps,
            step_size=step_size,
            huber_delta=huber_delta,
            inside_margin=inside_margin,
            alpha_out=alpha_out,
            beta_in=beta_in
        )

        q = (pts - np.array([cx, cy], dtype=float)) @ R_T.T

        # 当前坐标系下：
        # q[:,0] -> X方向，对应宽W
        # q[:,1] -> Y方向，对应长L
        a = W * 0.5
        b = L * 0.5
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
# Unified switch
# =============================================================================

def choose_best_fixed_box_prior_mode(
    points_xy: np.ndarray,
    priors: list,
    fit_mode: str = "center",  # "center" or "edge"
    yaw: float = 0.0,
    # center-loss params
    steps: int = 50,
    step_size: float = 0.5,
    huber_delta: float = 0.5,
    score_lambda: float = 1.0,
    # edge-loss extra params
    inside_margin: float = 0.5,
    alpha_out: float = 10.0,
    beta_in: float = 1.0,
):
    fit_mode = (fit_mode or "center").lower().strip()

    if fit_mode in ("center", "center_loss", "pull_center"):
        return choose_best_fixed_box_prior_center_loss(
            points_xy,
            priors=priors,
            yaw=yaw,
            steps=steps,
            step_size=step_size,
            huber_delta=huber_delta,
            score_lambda=score_lambda,
        )

    if fit_mode in ("edge", "edge_loss", "boundary"):
        return choose_best_fixed_box_prior_edge_loss(
            points_xy,
            priors=priors,
            yaw=yaw,
            steps=max(steps, 60),
            step_size=step_size,
            huber_delta=huber_delta,
            score_lambda=score_lambda,
            inside_margin=inside_margin,
            alpha_out=alpha_out,
            beta_in=beta_in,
        )

    raise ValueError(f"Unknown fit_mode={fit_mode}. Use 'center' or 'edge'.")


# =============================================================================
# Plot main
# =============================================================================

def _gt_boxes_from_list(gt_list):
    """
    gt_list: [{'id','x','y','model'},...]
    -> list[(gid, (xmin,ymin,w,h), model)]

    当前坐标系：
    X = 横向
    Y = 前向
    """
    out = []
    for g in gt_list:
        gid = int(g["id"])
        cx = float(g["x"])
        cy = float(g["y"])
        model = int(g["model"])

        L = float(_GT_DIM[model]["L"])
        W = float(_GT_DIM[model]["W"])

        xyxy = fixed_box_xyxy_from_center(cx, cy, L, W)
        x, y, w, h = rect_xywh_from_xyxy(xyxy)
        out.append((gid, (x, y, w, h), model))

    return out


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
    fixed_box_priors=None,
    fixed_box_yaw=0.0,
    fixed_box_steps=50,
    fixed_box_step_size=0.5,
    fixed_box_huber_delta=0.5,
    fixed_box_score_lambda=1.0,
    fixed_box_fit_mode: str = "center",  # "center" or "edge"
    # edge-loss extra
    fixed_box_inside_margin: float = 0.5,
    fixed_box_alpha_out: float = 10.0,
    fixed_box_beta_in: float = 1.0,
):
    """
    Left: raw points (+GT boxes)
    Right: clustered points (+per-cluster fixed-box fit when use_fixed_box=True)

    当前坐标系：
    - X：横向
    - Y：前向
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
    ax_clu.set_title(f"clusters ({fixed_box_fit_mode})")

    # raw plot colored by v
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
            ax_raw.add_patch(
                Rectangle((xmin, ymin), w, h, fill=False, linestyle="--", linewidth=1.5)
            )
            ax_raw.text(xmin, ymin + h, f"GT:{gid} M:{model}", fontsize=8)

    # clusters plot
    mask_noise = labels <= 0
    if np.any(mask_noise):
        ax_clu.scatter(pts[mask_noise, 0], pts[mask_noise, 1], s=8, alpha=0.4)

    u_ids = np.unique(labels[labels > 0])
    for cid in u_ids:
        m = labels == cid
        cpts = pts[m]
        if cpts.size == 0:
            continue

        ax_clu.scatter(cpts[:, 0], cpts[:, 1], s=10)
        mean_v = float(np.mean(v[m])) if np.any(m) else 0.0

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

            if best is not None:
                cx_fit, cy_fit = best["center"]
                prior_id = best["prior_id"]
                xyxy = best["box"]
                xmin, ymin, w, h = rect_xywh_from_xyxy(xyxy)

                ax_clu.add_patch(
                    Rectangle((xmin, ymin), w, h, fill=False, linewidth=1.8)
                )
                ax_clu.scatter([cx_fit], [cy_fit], marker="x", s=35)
                ax_clu.text(
                    xmin,
                    ymin + h,
                    f"ID:{int(cid)}  V:{mean_v:.1f}  P:{prior_id}",
                    fontsize=8
                )
        else:
            cx = float(np.mean(cpts[:, 0]))
            cy = float(np.mean(cpts[:, 1]))
            ax_clu.text(cx, cy, f"ID:{int(cid)}  V:{mean_v:.1f}", fontsize=8)

    if star_indices is not None and len(star_indices) > 0:
        star_indices = np.asarray(star_indices, int)
        star_indices = star_indices[(star_indices >= 0) & (star_indices < pts.shape[0])]
        if star_indices.size > 0:
            star_pts = pts[star_indices]
            ax_clu.scatter(star_pts[:, 0], star_pts[:, 1], marker="*", s=90)
    
    ax_raw.set_autoscale_on(False)
    ax_clu.set_autoscale_on(False)
    ax_clu.set_xlabel("X")
    ax_clu.set_ylabel("Y")
    ax_clu.grid(True)

    if title:
        fig.suptitle(title)

    # fig.tight_layout()
    return fig, axes
