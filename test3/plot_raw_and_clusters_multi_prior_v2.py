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

# =========================
# Current center strategy
# =========================
BIAS_SPLIT_Y = 100.0
BIAS_Y_NEAR = 1.149
BIAS_Y_FAR = 1.586


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


def compute_cluster_center_mean_bias(cpts: np.ndarray) -> np.ndarray:
    """
    当前主线：
    center = mean(points) + two-segment Y bias correction
    """
    center = np.mean(cpts, axis=0)

    if center[1] < BIAS_SPLIT_Y:
        bias_y = BIAS_Y_NEAR
    else:
        bias_y = BIAS_Y_FAR

    center = center + np.array([0.0, bias_y], dtype=float)
    return center


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


# =============================================================================
# Plot main
# =============================================================================

def plot_raw_and_clusters(
    pts_xy,
    labels,
    v=None,
    gt_list=None,
    star_indices=None,
    fig=None,
    axes=None,
    title="",
    use_fixed_box=True,              # 保留接口但不再使用
    fixed_box_priors=None,           # 保留接口但不再使用
    fixed_box_yaw=0.0,               # 保留接口但不再使用
    fixed_box_steps=50,              # 保留接口但不再使用
    fixed_box_step_size=0.5,         # 保留接口但不再使用
    fixed_box_huber_delta=0.5,       # 保留接口但不再使用
    fixed_box_score_lambda=1.0,      # 保留接口但不再使用
    fixed_box_fit_mode: str = "center",  # 保留接口但不再使用
    fixed_box_inside_margin: float = 0.5, # 保留接口但不再使用
    fixed_box_alpha_out: float = 10.0,    # 保留接口但不再使用
    fixed_box_beta_in: float = 1.0,       # 保留接口但不再使用
):
    """
    Left: raw points (+GT boxes)
    Right: clustered points + cluster centers only

    当前坐标系：
    - X：横向
    - Y：前向

    当前右图中心逻辑：
    - mean center
    - two-segment Y bias correction
    - 不画框，只画中心
    """

    pts = np.asarray(pts_xy, float)
    labels = np.asarray(labels)

    if v is None:
        v = np.zeros((pts.shape[0],), float)
    else:
        v = np.asarray(v, float)

    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    ax_raw, ax_clu = axes
    ax_raw.cla()
    ax_clu.cla()

    ax_raw.set_title("raw")
    ax_clu.set_title("clusters (mean + bias)")

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

        center = compute_cluster_center_mean_bias(cpts)
        cx, cy = float(center[0]), float(center[1])

        # 只画中心
        ax_clu.scatter([cx], [cy], marker="x", s=50, linewidths=2.0)
        ax_clu.text(cx, cy + 2.0, f"ID:{int(cid)}  V:{mean_v:.1f}", fontsize=8)

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

    return fig, axes
