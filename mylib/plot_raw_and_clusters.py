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

def _gt_boxes_from_list(gt_list):
    """
    gt_list: [{'id':int,'x':float,'y':float,'model':int}, ...]
    return: list of (id, (xmin,ymin,w,h))
    """
    out = []
    for g in gt_list:
        gid = int(g["id"])
        x = float(g["x"])
        y = float(g["y"])
        m = int(g["model"])
        L = _GT_DIM[m]["L"]
        W = _GT_DIM[m]["W"]
        xmin = x - L / 2.0
        ymin = y - W / 2.0
        out.append((gid, (xmin, ymin, L, W)))
    return out


def plot_raw_and_clusters(
    pts, vel, labels,
    xlim=(-60, 60), ylim=(0, 400),
    point_size_raw=10, point_size_cluster=15,
    colors=None, title=None, show=True, star_indices=None,
    rear_prob=None, rear_centers=None, rear_prob_thr=0.6,
    gt_list=None
):
    """
    左：原始点云 (X-Y, 速度或 rear_prob 着色) + 可选叠加 GT 框
    右：聚类结果 (噪声灰色 + 簇点彩色 + 包围框 + ID/均速标签 + 统计信息)

    pts: (N,2)  [x,y]
    vel: (N,)   速度
    labels: (N,) 聚类标签，噪声<=0（如 -1）
    gt_list: [{'id':..,'x':..,'y':..,'model':..}, ...]  真值车辆
    """

    pts = np.asarray(pts)
    vel = np.asarray(vel).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    # 默认颜色表（循环使用）
    if colors is None:
        colors = np.array([
            [0.1216, 0.4667, 0.7059],  # 蓝
            [1.0000, 0.4980, 0.0549],  # 橙
            [0.1725, 0.6275, 0.1725],  # 绿
            [0.8392, 0.1529, 0.1569],  # 红
            [0.5804, 0.4039, 0.7412],  # 紫
            [0.5490, 0.3373, 0.2941],  # 棕
            [0.8902, 0.4667, 0.7608],  # 粉
            [0.4980, 0.4980, 0.4980],  # 灰
            [0.7373, 0.7412, 0.1333],  # 黄绿
            [0.0902, 0.7451, 0.8118],  # 青
        ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    # -------------------------
    # 左：原始点云（速度 / rear_prob 着色）
    # -------------------------
    ax0 = axes[0]

    # raw 点云着色：默认速度；如果提供 rear_prob，则按 rear_prob 上色
    if rear_prob is not None:
        rear_prob = np.asarray(rear_prob).reshape(-1)
        sc0 = ax0.scatter(
            pts[:, 0], pts[:, 1],
            c=rear_prob, s=point_size_raw,
            vmin=0.0, vmax=1.0
        )
        cbar0 = plt.colorbar(sc0, ax=ax0)
        cbar0.set_label("rear_prob (0~1)")

        # 可选：把高概率点圈出来
        mask_hp = rear_prob >= rear_prob_thr
        if np.any(mask_hp):
            ax0.scatter(
                pts[mask_hp, 0], pts[mask_hp, 1],
                s=point_size_raw * 4,
                facecolors="none", edgecolors="k", linewidths=1.2
            )
    else:
        sc0 = ax0.scatter(pts[:, 0], pts[:, 1], c=vel, s=point_size_raw)
        cbar0 = plt.colorbar(sc0, ax=ax0)
        cbar0.set_label("V (m/s)")

    # 标记SNR最大点（如果你用 star_indices）
    if star_indices:
        star_pts = pts[np.array(star_indices)]
        ax0.scatter(
            star_pts[:, 0], star_pts[:, 1],
            marker="*", s=18, facecolors="none",
            edgecolors="k", linewidths=1.5
        )

    ax0.set_xlim(*xlim)
    ax0.set_ylim(*ylim)
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Y (m)")
    ax0.set_title("Raw (colored by velocity)" if rear_prob is None else "Raw (colored by rear_prob)")
    ax0.grid(True, alpha=0.3)

    if gt_list is not None and len(gt_list) > 0:
        for gid, (xmin, ymin, w, h) in _gt_boxes_from_list(gt_list):
            rect_gt = Rectangle(
                (xmin, ymin), w, h,
                fill=False, linewidth=2.0, linestyle="--"   
            )
            ax0.add_patch(rect_gt)
            ax0.text(xmin, ymin + h + 0.5, f"GT:{gid}", fontsize=9)

    # -------------------------
    # 右：聚类结果
    # -------------------------
    ax1 = axes[1]
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Clusters (DBSCAN)")
    ax1.grid(True, alpha=0.3)

    # 1) 噪声点 (<=0)
    mask_noise = labels <= 0
    if np.any(mask_noise):
        ax1.scatter(
            pts[mask_noise, 0], pts[mask_noise, 1],
            s=point_size_raw, c=[0.7, 0.7, 0.7], label="Noise"
        )

    # 2) 各簇
    u_ids = np.unique(labels[labels > 0])
    for cid in u_ids:
        mask = labels == cid
        c_pts = pts[mask]
        c_vel = vel[mask]
        col = colors[(cid - 1) % len(colors)]

        # A. 绘制点
        ax1.scatter(
            c_pts[:, 0], c_pts[:, 1],
            s=point_size_cluster, c=[col],
            edgecolors="k", linewidths=0.5
        )

        # 标记SNR最大点（如果你用 star_indices）
        if star_indices:
            star_pts = pts[np.array(star_indices)]
            ax1.scatter(
                star_pts[:, 0], star_pts[:, 1],
                marker="*", s=18, facecolors="none",
                edgecolors="k", linewidths=1.5
            )

        # B. 包围框 (AABB)
        min_p = c_pts.min(axis=0)
        max_p = c_pts.max(axis=0)
        w = max_p[0] - min_p[0]
        h = max_p[1] - min_p[1]

        # 如果只有一个点，给一点宽高
        if w < 0.1:
            w = 0.5
            min_p[0] -= 0.25
        if h < 0.1:
            h = 0.5
            min_p[1] -= 0.25

        rect = Rectangle(
            (min_p[0], min_p[1]), w, h,
            fill=False, edgecolor=col, linewidth=1.5, linestyle="-"
        )
        ax1.add_patch(rect)

        # C. 文本标签 (ID + 平均速度)
        mean_v = float(np.mean(c_vel)) if c_vel.size else 0.0
        ax1.text(
            min_p[0], max_p[1] + 0.5,
            f"ID:{int(cid)}\nV:{mean_v:.1f}",
            color=col, fontsize=9, fontweight="bold"
        )

    # 左上角统计信息
    ax1.text(
        xlim[0] + 0.5, ylim[1] - 0.5,
        f"Clusters: {len(u_ids)}",
        color="k", bbox=dict(facecolor="white", edgecolor="k")
    )

    # 右图：叠加 rear_center（红色星标）
    if rear_centers is not None and len(rear_centers) > 0:
        rc = []
        for item in rear_centers:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                rc.append(np.asarray(item[1]).reshape(2))
            else:
                rc.append(np.asarray(item).reshape(2))
        rc = np.vstack(rc)
        ax1.scatter(
            rc[:, 0], rc[:, 1],
            marker="*", s=160, c="red",
            edgecolors="k", linewidths=0.6, label="rear_center"
        )

    if np.any(mask_noise):
        ax1.legend(loc="lower right")

    # 总标题
    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes

