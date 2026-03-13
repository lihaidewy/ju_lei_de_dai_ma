import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_raw_and_clusters(
    pts, vel, labels,
    # 注意：现在显示为 “横轴=Y(向左), 纵轴=X(向上)”
    # 所以 xlim 是 Y 的范围，ylim 是 X 的范围
    xlim=(0, 400), ylim=(-60, 60),
    point_size_raw=10, point_size_cluster=15,
    colors=None, title=None, show=True, star_indices=None,
    fig=None, axes=None, clear=True
):
    """
    显示坐标系：
      - 纵轴向上：原 X
      - 横轴向左：原 Y（通过 invert_xaxis 实现）

    左：原始点云 (速度着色)
    右：聚类结果 (噪声灰色 + 簇点彩色 + 包围框 + ID/均速标签 + 统计信息)

    pts: (N,2)  [x,y]
    vel: (N,)   速度
    labels: (N,) 聚类标签，噪声<=0（如 -1 或 0）
    """

    pts = np.asarray(pts)
    vel = np.asarray(vel).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    # 绘图坐标：横轴用 Y，纵轴用 X
    plot_xy = np.column_stack([pts[:, 1], pts[:, 0]])  # [Y, X]

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

    # 复用或新建 figure/axes
    created_new = False
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        created_new = True

    ax0, ax1 = axes[0], axes[1]

    # 动画/复用时清空
    if clear:
        ax0.cla()
        ax1.cla()

    # -------------------------
    # 左：原始点云（速度着色）
    # -------------------------
    sc0 = ax0.scatter(plot_xy[:, 0], plot_xy[:, 1], c=vel, s=point_size_raw)

    if star_indices is not None and len(star_indices) > 0:
        star_xy = plot_xy[np.array(star_indices)]
        ax0.scatter(star_xy[:, 0], star_xy[:, 1],
                    marker="*", s=18, facecolors="none",
                    edgecolors="k", linewidths=1.5)

    # colorbar：只在新建fig时创建一次；复用时避免每帧叠加
    if created_new:
        cbar0 = plt.colorbar(sc0, ax=ax0)
        cbar0.set_label("V (m/s)")
    else:
        # 复用时简单做法：不再重复创建 colorbar（否则会越来越多）
        pass

    ax0.set_xlim(*xlim)   # 横轴：Y范围
    ax0.set_ylim(*ylim)   # 纵轴：X范围
    ax0.set_xlabel("Y (m)")
    ax0.set_ylabel("X (m)")
    ax0.set_title("Raw ")
    ax0.grid(True, alpha=0.3)

    # 关键：让“Y 增大方向朝左”
    ax0.invert_xaxis()

    # -------------------------
    # 右：聚类结果
    # -------------------------
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax1.set_xlabel("Y (m)")
    ax1.set_ylabel("X (m)")
    ax1.set_title("Clusters")
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # 1) 噪声点 (<=0)
    mask_noise = labels <= 0
    if np.any(mask_noise):
        ax1.scatter(plot_xy[mask_noise, 0], plot_xy[mask_noise, 1],
                    s=point_size_raw, c=[0.7, 0.7, 0.7], label="Noise")

    # 2) 各簇
    u_ids = np.unique(labels[labels > 0])
    for cid in u_ids:
        mask = labels == cid
        c_xy = plot_xy[mask]   # [Y, X]
        c_vel = vel[mask]

        col = colors[(int(cid) - 1) % len(colors)]

        # A. 绘制簇点
        ax1.scatter(c_xy[:, 0], c_xy[:, 1],
                    s=point_size_cluster, c=[col],
                    edgecolors="k", linewidths=0.5)

        # 标记星点
        if star_indices is not None and len(star_indices) > 0:
            star_xy = plot_xy[np.array(star_indices)]
            ax1.scatter(star_xy[:, 0], star_xy[:, 1],
                        marker="*", s=18, facecolors="none",
                        edgecolors="k", linewidths=1.5)

        # B. 包围框 (AABB) —— 在新坐标系 [Y,X] 下直接算
        min_p = c_xy.min(axis=0)
        max_p = c_xy.max(axis=0)
        w = max_p[0] - min_p[0]  # 宽：Y方向
        h = max_p[1] - min_p[1]  # 高：X方向

        if w < 0.1:
            w = 0.5
            min_p[0] -= 0.25
        if h < 0.1:
            h = 0.5
            min_p[1] -= 0.25

        rect = Rectangle((min_p[0], min_p[1]), w, h,
                         fill=False, edgecolor=col, linewidth=1.5, linestyle="-")
        ax1.add_patch(rect)

        # C. 文本标签 (ID + 平均速度)
        mean_v = float(np.mean(c_vel)) if c_vel.size else 0.0
        ax1.text(min_p[0], max_p[1] + 0.5,
                 f"ID:{int(cid)}\nV:{mean_v:.1f}",
                 color=col, fontsize=9, fontweight="bold")

    # 左上角统计信息（考虑 invert_xaxis 后 xlim 顺序可能反）
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    left = min(xmin, xmax)
    top = max(ymin, ymax)

    ax1.text(left + 0.5, top - 0.5,
             f"Clusters: {len(u_ids)}",
             color="k", bbox=dict(facecolor="white", edgecolor="k"))

    if np.any(mask_noise):
        ax1.legend(loc="lower right")

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes
