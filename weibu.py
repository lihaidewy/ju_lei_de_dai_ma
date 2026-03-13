import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mylib.load_data2 import load_data
from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from mylib.plot_raw_and_clusters import plot_raw_and_clusters
from mylib.wei import rear_prob_and_center_for_cluster

# 跟踪器
from mylib.mot_kf import MOTKF, Measurement


def main():
    path = "radar1.csv"
    frame_data = load_data(path)

    # 你要看的帧范围
    frame_ids = sorted(frame_data.keys())[100:105]

    # tracker 参数：dt 请按你真实帧率修改
    tracker = MOTKF(
        dt=0.1,            # 10Hz -> 0.1; 20Hz -> 0.05
        sigma_a=3.0,       # 过程噪声（加速度）
        sigma_z=1.0,       # 测量噪声（rear_center 抖动程度）
        w_v=1.0,           # 速度一致性权重
        M=3, N=5,          # M/N 确认逻辑
        max_missed=10,     # 丢失多少帧删掉
        min_birth_points=2 # 至少多少点才允许生轨
    )

    # --- 交互模式：动图刷新 ---
    plt.ion()

    # 图2：专门画轨迹
    fig_trk, ax_trk = plt.subplots(figsize=(10, 4))
    ax_trk.set_xlim(0, 400)
    ax_trk.set_ylim(-40, 40)
    ax_trk.set_aspect("equal", "box")
    ax_trk.set_title("Tracks (rear-center MOT)")

    for fid in frame_ids:
        # --------------- 读取当前帧点云 ---------------
        x = np.asarray(frame_data[fid]["X"])
        y = np.asarray(frame_data[fid]["Y"])
        v = np.asarray(frame_data[fid]["V"])
        snr = np.asarray(frame_data[fid]["SNR"])
        pts = np.column_stack([x, y])

        # --------------- 单帧聚类 ---------------
        labels = cluster_frame_dbscan(
            frame_data, fid,
            eps_x=4.0, eps_y=1.5, eps_v=1.5, min_pts=2
        )

        frame_df = pd.DataFrame({"X": x, "Y": y, "V": v, "SNR": snr, "Label": labels})

        # --------------- 计算 rear_prob + rear_center ---------------
        rear_centers = []                 # [(lab, rear_c(2,), info), ...]
        rear_prob = np.zeros(len(frame_df), dtype=float)

        for lab in sorted(frame_df["Label"].unique()):
            if lab == -1:
                continue

            g = frame_df[frame_df["Label"] == lab]

            P, rear_c, info = rear_prob_and_center_for_cluster(
                g["X"].values, g["Y"].values, g["V"].values, g["SNR"].values
            )

            rear_prob[g.index.values] = P

            # 给 tracker 的质量过滤准备一些统计
            info["n_points"] = int(len(g))
            info["snr_mean"] = float(np.mean(g["SNR"].values)) if len(g) > 0 else np.nan

            rear_centers.append((lab, np.asarray(rear_c, dtype=float), info))

        frame_df["rear_prob"] = rear_prob

        # --------------- 生成 measurements，喂给 tracker ---------------
        meas_list = []
        for (lab, rear_c, info) in rear_centers:
            meas_list.append(
                Measurement(
                    frame=int(fid),
                    z=np.array(rear_c, dtype=float),
                    v_median=float(info.get("v_median", np.nan)),
                    width=float(info.get("width", np.nan)),
                    n_points=int(info.get("n_points", 0)),
                    snr_mean=float(info.get("snr_mean", np.nan)),
                )
            )

        tracks = tracker.step(meas_list)

        # ============================================================
        # 图1：点云 + 聚类 + rear_prob + rear_center  (你自己的函数)
        # ============================================================
        plot_raw_and_clusters(
            pts, v, labels,
            xlim=(0, 400), ylim=(-40, 40),
            title=f"Frame {fid}",
            show=False,
            rear_prob=frame_df["rear_prob"].values,
            rear_centers=rear_centers,
            rear_prob_thr=0.6
        )

        # ============================================================
        # 图2：轨迹动图（单独窗口，每帧刷新）
        # ============================================================
        ax_trk.clear()
        ax_trk.set_xlim(0, 400)
        ax_trk.set_ylim(-40, 40)
        ax_trk.set_aspect("equal", "box")
        ax_trk.set_title(f"Tracks (Frame {fid})")

        # 画当前帧的观测点（rear_center）
        if len(rear_centers) > 0:
            obs = np.array([rc for (_, rc, _) in rear_centers], dtype=float)
            ax_trk.scatter(
                obs[:, 0], obs[:, 1],
                s=60, marker="o",
                facecolors="none", edgecolors="gray",
                linewidths=1.2,
                label="rear_center meas"
            )

        # 画所有 confirmed tracks：轨迹线 + 当前点 + ID
        for tr in tracks:
            px, py, vx, vy = tr.x

            if not tr.confirmed:
                continue

            # 轨迹线（最近 30 个点）
            if hasattr(tr, "trace") and len(tr.trace) >= 2:
                trace = np.array(tr.trace[-30:], dtype=float)
                ax_trk.plot(trace[:, 0], trace[:, 1], linewidth=1.8)

            # 当前点
            ax_trk.scatter([px], [py], s=90, marker="x")

            # ID 标注
            ax_trk.text(px, py, f"ID{tr.tid}", fontsize=10)

        ax_trk.legend(loc="upper right")

        # --- 刷新 ---
        plt.show(block=False)
        plt.pause(0.05)  # 0.03~0.1 都行，越大越慢

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
