import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mylib.load_data2 import load_data
from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from mylib.wei import rear_prob_and_center_for_cluster

from mylib.mot_kf import MOTKF, Measurement


def main():
    path = "radar1.csv"
    frame_data = load_data(path)

    # 选择帧范围
    frame_ids = sorted(frame_data.keys())[1:200]

    # 创建 tracker（dt 按真实帧率改）
    tracker = MOTKF(
        dt=0.1,            
        sigma_a=3.0,
        sigma_z=1.0,
        w_v=1.0,
        M=3, N=5,
        max_missed=10,
        min_birth_points=2
    )

    # ===== 动图显示：只显示轨迹 =====
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 400)
    ax.set_ylim(-40, 40)
    ax.set_aspect("equal", "box")

    for fid in frame_ids:
        # 读取当前帧
        x = np.asarray(frame_data[fid]["X"])
        y = np.asarray(frame_data[fid]["Y"])
        v = np.asarray(frame_data[fid]["V"])
        snr = np.asarray(frame_data[fid]["SNR"])

        # 聚类（你仍然需要它来获得每个 cluster，再算 rear-center）
        labels = cluster_frame_dbscan(
            frame_data, fid,
            eps_x=4.0, eps_y=1.5, eps_v=1.5, min_pts=2
        )

        frame_df = pd.DataFrame({"X": x, "Y": y, "V": v, "SNR": snr, "Label": labels})

        # ===== 计算每个 cluster 的 rear_center
        rear_centers = []  # [(lab, rear_c(2,), info), ...]
        for lab in np.unique(labels):
            if lab == -1:
                continue

            g = frame_df[frame_df["Label"] == lab]
            if len(g) == 0:
                continue

            P, rear_c, info = rear_prob_and_center_for_cluster(
                g["X"].values, g["Y"].values, g["V"].values, g["SNR"].values
            )

            # 给 tracker 用的质量字段
            info["n_points"] = int(len(g))
            info["snr_mean"] = float(np.mean(g["SNR"].values)) if len(g) > 0 else np.nan

            rear_centers.append((lab, np.asarray(rear_c, dtype=float), info))

        # ===== 组装 measurements =====
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

        # ===== 跟踪更新 =====
        tracks = tracker.step(meas_list)
        vehicles = tracker.get_confirmed_vehicles(conf_thr=0.5)
        for v in vehicles:
            print(v["id"], v["x"], v["y"], "TAIL" if v["tail_visible"] else "HEAD", v["confidence"])

        # ===== 动图刷新：只画轨迹 + 当前点 + ID（不画聚类点云）=====
        ax.clear()
        ax.set_xlim(0, 400)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Tracks (Frame {fid})")

        # 可选：画当前帧观测点（rear_center），便于看关联；不需要就注释掉
        if len(rear_centers) > 0:
            obs = np.array([rc for (_, rc, _) in rear_centers], dtype=float)
            ax.scatter(obs[:, 0], obs[:, 1],
                       s=50, marker="o",
                       facecolors="none", edgecolors="gray",
                       linewidths=1.2, label="meas")

        # 画 confirmed tracks（更干净）；想全画就把 if 去掉
        for tr in tracks:
            if not tr.confirmed:
                continue

            px, py, vx, vy = tr.x

            # 轨迹线（最近 N 点）
            if hasattr(tr, "trace") and len(tr.trace) >= 2:
                trace = np.array(tr.trace[-40:], dtype=float)  # 最近40点
                ax.plot(trace[:, 0], trace[:, 1], linewidth=1.8)

            # 当前点（track state）
            ax.scatter([px], [py], s=90, marker="x")

            # ID
            ax.text(px, py, f"ID{tr.tid}", fontsize=10)

        ax.legend(loc="upper right")

        plt.show(block=False)
        plt.pause(0.05)  # 越大越慢，0.03~0.1都可以

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
