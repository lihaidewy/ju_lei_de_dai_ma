import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class TrackingDebugTool:

    def __init__(self):
        self.frames = []
        self.raw_err = []
        self.filtered_err = []

        self.track_lengths = defaultdict(int)
        self.traj_x = defaultdict(list)
        self.traj_y = defaultdict(list)

        self.err_vs_dist = []

    def update(
        self,
        fid,
        metrics_raw,
        metrics_filtered,
        cluster_centers,
        track_assignments,
        gt_list
    ):
        if metrics_raw is None or metrics_filtered is None:
            return

        self.frames.append(fid)
        self.raw_err.append(metrics_raw["mean_center_error"])
        self.filtered_err.append(metrics_filtered["mean_center_error"])

        # 记录每条 track 的长度和轨迹
        for cid, center in cluster_centers.items():
            if cid in track_assignments:
                tid = track_assignments[cid]
                self.track_lengths[tid] += 1
                self.traj_x[tid].append(float(center[0]))
                self.traj_y[tid].append(float(center[1]))

        # 用 cluster_centers + gt_list 自己构造误差
        gt_map = {
            int(g["id"]): np.array([float(g["x"]), float(g["y"])])
            for g in gt_list
        }

        for m in metrics_filtered.get("matches", []):
            cid = int(m["cid"])
            gid = int(m["gid"])

            if cid not in cluster_centers:
                continue
            if gid not in gt_map:
                continue

            pred = np.array(cluster_centers[cid], dtype=float)
            gt = gt_map[gid]

            err = np.linalg.norm(pred - gt)
            dist = np.linalg.norm(gt)

            self.err_vs_dist.append((dist, err))

    def show(self):
        if len(self.frames) == 0:
            print("No debug data")
            return

        raw = np.array(self.raw_err)
        filt = np.array(self.filtered_err)

        print("\n===== Temporal Filtering Effect =====")
        print("Mean raw error:", raw.mean())
        print("Mean filtered error:", filt.mean())
        print("Improvement:", raw.mean() - filt.mean())

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # 1) center error curve
        axs[0, 0].plot(self.frames, raw, label="Raw", linewidth=2)
        axs[0, 0].plot(self.frames, filt, label="Filtered", linewidth=2)
        axs[0, 0].set_title("Center Error vs Frame")
        axs[0, 0].set_xlabel("Frame")
        axs[0, 0].set_ylabel("Error (m)")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # 2) track length histogram
        lengths = list(self.track_lengths.values())
        if len(lengths) > 0:
            axs[0, 1].hist(lengths, bins=20)
        axs[0, 1].set_title("Track Length Histogram")
        axs[0, 1].set_xlabel("Track Length (frames)")
        axs[0, 1].set_ylabel("Count")

        # 3) trajectories
        for tid in self.traj_x:
            axs[1, 0].plot(
                self.traj_x[tid],
                self.traj_y[tid],
                linewidth=1
            )
        axs[1, 0].set_title("Track Trajectories")
        axs[1, 0].set_xlabel("X")
        axs[1, 0].set_ylabel("Y")
        axs[1, 0].grid(True)

        # 4) error vs distance
        if len(self.err_vs_dist) > 0:
            dist = np.array([d for d, e in self.err_vs_dist])
            err = np.array([e for d, e in self.err_vs_dist])

            axs[1, 1].scatter(dist, err, s=5)
            axs[1, 1].set_title("Error vs Distance")
            axs[1, 1].set_xlabel("Distance")
            axs[1, 1].set_ylabel("Error")
            axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
