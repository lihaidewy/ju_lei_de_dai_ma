import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from debug_temporal_filter import _BaseTemporalDebugTool


class TrackingDebugTool(_BaseTemporalDebugTool):
    def __init__(self):
        super().__init__()
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
        gt_list,
    ):
        values = self._extract_errors(metrics_raw, metrics_filtered)
        if values is None:
            return

        raw_err, filtered_err = values
        self.frame_list.append(int(fid))
        self.raw_err.append(raw_err)
        self.filtered_err.append(filtered_err)

        for cid, center in cluster_centers.items():
            if cid not in track_assignments:
                continue
            tid = int(track_assignments[cid])
            self.track_lengths[tid] += 1
            self.traj_x[tid].append(float(center[0]))
            self.traj_y[tid].append(float(center[1]))

        gt_map = {
            int(g["id"]): np.array([float(g["x"]), float(g["y"])], dtype=float)
            for g in gt_list
        }

        for match in metrics_filtered.get("matches", []):
            cid = int(match["cid"])
            gid = int(match["gid"])
            if cid not in cluster_centers or gid not in gt_map:
                continue

            pred = np.asarray(cluster_centers[cid], dtype=float)
            gt = gt_map[gid]
            err = float(np.linalg.norm(pred - gt))
            dist = float(np.linalg.norm(gt))
            self.err_vs_dist.append((dist, err))

    def show(self):
        if len(self.frame_list) == 0:
            print("No debug data")
            return None

        summary = self.summary()
        raw = np.asarray(self.raw_err, dtype=float)
        filt = np.asarray(self.filtered_err, dtype=float)

        print("\n===== Temporal Filtering Effect =====")
        print("Mean raw error:", summary["mean_raw"])
        print("Mean filtered error:", summary["mean_filtered"])
        print("Improvement:", summary["improvement"])

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(self.frame_list, raw, label="Raw", linewidth=2)
        axs[0, 0].plot(self.frame_list, filt, label="Filtered", linewidth=2)
        axs[0, 0].set_title("Center Error vs Frame")
        axs[0, 0].set_xlabel("Frame")
        axs[0, 0].set_ylabel("Error (m)")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        lengths = list(self.track_lengths.values())
        if len(lengths) > 0:
            axs[0, 1].hist(lengths, bins=20)
        axs[0, 1].set_title("Track Length Histogram")
        axs[0, 1].set_xlabel("Track Length (frames)")
        axs[0, 1].set_ylabel("Count")

        for tid in self.traj_x:
            axs[1, 0].plot(self.traj_x[tid], self.traj_y[tid], linewidth=1)
        axs[1, 0].set_title("Track Trajectories")
        axs[1, 0].set_xlabel("X")
        axs[1, 0].set_ylabel("Y")
        axs[1, 0].grid(True)

        if len(self.err_vs_dist) > 0:
            dist = np.asarray([d for d, _ in self.err_vs_dist], dtype=float)
            err = np.asarray([e for _, e in self.err_vs_dist], dtype=float)
            axs[1, 1].scatter(dist, err, s=5)
            axs[1, 1].set_title("Error vs Distance")
            axs[1, 1].set_xlabel("Distance")
            axs[1, 1].set_ylabel("Error")
            axs[1, 1].grid(True)

        fig.tight_layout()
        return fig, axs
