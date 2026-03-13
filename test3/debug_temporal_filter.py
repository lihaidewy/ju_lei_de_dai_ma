import numpy as np
import matplotlib.pyplot as plt

class TemporalDebugTool:

    def __init__(self):
        self.frame_list = []
        self.raw_err = []
        self.filtered_err = []

    def update(self, fid, metrics_raw, metrics_filtered):
        """
        每帧调用一次
        """
        if metrics_raw is None or metrics_filtered is None:
            return

        self.frame_list.append(fid)
        self.raw_err.append(metrics_raw["mean_center_error"])
        self.filtered_err.append(metrics_filtered["mean_center_error"])

    def show(self):
        """
        程序结束后调用
        """
        if len(self.raw_err) == 0:
            print("TemporalDebugTool: no data.")
            return

        raw = np.array(self.raw_err)
        filt = np.array(self.filtered_err)

        print("\n===== Temporal Filtering Improvement =====")
        print("Mean raw error:", raw.mean())
        print("Mean filtered error:", filt.mean())
        print("Average improvement:", raw.mean() - filt.mean())

        plt.figure(figsize=(10,5))

        plt.plot(self.frame_list, raw, label="Raw Center Error", linewidth=2)
        plt.plot(self.frame_list, filt, label="Filtered Center Error", linewidth=2)

        plt.xlabel("Frame")
        plt.ylabel("Mean Center Error (m)")
        plt.title("Temporal Filtering Effect")

        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
