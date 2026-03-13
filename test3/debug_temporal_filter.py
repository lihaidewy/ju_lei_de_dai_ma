import numpy as np
import matplotlib.pyplot as plt


class _BaseTemporalDebugTool:
    def __init__(self):
        self.frame_list = []
        self.raw_err = []
        self.filtered_err = []

    def _extract_errors(self, metrics_raw, metrics_filtered):
        if metrics_raw is None or metrics_filtered is None:
            return None

        raw_err = metrics_raw.get("mean_center_error")
        filtered_err = metrics_filtered.get("mean_center_error")
        if raw_err is None or filtered_err is None:
            return None
        if not np.isfinite(raw_err) or not np.isfinite(filtered_err):
            return None
        return float(raw_err), float(filtered_err)

    def update(self, fid, metrics_raw, metrics_filtered):
        values = self._extract_errors(metrics_raw, metrics_filtered)
        if values is None:
            return

        raw_err, filtered_err = values
        self.frame_list.append(int(fid))
        self.raw_err.append(raw_err)
        self.filtered_err.append(filtered_err)

    def summary(self):
        if len(self.raw_err) == 0:
            return None

        raw = np.asarray(self.raw_err, dtype=float)
        filt = np.asarray(self.filtered_err, dtype=float)
        return {
            "mean_raw": float(np.mean(raw)),
            "mean_filtered": float(np.mean(filt)),
            "improvement": float(np.mean(raw) - np.mean(filt)),
        }


class TemporalDebugTool(_BaseTemporalDebugTool):
    def show(self):
        if len(self.raw_err) == 0:
            print("TemporalDebugTool: no data.")
            return None

        summary = self.summary()
        raw = np.asarray(self.raw_err, dtype=float)
        filt = np.asarray(self.filtered_err, dtype=float)

        print("\n===== Temporal Filtering Improvement =====")
        print("Mean raw error:", summary["mean_raw"])
        print("Mean filtered error:", summary["mean_filtered"])
        print("Average improvement:", summary["improvement"])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.frame_list, raw, label="Raw Center Error", linewidth=2)
        ax.plot(self.frame_list, filt, label="Filtered Center Error", linewidth=2)

        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean Center Error (m)")
        ax.set_title("Temporal Filtering Effect")
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        return fig, ax
