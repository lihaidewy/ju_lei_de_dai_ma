import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mylib.load_data2 import load_data
from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from mylib.plot_raw_and_clusters import plot_raw_and_clusters
from mylib.plot_gt_main import load_gt_reference
from mylib.eval_clusters2 import eval_one_frame_target_level

sum_TP = sum_FP = sum_FN = 0
all_center_err = []
model_tot = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}

# ===== 1) Load GT =====
gt_df = load_gt_reference("reference.csv", H=4.0)

# ===== 2) Load radar point cloud =====
path = "radar1.csv"
frame_data = load_data(path)

# ===== 3) Choose frames (use intersection to avoid missing keys) =====
radar_frames = sorted(frame_data.keys())
gt_frames = sorted(gt_df["Frame"].unique())

common_frames = sorted(set(radar_frames).intersection(gt_frames))
if len(common_frames) == 0:
    raise ValueError("No common frames between radar1.csv and reference.csv")

frame_ids = common_frames[:300]
# frame_ids = common_frames[150:160]

print("Will plot frames:", frame_ids)

# ===== 4) Loop: cluster + plot raw with GT boxes =====
for fid in frame_ids:
    # ---- Radar points ----
    x = frame_data[fid]["X"]
    y = frame_data[fid]["Y"]
    v = frame_data[fid]["V"]
    pts = np.column_stack([x, y])

    # ---- Clustering ----
    labels = cluster_frame_dbscan(frame_data, fid, eps_x=4.0, eps_y=1.5, eps_v=1.5, min_pts=2)

    # ---- GT list for this frame ----
    g = gt_df[gt_df["Frame"] == fid]
    gt_list = [
        {"id": int(r.ID), "x": float(r.X), "y": float(r.Y), "model": int(r.model)}
        for r in g.itertuples(index=False)
    ]
    
    m = eval_one_frame_target_level(
    pts_xy=pts,
    labels=labels,      # noise = -1, clusters >= 1
    gt_list=gt_list,
    dist_thr=4.0,
    iou_thr=0.10
                )
    print(f"[Frame {fid}] TP={m['TP']} FP={m['FP']} FN={m['FN']} "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
        f"mean_err={m['mean_center_error']:.2f}")

    sum_TP += m["TP"]
    sum_FP += m["FP"]
    sum_FN += m["FN"]
    all_center_err += m["center_errors"]
    for mm in [0, 1, 2]:
        for k in ["TP", "FP", "FN"]:
            model_tot[mm][k] += m["model_counts"][mm][k]

    # ---- Plot: RAW + GT boxes on LEFT, clusters on RIGHT ----
    plot_raw_and_clusters(
        pts, v, labels,
        xlim=(0, 400), ylim=(-40, 40),
        title=f"Frame {fid}",
        show=False,
        gt_list=gt_list
    )

    plt.show(block=False)
    plt.pause(0.001)

# =========================
# 5) Global summary
# =========================
print("\n" + "=" * 60)
print("GLOBAL SUMMARY (over selected frames)")
print("=" * 60)
print(f"Frames evaluated: {len(frame_ids)}")
print(f"Total: TP={sum_TP} FP={sum_FP} FN={sum_FN}")

# Overall precision/recall/f1 using accumulated counts
overall_P = sum_TP / (sum_TP + sum_FP) if (sum_TP + sum_FP) else 1.0
overall_R = sum_TP / (sum_TP + sum_FN) if (sum_TP + sum_FN) else 1.0
overall_F1 = (2 * overall_P * overall_R / (overall_P + overall_R)) if (overall_P + overall_R) else 0.0

print(f"Overall: P={overall_P:.4f} R={overall_R:.4f} F1={overall_F1:.4f}")

# Center error stats (only on TPs)
if len(all_center_err) > 0:
    ce = np.asarray(all_center_err, dtype=float)
    ce_mean = float(np.mean(ce))
    ce_median = float(np.median(ce))
    ce_p90 = float(np.percentile(ce, 90))
    ce_p95 = float(np.percentile(ce, 95))
    print(f"Center error (TP only): mean={ce_mean:.3f}  median={ce_median:.3f}  p90={ce_p90:.3f}  p95={ce_p95:.3f}")
else:
    print("Center error (TP only): n/a (no TP matches)")

# Per-model summary
print("\nPer-model (GT-side TP/FN, cluster-side FP assigned by nearest GT within fp_assign_dist):")
for mm in [0, 1, 2]:
    tp = model_tot[mm]["TP"]
    fp = model_tot[mm]["FP"]
    fn = model_tot[mm]["FN"]
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    print(f"  model={mm}: TP={tp} FP={fp} FN={fn} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

print("=" * 60 + "\n")


plt.show()


