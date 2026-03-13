import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

from mylib.load_data2 import load_data
from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from mylib.plot_raw_and_clusters_updated import plot_raw_and_clusters
from mylib.plot_gt_main import load_gt_reference
from mylib.eval_clusters2 import eval_one_frame_target_level

RADAR_PATH = "data\\1000\\radar.csv"
GT_PATH = "data\\1000\\reference.csv"

EPS_X = 4.0
EPS_Y = 1.5
EPS_V = 1.5
MIN_PTS = 2
DIST_THR = 4.0
# DIST_THR = 8.0
IOU_THR = 0.0
MAX_FRAMES_TO_VIEW = 300

sum_TP = sum_FP = sum_FN = 0
all_center_err = []
model_tot = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}

gt_df = load_gt_reference(GT_PATH, H=4.0)
frame_data = load_data(RADAR_PATH)

radar_frames = sorted(frame_data.keys())
gt_frames = sorted(gt_df["Frame"].unique())

common_frames = sorted(set(radar_frames).intersection(gt_frames))
if len(common_frames) == 0:
    raise ValueError(f"No common frames between {RADAR_PATH} and {GT_PATH}")

frame_ids = common_frames[:MAX_FRAMES_TO_VIEW]
n_frames = len(frame_ids)
print("Will view frames:", frame_ids[:10], "..." if n_frames > 10 else "")

cache = {}
for fid in frame_ids:
    x = frame_data[fid]["X"]
    y = frame_data[fid]["Y"]
    v = frame_data[fid]["V"]
    pts = np.column_stack([x, y])

    labels = cluster_frame_dbscan(
        frame_data, fid,
        eps_x=EPS_X, eps_y=EPS_Y, eps_v=EPS_V, min_pts=MIN_PTS
    )

    g = gt_df[gt_df["Frame"] == fid]
    gt_list = [
        {"id": int(r.ID), "x": float(r.X), "y": float(r.Y), "model": int(r.model)}
        for r in g.itertuples(index=False)
    ]

    m = eval_one_frame_target_level(
        pts_xy=pts,
        labels=labels,      # noise = -1, clusters >= 1
        gt_list=gt_list,
        dist_thr=DIST_THR,
        iou_thr=IOU_THR
    )

    # 累计全局统计
    sum_TP += m["TP"]
    sum_FP += m["FP"]
    sum_FN += m["FN"]
    all_center_err += m["center_errors"]
    for mm in [0, 1, 2]:
        for k in ["TP", "FP", "FN"]:
            model_tot[mm][k] += m["model_counts"][mm][k]

    cache[fid] = {
        "pts": pts,
        "v": v,
        "labels": labels,
        "gt_list": gt_list,
        "metrics": m
    }
# =========================
# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.20)
state = {"playing": True, "i": 0, "lock": False}
def update(i):
    if state["playing"]:
        state["i"] = int(i)
    else:
        i = int(state["i"])
    fid = frame_ids[i]
    item = cache[fid]
    m = item["metrics"]
    title = (
        f"Frame {fid} | TP={m['TP']} FP={m['FP']} FN={m['FN']} "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
        f"mean_err={m['mean_center_error']:.2f}"
    )
    plot_raw_and_clusters(
        item["pts"], item["v"], item["labels"],
        xlim=(0, 400), ylim=(-40, 40),
        title=title,
        show=False,
        fig=fig, axes=axes, clear=True,
        gt_list=item["gt_list"]
    )
    # -----------------新增-----------------
    ax_c = axes[1]   # 右图：clusters
    ax_r = axes[0]   # 左图：raw

    # 计算每个 cluster 的中心（用于放置文本）
    labels = item["labels"]
    pts = item["pts"]
    cluster_centers = {}
    for cid in np.unique(labels):
        if cid < 1:
            continue
        cpts = pts[labels == cid]
        if cpts.size == 0:
            continue
        cluster_centers[int(cid)] = (float(cpts[:, 0].mean()), float(cpts[:, 1].mean()))

    # 建一个 gid -> (x,y) 方便标 FN
    gt_pos = {int(g["id"]): (float(g["x"]), float(g["y"])) for g in item["gt_list"]}

    # 1) TP：matches 里的是最终 TP
    for (cid, gid, d, iou) in m.get("matches", []):
        if cid not in cluster_centers:
            continue
        cx, cy = cluster_centers[cid]
        ax_c.text(cx, cy + 2.0, f"C{cid}→GT{gid}\nd={d:.1f}, IoU={iou:.2f}", fontsize=9)

    # 2) FP：unmatched_clusters
    for cid in m.get("unmatched_clusters", []):
        if cid not in cluster_centers:
            continue
        cx, cy = cluster_centers[cid]
        ax_c.text(cx, cy + 2.0, f"C{cid}→FP", fontsize=9)

    # 3) FN：unmatched_gts（标在 GT 位置）
    for gid in m.get("unmatched_gts", []):
        if gid not in gt_pos:
            continue
        gx, gy = gt_pos[gid]
        ax_r.text(gx, gy + 2.0, f"GT{gid}(FN)", fontsize=9)
# --------------------------------------------

    if state["playing"]:
        state["lock"] = True
        frame_slider.set_val(i)
        state["lock"] = False

    return []
ani = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False, repeat=True)
# Slider
ax_slider = plt.axes([0.20, 0.00, 0.55, 0.02])  # [left, bottom, width, height]
frame_slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)
def on_slider(val):
    if state["lock"]:
        return
    state["i"] = int(val)
    update(state["i"])
    fig.canvas.draw_idle()
frame_slider.on_changed(on_slider)
# Play/Pause Button
ax_btn = plt.axes([0.78, 0.0, 0.12, 0.055])
btn = Button(ax_btn, "Pause")  # 初始 playing=True
def on_btn(event):
    state["playing"] = not state["playing"]
    btn.label.set_text("Pause" if state["playing"] else "Play")
    fig.canvas.draw_idle()
btn.on_clicked(on_btn)


# =========================
# 汇总输出
print("\n" + "=" * 60)
print("GLOBAL SUMMARY (over selected frames)")
print("=" * 60)
print(f"Frames evaluated: {len(frame_ids)}")
print(f"Total: TP={sum_TP} FP={sum_FP} FN={sum_FN}")

overall_P = sum_TP / (sum_TP + sum_FP) if (sum_TP + sum_FP) else 1.0
overall_R = sum_TP / (sum_TP + sum_FN) if (sum_TP + sum_FN) else 1.0
overall_F1 = (2 * overall_P * overall_R / (overall_P + overall_R)) if (overall_P + overall_R) else 0.0
print(f"Overall: P={overall_P:.4f} R={overall_R:.4f} F1={overall_F1:.4f}")

if len(all_center_err) > 0:
    ce = np.asarray(all_center_err, dtype=float)
    ce_mean = float(np.mean(ce))
    ce_median = float(np.median(ce))
    ce_p90 = float(np.percentile(ce, 90))
    ce_p95 = float(np.percentile(ce, 95))
    print(f"Center error (TP only): mean={ce_mean:.3f}  median={ce_median:.3f}  p90={ce_p90:.3f}  p95={ce_p95:.3f}")
else:
    print("Center error (TP only): n/a (no TP matches)")

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