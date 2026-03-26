import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.widgets import Slider, Button

from mylib.load_data2 import load_data
from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from mylib.plot_raw_and_clusters import plot_raw_and_clusters
from mylib.plot_gt_main import load_gt_reference
from mylib.eval_clusters2 import eval_one_frame_target_level

# =========================
# 0) Accumulators
# =========================
sum_TP = sum_FP = sum_FN = 0
all_center_err = []
model_tot = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}

# =========================
# 1) Load GT
# =========================
gt_df = load_gt_reference("reference.csv", H=4.0)

# =========================
# 2) Load radar point cloud
# =========================
path = "radar1.csv"
frame_data = load_data(path)

# =========================
# 3) Choose frames (intersection)
# =========================
radar_frames = sorted(frame_data.keys())
gt_frames = sorted(gt_df["Frame"].unique())
common_frames = sorted(set(radar_frames).intersection(gt_frames))
if len(common_frames) == 0:
    raise ValueError("No common frames between radar1.csv and reference.csv")

frame_ids = common_frames[:300]
print("Frames to evaluate:", len(frame_ids))

# =========================
# 4) Pre-compute: cluster + eval (NO plotting), show progress bar
# =========================
# 缓存每帧的 labels / gt_list / metrics，为了拖动时能瞬间显示
cache = {}  

for fid in tqdm(frame_ids, desc="Processing frames"):
    x = frame_data[fid]["X"]
    y = frame_data[fid]["Y"]
    v = frame_data[fid]["V"]
    pts = np.column_stack([x, y])

    labels = cluster_frame_dbscan(frame_data, fid, eps_x=4.0, eps_y=1.5, eps_v=1.5, min_pts=2)

    g = gt_df[gt_df["Frame"] == fid]
    gt_list = [
        {"id": int(r.ID), "x": float(r.X), "y": float(r.Y), "model": int(r.model)}
        for r in g.itertuples(index=False)
    ]

    m = eval_one_frame_target_level(
        pts_xy=pts,
        labels=labels,
        gt_list=gt_list,
        dist_thr=4.0,
        iou_thr=0.10
    )

    cache[fid] = {"labels": labels, "gt_list": gt_list, "metrics": m}

    # --- global summary accumulate ---
    sum_TP += m["TP"]
    sum_FP += m["FP"]
    sum_FN += m["FN"]
    all_center_err += m["center_errors"]
    for mm in [0, 1, 2]:
        for k in ["TP", "FP", "FN"]:
            model_tot[mm][k] += m["model_counts"][mm][k]

# =========================
# 5) Print global summary once
# =========================
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
    print(
        f"Center error (TP only): "
        f"mean={np.mean(ce):.3f}  median={np.median(ce):.3f}  "
        f"p90={np.percentile(ce, 90):.3f}  p95={np.percentile(ce, 95):.3f}"
    )
else:
    print("Center error (TP only): n/a (no TP matches)")

print("\nPer-model:")
for mm in [0, 1, 2]:
    tp = model_tot[mm]["TP"]
    fp = model_tot[mm]["FP"]
    fn = model_tot[mm]["FN"]
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    print(f"  model={mm}: TP={tp} FP={fp} FN={fn} | P={p:.4f} R={r:.4f} F1={f1:.4f}")
print("=" * 60 + "\n")


# =========================
# 6) Interactive viewer: ONE window + Slider to scrub frames
# =========================
# 创建全局唯一画布
fig = plt.figure(figsize=(12, 7))

# 预先划分好区域 [left, bottom, width, height]
ax_main = plt.axes([0.08, 0.25, 0.85, 0.65])   # 主绘图区
ax_slider = plt.axes([0.15, 0.1, 0.65, 0.04])  # 进度条区
ax_prev = plt.axes([0.82, 0.09, 0.06, 0.06])   # 上一帧按钮
ax_next = plt.axes([0.89, 0.09, 0.06, 0.06])   # 下一帧按钮

# 指标文本（固定在左上角外侧）
info_text = fig.text(0.08, 0.92, "", va="top", ha="left", fontsize=11, fontweight='bold')

# 初始化交互控件
slider = Slider(
    ax=ax_slider,
    label="Frame index",
    valmin=0,
    valmax=len(frame_ids) - 1,
    valinit=0,
    valstep=1
)
btn_prev = Button(ax_prev, "Prev")
btn_next = Button(ax_next, "Next")

def render_frame(idx: int):
    fid = frame_ids[int(idx)]
    x = frame_data[fid]["X"]
    y = frame_data[fid]["Y"]
    v = frame_data[fid]["V"]
    pts = np.column_stack([x, y])

    labels = cache[fid]["labels"]
    gt_list = cache[fid]["gt_list"]
    m = cache[fid]["metrics"]

    # 1. 仅清空主绘图区的内容，保留坐标轴和控件，彻底解决闪烁问题
    ax_main.clear()

    # 2. 将当前活动坐标轴设置为 ax_main，确保你的绘图函数画在这里
    plt.sca(ax_main)

    # 3. 调用你的绘图函数
    plot_raw_and_clusters(
        pts, v, labels,
        xlim=(0, 400), ylim=(-40, 40),
        title=f"Frame {fid}",
        show=False,
        gt_list=gt_list
    )

    # 4. 更新文本信息
    info_text.set_text(
        f"[Frame {fid}] TP={m['TP']} FP={m['FP']} FN={m['FN']}  |  "
        f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  |  "
        f"mean_err={m['mean_center_error']:.2f}"
    )
    
    # 5. 通知 Matplotlib 刷新画布
    fig.canvas.draw_idle()

def on_slider_change(val):
    render_frame(int(val))

def on_prev(event):
    new_val = max(0, int(slider.val) - 1)
    slider.set_val(new_val)

def on_next(event):
    new_val = min(len(frame_ids) - 1, int(slider.val) + 1)
    slider.set_val(new_val)

# 绑定事件
slider.on_changed(on_slider_change)
btn_prev.on_clicked(on_prev)
btn_next.on_clicked(on_next)

# 初次渲染第 0 帧
render_frame(0)

# 启动界面，拖动进度条即可丝滑查看每一帧
plt.show()
