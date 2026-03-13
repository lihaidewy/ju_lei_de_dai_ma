import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 把项目根目录加入搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from load_data2 import load_data
from plot_raw_and_clusters_updated import plot_raw_and_clusters, fit_center_fixed_yaw
from plot_gt_main import load_gt_reference
from eval_clusters2 import eval_one_frame_target_level

RADAR_PATH = "data\\radar.csv"
GT_PATH = "data\\reference3.csv"

EPS_X = 1.5
EPS_Y = 4.0
EPS_V = 1.5
MIN_PTS = 2
DIST_THR = 4.0
IOU_THR = 0.0
MAX_FRAMES_TO_VIEW = 1200

FRAMES_TO_SHOW = None
# FRAMES_TO_SHOW = [10, 11, 25, 80]

# =========================
# 统计量
# =========================
sum_TP = 0
sum_FP = 0
sum_FN = 0
all_center_err = []
model_tot = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}

# =========================
# 读取数据
# =========================
gt_df = load_gt_reference(GT_PATH, H=4.0)
frame_data = load_data(RADAR_PATH)

radar_frames = sorted(frame_data.keys())
gt_frames = sorted(gt_df["Frame"].unique())

common_frames = sorted(set(radar_frames).intersection(gt_frames))
if len(common_frames) == 0:
    raise ValueError(f"No common frames between {RADAR_PATH} and {GT_PATH}")

if FRAMES_TO_SHOW is None:
    frame_ids = common_frames[:MAX_FRAMES_TO_VIEW]
else:
    frame_ids = [fid for fid in FRAMES_TO_SHOW if fid in common_frames]
    if len(frame_ids) == 0:
        raise ValueError("None of FRAMES_TO_SHOW are in common_frames")

n_frames = len(frame_ids)
print("Will view frames:", frame_ids[:10], "..." if n_frames > 10 else "")

# =========================
# 预计算 & 缓存
# =========================
cache = {}

for fid in frame_ids:
    x = frame_data[fid]["X"]   # 横向
    y = frame_data[fid]["Y"]   # 前向
    v = frame_data[fid]["V"]
    pts = np.column_stack([x, y])

    labels = cluster_frame_dbscan(
        frame_data,
        fid,
        eps_x=EPS_X,
        eps_y=EPS_Y,
        eps_v=EPS_V,
        min_pts=MIN_PTS
    )

    g = gt_df[gt_df["Frame"] == fid]
    gt_list = [
        {"id": int(r.ID), "x": float(r.X), "y": float(r.Y), "model": int(r.model)}
        for r in g.itertuples(index=False)
    ]

    m = eval_one_frame_target_level(
        pts_xy=pts,
        labels=labels,          # noise = -1, clusters >= 1
        gt_list=gt_list,
        dist_thr=DIST_THR,
        iou_thr=IOU_THR,
        use_fixed_box=False,
        fixed_box_L=4.5,
        fixed_box_W=2.0,
    )

    # 累计全局统计
    sum_TP += m["TP"]
    sum_FP += m["FP"]
    sum_FN += m["FN"]
    all_center_err += m.get("center_errors", [])

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
# 图窗
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.08)

state = {"i": 0}

def render_frame(i: int):
    i = int(np.clip(i, 0, n_frames - 1))
    state["i"] = i

    fid = frame_ids[i]
    item = cache[fid]
    m = item["metrics"]

    mean_err = m["mean_center_error"]
    mean_err_str = f"{mean_err:.2f}" if np.isfinite(mean_err) else "nan"

    title = (
        f"Frame {fid} [{i+1}/{n_frames}] | "
        f"TP={m['TP']} FP={m['FP']} FN={m['FN']} "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
        f"mean_err={mean_err_str}"
    )

    plot_raw_and_clusters(
        item["pts"],
        item["v"],
        item["labels"],
        xlim=(-40, 40),   # X: 横向
        ylim=(0, 250),    # Y: 前向
        title=title,
        show=False,
        fig=fig,
        axes=axes,
        clear=True,
        use_fixed_box=False,
        gt_list=item["gt_list"]
    )

    # ------- 文字标注：TP/FP/FN -------
    ax_r = axes[0]  # raw
    ax_c = axes[1]  # clusters

    labels = item["labels"]
    pts = item["pts"]

    # cluster center for placing text
    cluster_centers = {}
    for cid in np.unique(labels):
        if cid < 1:
            continue

        cpts = pts[labels == cid]
        if cpts.size == 0:
            continue

        (cx_fit, cy_fit), _ = fit_center_fixed_yaw(
            cpts,
            L=4.5,
            W=2.0,
            yaw=0.0
        )
        cluster_centers[int(cid)] = (cx_fit, cy_fit)

    gt_pos = {int(g["id"]): (float(g["x"]), float(g["y"])) for g in item["gt_list"]}

    # 1) TP matches
    for (cid, gid, d, iou) in m.get("matches", []):
        if cid in cluster_centers:
            cx, cy = cluster_centers[cid]
            ax_c.text(cx, cy + 2.0, f"C{cid}→GT{gid}\nd={d:.1f}, IoU={iou:.2f}", fontsize=9)

    # 2) FP clusters
    for cid in m.get("unmatched_clusters", []):
        if cid in cluster_centers:
            cx, cy = cluster_centers[cid]
            ax_c.text(cx, cy + 2.0, f"C{cid}→FP", fontsize=9)

    # 3) FN gts
    for gid in m.get("unmatched_gts", []):
        if gid in gt_pos:
            gx, gy = gt_pos[gid]
            ax_r.text(gx, gy + 2.0, f"GT{gid}(FN)", fontsize=9)

    fig.canvas.draw_idle()

def on_key(event):
    k = (event.key or "").lower()
    if k == "n":
        render_frame(state["i"] + 1)
    elif k == "p":
        render_frame(state["i"] - 1)
    elif k in ("q", "escape"):
        plt.close(fig)

fig.canvas.mpl_connect("key_press_event", on_key)

# 初始显示
render_frame(0)

# =========================
# 汇总输出
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
