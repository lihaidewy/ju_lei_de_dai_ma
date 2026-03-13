import numpy as np
import matplotlib.pyplot as plt

from mylib.load_data2 import load_data
from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from mylib.plot_raw_and_clusters_multi_prior import plot_raw_and_clusters, choose_best_fixed_box_prior
from mylib.plot_gt_main import load_gt_reference
from mylib.eval_clusters2_multi_prior import eval_one_frame_target_level, GT_DIM

RADAR_PATH = "radar1.csv"
GT_PATH = "reference.csv"

EPS_X = 4.0
EPS_Y = 1.5
EPS_V = 1.5
MIN_PTS = 2

DIST_THR = 6.0
IOU_THR = 0.0

MAX_FRAMES_TO_VIEW = 300

FRAMES_TO_SHOW = None
# FRAMES_TO_SHOW = [10, 11, 25, 80]

# =========================
# 多先验（3类车型尺寸）
FIXED_BOX_PRIORS = [(GT_DIM[m]["L"], GT_DIM[m]["W"]) for m in sorted(GT_DIM.keys())]
FIXED_BOX_YAW = 0.0
FIXED_BOX_SCORE_LAMBDA = 1.0

# =========================
# 统计量
sum_TP = sum_FP = sum_FN = 0
all_center_err = []
model_tot = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}

# =========================
# 读取数据
gt_df = load_gt_reference(GT_PATH, H=6.0)
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
        labels=labels,      # noise=-1, clusters>=1（与你现有pipeline一致）
        gt_list=gt_list,
        dist_thr=DIST_THR,
        iou_thr=IOU_THR,
        use_fixed_box=True,
        fixed_box_priors=FIXED_BOX_PRIORS,
        fixed_box_yaw=FIXED_BOX_YAW,
        fixed_box_score_lambda=FIXED_BOX_SCORE_LAMBDA,
    )

    # 累计全局统计
    sum_TP += int(m["TP"])
    sum_FP += int(m["FP"])
    sum_FN += int(m["FN"])
    all_center_err += list(m.get("center_errors", []))
    for mm in [0, 1, 2]:
        for k in ["TP", "FP", "FN"]:
            model_tot[mm][k] += int(m["model_counts"][mm][k])

    cache[fid] = {
        "pts": pts,
        "v": v,
        "labels": labels,
        "gt_list": gt_list,
        "metrics": m
    }

# =========================
# 交互显示
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.08)

state = {"i": 0}

def render_frame(i: int):
    i = int(np.clip(i, 0, n_frames - 1))
    state["i"] = i

    fid = frame_ids[i]
    item = cache[fid]
    m = item["metrics"]

    title = (
        f"Frame {fid} [{i+1}/{n_frames}] | "
        f"TP={m['TP']} FP={m['FP']} FN={m['FN']} "
        f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
        f"mean_err={m['mean_center_error']:.2f}"
    )

    plot_raw_and_clusters(
    pts_xy=item["pts"],
    labels=item["labels"],
    v=item["v"],
    gt_list=item["gt_list"],
    fig=fig,
    axes=axes,
    title=title,
    use_fixed_box=True,
    fixed_box_priors=FIXED_BOX_PRIORS,
    fixed_box_yaw=FIXED_BOX_YAW,
    fixed_box_score_lambda=FIXED_BOX_SCORE_LAMBDA,
    )

    # 手动设置视野范围（两张子图都设）
    for ax in axes:
        ax.set_xlim(0, 400)
        ax.set_ylim(-40, 40)


    # ------- 文字标注：TP/FP/FN -------
    ax_r = axes[0]  # raw
    ax_c = axes[1]  # clusters

    labels = item["labels"]
    pts = item["pts"]

    # cluster center（必须与 fixed-box 选择一致）
    cluster_centers = {}
    for cid in np.unique(labels):
        if cid < 1:
            continue
        cpts = pts[labels == cid]
        if cpts.size == 0:
            continue

        best = choose_best_fixed_box_prior(
            cpts,
            priors=FIXED_BOX_PRIORS,
            yaw=FIXED_BOX_YAW,
            score_lambda=FIXED_BOX_SCORE_LAMBDA,
        )
        cx_fit, cy_fit = best["center"]
        cluster_centers[int(cid)] = (float(cx_fit), float(cy_fit))

    gt_pos = {int(g["id"]): (float(g["x"]), float(g["y"])) for g in item["gt_list"]}

    # 1) TP matches（新版 eval 返回 list[dict]）
    for mmatch in m.get("matches", []):
        cid = int(mmatch["cid"])
        gid = int(mmatch["gid"])
        d = float(mmatch["center_dist"])
        iou = float(mmatch["iou"])

        if cid in cluster_centers:
            cx, cy = cluster_centers[cid]
            # ax_c.text(cx, cy + 2.0, f"C{cid}→GT{gid}\nd={d:.1f}, IoU={iou:.2f}", fontsize=9)
            ax_c.text(cx, cy + 2.0, f"C{cid}→GT{gid}\nd={d:.1f}", fontsize=9)

    # 2) FP clusters
    for cid in m.get("unmatched_clusters", []):
        cid = int(cid)
        if cid in cluster_centers:
            cx, cy = cluster_centers[cid]
            ax_c.text(cx, cy + 2.0, f"C{cid}→FP", fontsize=9)

    # 3) FN gts
    for gid in m.get("unmatched_gts", []):
        gid = int(gid)
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
