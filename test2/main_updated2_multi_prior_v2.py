import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 把项目根目录加入搜索路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

from load_data2 import load_data
from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from plot_raw_and_clusters_multi_prior_v2 import (
    plot_raw_and_clusters,
    choose_best_fixed_box_prior_mode,
)
from plot_gt_main import load_gt_reference
from eval_clusters2_multi_prior_v2 import eval_one_frame_target_level, GT_DIM


RADAR_PATH = "data\\radar.csv"
GT_PATH = "data\\reference3.csv"

# 当前坐标系：
# X = 横向
# Y = 前向
#
# 所以 DBSCAN 参数建议理解为：
# eps_x -> 横向容差
# eps_y -> 前向容差
EPS_X = 1.5
EPS_Y = 4.0
EPS_V = 1.5
MIN_PTS = 2

DIST_THR = 6.0
IOU_THR = 0.0

MAX_FRAMES_TO_VIEW = 1200
FRAMES_TO_SHOW = None

# =========================
# 多先验（3类车型尺寸）
# =========================
FIXED_BOX_PRIORS = [(GT_DIM[m]["L"], GT_DIM[m]["W"]) for m in sorted(GT_DIM.keys())]
FIXED_BOX_YAW = 0.0
FIXED_BOX_SCORE_LAMBDA = 1.0

# edge-loss extra
FIXED_BOX_INSIDE_MARGIN = 0.2
FIXED_BOX_ALPHA_OUT = 10.0
FIXED_BOX_BETA_IN = 2.0

CENTER_BIAS_X = 0.0
CENTER_BIAS_Y = 0.0


def get_point_level_cluster_centers(pts: np.ndarray, labels: np.ndarray):
    """
    给每个点分配其所属 cluster 的中心：
    center = median(points in cluster) + two-segment Y bias correction

    返回：
    - center_x_per_point
    - center_y_per_point
    """
    center_x_per_point = np.full(len(labels), np.nan, dtype=float)
    center_y_per_point = np.full(len(labels), np.nan, dtype=float)

    for cid in np.unique(labels):
        if cid < 1:
            continue

        mask = (labels == cid)
        cpts = pts[mask]
        if cpts.size == 0:
            continue

       
        # center = np.median(cpts, axis=0)
        center = cpts.mean(axis=0)


        # 两段 Y bias correction（与你当前评估保持一致）
        if center[1] < 100.0:
            bias_y = 1.149
        else:
            bias_y = 1.586

        center = center + np.array([0.0, bias_y], dtype=float)

        center_x_per_point[mask] = float(center[0])
        center_y_per_point[mask] = float(center[1])

    return center_x_per_point, center_y_per_point


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fit_mode",
        type=str,
        default="center",
        choices=["center", "edge"],
        help="fixed-box fit mode: center (pull-to-center) or edge (boundary-seeking)",
    )
    p.add_argument("--max_frames", type=int, default=MAX_FRAMES_TO_VIEW)
    return p.parse_args()


def main():
    args = parse_args()
    fixed_box_fit_mode = args.fit_mode

    # =========================
    # 统计量
    # =========================
    sum_TP = 0
    sum_FP = 0
    sum_FN = 0
    all_center_err = []
    all_dx_err = []
    all_dy_err = []
    labeled_rows = []

    range_bins = [(0.0, 100.0), (100.0, 1e9)]
    range_bias_stats = {rb: [] for rb in range_bins}

    model_tot = {m: {"TP": 0, "FP": 0, "FN": 0} for m in [0, 1, 2]}

    # =========================
    # 读取数据
    # =========================
    gt_df = load_gt_reference(GT_PATH, H=6.0)
    frame_data = load_data(RADAR_PATH)

    radar_frames = sorted(frame_data.keys())
    gt_frames = sorted(gt_df["Frame"].unique())

    common_frames = sorted(set(radar_frames).intersection(gt_frames))
    if len(common_frames) == 0:
        raise ValueError(f"No common frames between {RADAR_PATH} and {GT_PATH}")

    if FRAMES_TO_SHOW is None:
        frame_ids = common_frames[:args.max_frames]
    else:
        frame_ids = [fid for fid in FRAMES_TO_SHOW if fid in common_frames]
        if len(frame_ids) == 0:
            raise ValueError("None of FRAMES_TO_SHOW are in common_frames")

    n_frames = len(frame_ids)
    print("Will view frames:", frame_ids[:10], "..." if n_frames > 10 else "")
    print("FIT MODE:", fixed_box_fit_mode)

    # =========================
    # 预计算 & 缓存
    # =========================
    cache = {}

    for fid in frame_ids:
        x = frame_data[fid]["X"]   # 横向
        y = frame_data[fid]["Y"]   # 前向
        v = frame_data[fid]["V"]
        snr = frame_data[fid]["SNR"]

        pts = np.column_stack([x, y])

        labels = cluster_frame_dbscan(
            frame_data,
            fid,
            eps_x=EPS_X,
            eps_y=EPS_Y,
            eps_v=EPS_V,
            min_pts=MIN_PTS,
        )

        # =========================
        # 点集表：每个点附上 Label 和所属中心
        # =========================
        center_x_per_point, center_y_per_point = get_point_level_cluster_centers(pts, labels)

        frame_df = pd.DataFrame({
            "Frame": np.full(len(labels), fid, dtype=int),
            "X": frame_data[fid]["X"],
            "Y": frame_data[fid]["Y"],
            "V": frame_data[fid]["V"],
            "SNR": frame_data[fid]["SNR"],
            "Label": labels,
            "Center_X": center_x_per_point,
            "Center_Y": center_y_per_point,
        })

        # 如果 load_data2.py 里保留了这些原始列，就一起导出
        for col in ["Range", "Angle", "Speed"]:
            if col in frame_data[fid]:
                frame_df[col] = frame_data[fid][col]

        labeled_rows.append(frame_df)

        g = gt_df[gt_df["Frame"] == fid]
        gt_list = [
            {"id": int(r.ID), "x": float(r.X), "y": float(r.Y), "model": int(r.model)}
            for r in g.itertuples(index=False)
        ]

        # 评估使用 fixed_box，并与 fit_mode 一致
        m = eval_one_frame_target_level(
            pts_xy=pts,
            labels=labels,
            gt_list=gt_list,
            dist_thr=DIST_THR,
            iou_thr=IOU_THR,
            use_fixed_box=True,
            fixed_box_priors=FIXED_BOX_PRIORS,
            fixed_box_fit_mode=fixed_box_fit_mode,
            fixed_box_yaw=FIXED_BOX_YAW,
            fixed_box_score_lambda=FIXED_BOX_SCORE_LAMBDA,
            fixed_box_inside_margin=FIXED_BOX_INSIDE_MARGIN,
            fixed_box_alpha_out=FIXED_BOX_ALPHA_OUT,
            fixed_box_beta_in=FIXED_BOX_BETA_IN,
            snr=snr,
            cluster_center_mode="mean",   # mean / median / snr_mean / fixed_box
            center_bias_x=CENTER_BIAS_X,
            center_bias_y=CENTER_BIAS_Y,
            use_range_bias_y=True,
            bias_y_near=1.149,
            bias_y_far=1.586,
            bias_split_y=100.0,
        )

        for mmatch in m.get("matches", []):
            gid = int(mmatch["gid"])
            dy = float(mmatch["dy"])

            gt_item = next((gg for gg in gt_list if int(gg["id"]) == gid), None)
            if gt_item is None:
                continue

            gy = float(gt_item["y"])
            for rb in range_bins:
                lo, hi = rb
                if lo <= gy < hi:
                    range_bias_stats[rb].append(dy)
                    break

        sum_TP += int(m["TP"])
        sum_FP += int(m["FP"])
        sum_FN += int(m["FN"])
        all_center_err += list(m.get("center_errors", []))
        all_dx_err += list(m.get("dx_errors", []))
        all_dy_err += list(m.get("dy_errors", []))

        for mm in [0, 1, 2]:
            for k in ["TP", "FP", "FN"]:
                model_tot[mm][k] += int(m["model_counts"][mm][k])

        cache[fid] = {
            "pts": pts,
            "v": v,
            "labels": labels,
            "gt_list": gt_list,
            "metrics": m,
        }

    # =========================
    # 导出点集表
    # =========================
    if len(labeled_rows) > 0:
        labeled_df = pd.concat(labeled_rows, ignore_index=True)

        labeled_df.to_csv(
            "data/radar_points_with_labels.csv",
            index=False,
            encoding="utf-8-sig"
        )
        labeled_df.to_excel(
            "data/radar_points_with_labels.xlsx",
            index=False
        )

        print("Saved labeled point table:")
        print("  data/radar_points_with_labels.csv")
        print("  data/radar_points_with_labels.xlsx")

    # =========================
    # 交互显示
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
            f"Frame {fid} [{i + 1}/{n_frames}] | "
            f"mode={fixed_box_fit_mode} | "
            f"TP={m['TP']} FP={m['FP']} FN={m['FN']} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"mean_err={mean_err_str}"
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
            fixed_box_fit_mode=fixed_box_fit_mode,
            fixed_box_inside_margin=FIXED_BOX_INSIDE_MARGIN,
            fixed_box_alpha_out=FIXED_BOX_ALPHA_OUT,
            fixed_box_beta_in=FIXED_BOX_BETA_IN,
        )

        # 当前坐标系：
        # X = 横向
        # Y = 前向
        for ax in axes:
            ax.set_xlim(-40, 40)
            ax.set_ylim(0, 250)
            ax.set_autoscale_on(False)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks(np.arange(-40, 41, 5))
            ax.set_yticks(np.arange(0, 251, 5))
            ax.grid(True)
            ax.set_xlabel("X (lateral)")
            ax.set_ylabel("Y (forward)")

        # ------- 文字标注：TP / FP / FN -------
        ax_r = axes[0]  # raw
        ax_c = axes[1]  # clusters

        labels_local = item["labels"]
        pts_local = item["pts"]

        # cluster center（与 fit_mode 保持一致，仅用于显示）
        cluster_centers = {}
        for cid in np.unique(labels_local):
            if cid < 1:
                continue

            cpts = pts_local[labels_local == cid]
            if cpts.size == 0:
                continue

            best = choose_best_fixed_box_prior_mode(
                cpts,
                priors=FIXED_BOX_PRIORS,
                fit_mode=fixed_box_fit_mode,
                yaw=FIXED_BOX_YAW,
                score_lambda=FIXED_BOX_SCORE_LAMBDA,
                inside_margin=FIXED_BOX_INSIDE_MARGIN,
                alpha_out=FIXED_BOX_ALPHA_OUT,
                beta_in=FIXED_BOX_BETA_IN,
            )
            if best is not None:
                cx_fit, cy_fit = best["center"]
                cluster_centers[int(cid)] = (float(cx_fit), float(cy_fit))

        gt_pos = {int(g["id"]): (float(g["x"]), float(g["y"])) for g in item["gt_list"]}

        # 1) TP matches
        for mmatch in m.get("matches", []):
            cid = int(mmatch["cid"])
            gid = int(mmatch["gid"])
            d = float(mmatch["center_dist"])
            iou = float(mmatch["iou"]) if "iou" in mmatch else float("nan")

            if cid in cluster_centers:
                cx, cy = cluster_centers[cid]
                if np.isfinite(iou):
                    ax_c.text(cx, cy + 2.0, f"C{cid}→GT{gid}\nd={d:.1f}, IoU={iou:.2f}", fontsize=9)
                else:
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
    overall_F1 = (
        2 * overall_P * overall_R / (overall_P + overall_R)
        if (overall_P + overall_R)
        else 0.0
    )
    print(f"Overall: P={overall_P:.4f} R={overall_R:.4f} F1={overall_F1:.4f}")

    if len(all_center_err) > 0:
        ce = np.asarray(all_center_err, dtype=float)
        dxe = np.asarray(all_dx_err, dtype=float)
        dye = np.asarray(all_dy_err, dtype=float)

        mean_ce = float(np.mean(ce))
        median_ce = float(np.median(ce))
        p90_ce = float(np.percentile(ce, 90))
        p95_ce = float(np.percentile(ce, 95))
        acc_0p3m = float(np.mean(ce <= 0.3))
        acc_0p5m = float(np.mean(ce <= 0.5))

        mean_dx = float(np.mean(dxe))
        mean_dy = float(np.mean(dye))
        median_dx = float(np.median(dxe))
        median_dy = float(np.median(dye))
        std_dx = float(np.std(dxe))
        std_dy = float(np.std(dye))

        print(
            f"Center error (TP only): "
            f"mean={mean_ce:.3f}  "
            f"median={median_ce:.3f}  "
            f"p90={p90_ce:.3f}  "
            f"p95={p95_ce:.3f}  "
            f"<=0.3m={acc_0p3m * 100:.2f}%  "
            f"<=0.5m={acc_0p5m * 100:.2f}%"
        )

        print(
            f"Center bias (GT - Pred, TP only): "
            f"mean_dx={mean_dx:.3f}  "
            f"mean_dy={mean_dy:.3f}  "
            f"median_dx={median_dx:.3f}  "
            f"median_dy={median_dy:.3f}  "
            f"std_dx={std_dx:.3f}  "
            f"std_dy={std_dy:.3f}"
        )
    else:
        print("Center error (TP only): n/a (no TP matches)")
        print("Center bias (GT - Pred, TP only): n/a")

    print("\nPer-model (GT-side TP/FN, cluster-side FP assigned by nearest GT within fp_assign_dist):")
    for mm in [0, 1, 2]:
        tp = model_tot[mm]["TP"]
        fp = model_tot[mm]["FP"]
        fn = model_tot[mm]["FN"]
        p = tp / (tp + fp) if (tp + fp) else 1.0
        r = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        print(f"  model={mm}: TP={tp} FP={fp} FN={fn} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

    print("\nRange-wise Y bias (GT - Pred, TP only):")
    for rb in range_bins:
        vals = range_bias_stats[rb]
        lo, hi = rb
        hi_str = f"{hi:.0f}" if hi < 1e8 else "inf"

        if len(vals) == 0:
            print(f"  [{lo:.0f}, {hi_str}): n=0")
            continue

        arr = np.asarray(vals, dtype=float)
        print(
            f"  [{lo:.0f}, {hi_str}): "
            f"n={len(arr)}  "
            f"mean_dy={np.mean(arr):.3f}  "
            f"median_dy={np.median(arr):.3f}  "
            f"std_dy={np.std(arr):.3f}"
        )

    print("=" * 60 + "\n")
    plt.show()


if __name__ == "__main__":
    main()
