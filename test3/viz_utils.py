import numpy as np
import matplotlib.pyplot as plt

from plot_raw_and_clusters_multi_prior_v2 import plot_raw_and_clusters
# from centers import compute_center_mean, apply_two_segment_bias


def render_frame(fig, axes, cache, frame_ids, n_frames, state, i, fit_mode, cfg):
    i = int(np.clip(i, 0, n_frames - 1))
    state["i"] = i

    fid = frame_ids[i]
    item = cache[fid]
    m = item["metrics"]

    mean_err = m["mean_center_error"]
    mean_err_str = f"{mean_err:.2f}" if np.isfinite(mean_err) else "nan"

    title = (
        f"Frame {fid} [{i + 1}/{n_frames}] | "
        f"mode={fit_mode} | "
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
        fixed_box_priors=cfg.FIXED_BOX_PRIORS,
        fixed_box_yaw=cfg.FIXED_BOX_YAW,
        fixed_box_score_lambda=cfg.FIXED_BOX_SCORE_LAMBDA,
        fixed_box_fit_mode=fit_mode,
        fixed_box_inside_margin=cfg.FIXED_BOX_INSIDE_MARGIN,
        fixed_box_alpha_out=cfg.FIXED_BOX_ALPHA_OUT,
        fixed_box_beta_in=cfg.FIXED_BOX_BETA_IN,
    )

    for ax in axes:
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, 250)
        ax.set_autoscale_on(False)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(-30, 31, 5))
        ax.set_yticks(np.arange(0, 251, 5))
        ax.grid(True)
        ax.set_xlabel("X (lateral)")
        ax.set_ylabel("Y (forward)")

    ax_r = axes[0]
    ax_c = axes[1]

    labels_local = item["labels"]
    pts_local = item["pts"]

    # cluster_centers = {}
    # for cid in np.unique(labels_local):
    #     if cid < 1:
    #         continue

    #     cpts = pts_local[labels_local == cid]
    #     if cpts.size == 0:
    #         continue

    #     center = compute_center_mean(cpts)
    #     center = apply_two_segment_bias(center, cfg)
    #     cluster_centers[int(cid)] = (float(center[0]), float(center[1]))

    cluster_centers = {}
    for cid, center in item.get("cluster_centers", {}).items():
        cluster_centers[int(cid)] = (float(center[0]), float(center[1]))


    gt_pos = {int(g["id"]): (float(g["x"]), float(g["y"])) for g in item["gt_list"]}

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

    for cid in m.get("unmatched_clusters", []):
        cid = int(cid)
        if cid in cluster_centers:
            cx, cy = cluster_centers[cid]
            ax_c.text(cx, cy + 2.0, f"C{cid}→FP", fontsize=9)

    for gid in m.get("unmatched_gts", []):
        gid = int(gid)
        if gid in gt_pos:
            gx, gy = gt_pos[gid]
            ax_r.text(gx, gy + 2.0, f"GT{gid}(FN)", fontsize=9)

    fig.canvas.draw_idle()
