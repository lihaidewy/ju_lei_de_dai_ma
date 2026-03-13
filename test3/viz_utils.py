import numpy as np

from plot_raw_and_clusters_multi_prior_v2 import plot_raw_and_clusters



def _safe_mean_error_str(metrics):
    mean_err = metrics.get("mean_center_error", float("nan"))
    return "{:.2f}".format(mean_err) if np.isfinite(mean_err) else "nan"



def _build_title(fid, frame_idx, n_frames, fit_mode, metrics):
    return (
        "Frame {} [{}/{}] | mode={} | TP={} FP={} FN={} P={:.3f} R={:.3f} F1={:.3f} mean_err={}".format(
            fid,
            frame_idx + 1,
            n_frames,
            fit_mode,
            int(metrics.get("TP", 0)),
            int(metrics.get("FP", 0)),
            int(metrics.get("FN", 0)),
            float(metrics.get("precision", 0.0)),
            float(metrics.get("recall", 0.0)),
            float(metrics.get("f1", 0.0)),
            _safe_mean_error_str(metrics),
        )
    )



def _configure_axes(axes):
    for ax in axes:
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, 250)
        ax.set_autoscale_on(False)
        ax.set_xticks(np.arange(-30, 31, 5))
        ax.set_yticks(np.arange(0, 251, 5))
        ax.grid(True)
        ax.set_xlabel("X (lateral)")
        ax.set_ylabel("Y (forward)")



def _normalize_cluster_centers(cluster_centers):
    out = {}
    for cid, center in cluster_centers.items():
        out[int(cid)] = (float(center[0]), float(center[1]))
    return out



def _normalize_track_assignments(track_assignments):
    out = {}
    for cid, tid in track_assignments.items():
        out[int(cid)] = int(tid)
    return out



def _get_gt_positions(item):
    if "gt_pos_map" in item and item["gt_pos_map"] is not None:
        gt_pos = {}
        for gid, pos in item["gt_pos_map"].items():
            gt_pos[int(gid)] = (float(pos[0]), float(pos[1]))
        return gt_pos

    gt_pos = {}
    for g in item.get("gt_list", []):
        gt_pos[int(g["id"])] = (float(g["x"]), float(g["y"]))
    return gt_pos



def _annotate_matches(ax, matches, cluster_centers, track_assignments):
    for mmatch in matches:
        cid = int(mmatch["cid"])
        gid = int(mmatch["gid"])
        if cid not in cluster_centers:
            continue

        cx, cy = cluster_centers[cid]
        tid = track_assignments.get(cid, -1)
        dist = float(mmatch.get("center_dist", float("nan")))
        iou = float(mmatch.get("iou", float("nan")))

        if np.isfinite(iou):
            text = "C{}/T{}→GT{}\nd={:.1f}, IoU={:.2f}".format(cid, tid, gid, dist, iou)
        else:
            text = "C{}/T{}→GT{}\nd={:.1f}".format(cid, tid, gid, dist)
        ax.text(cx, cy + 2.0, text, fontsize=9)



def _annotate_unmatched_clusters(ax, unmatched_clusters, cluster_centers, track_assignments):
    for cid in unmatched_clusters:
        cid = int(cid)
        if cid not in cluster_centers:
            continue

        cx, cy = cluster_centers[cid]
        tid = track_assignments.get(cid, -1)
        ax.text(cx, cy + 2.0, "C{}/T{}→FP".format(cid, tid), fontsize=9)



def _annotate_unmatched_gts(ax, unmatched_gts, gt_pos):
    for gid in unmatched_gts:
        gid = int(gid)
        if gid not in gt_pos:
            continue

        gx, gy = gt_pos[gid]
        ax.text(gx, gy + 2.0, "GT{}(FN)".format(gid), fontsize=9)



def render_frame(fig, axes, cache, frame_ids, n_frames, state, i, fit_mode, cfg):
    i = int(np.clip(i, 0, n_frames - 1))
    state["i"] = i

    fid = frame_ids[i]
    item = cache[fid]
    metrics = item["metrics"]

    title = _build_title(fid, i, n_frames, fit_mode, metrics)

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

    _configure_axes(axes)

    ax_raw = axes[0]
    ax_cluster = axes[1]

    cluster_centers = _normalize_cluster_centers(item.get("cluster_centers", {}))
    track_assignments = _normalize_track_assignments(item.get("track_assignments", {}))
    gt_pos = _get_gt_positions(item)

    _annotate_matches(ax_cluster, metrics.get("matches", []), cluster_centers, track_assignments)
    _annotate_unmatched_clusters(
        ax_cluster,
        metrics.get("unmatched_clusters", []),
        cluster_centers,
        track_assignments,
    )
    _annotate_unmatched_gts(ax_raw, metrics.get("unmatched_gts", []), gt_pos)

    fig.canvas.draw_idle()
