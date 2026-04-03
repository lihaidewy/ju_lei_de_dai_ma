import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from geometry import make_box_corners
from roi_analysis import make_side_band_polygon


def _compute_global_limits(frame_ids, frame_cache):
    all_x = []
    all_y = []

    for fid in frame_ids:
        cache = frame_cache[fid]

        meas_df = cache.get("meas_df", None)
        gt_frame_df = cache.get("gt_frame_df", None)
        targets = cache.get("targets", [])

        if meas_df is not None and len(meas_df) > 0:
            all_x.extend(meas_df["X"].values.tolist())
            all_y.extend(meas_df["Y"].values.tolist())

        if gt_frame_df is not None and len(gt_frame_df) > 0:
            all_x.extend(gt_frame_df["X"].values.tolist())
            all_y.extend(gt_frame_df["Y"].values.tolist())

        for t in targets:
            if np.isfinite(t.get("gx", np.nan)) and np.isfinite(t.get("gy", np.nan)):
                all_x.append(float(t["gx"]))
                all_y.append(float(t["gy"]))

            if np.isfinite(t.get("x_gt", np.nan)) and np.isfinite(t.get("y_gt", np.nan)):
                all_x.append(float(t["x_gt"]))
                all_y.append(float(t["y_gt"]))

            if np.isfinite(t.get("x_hat", np.nan)) and np.isfinite(t.get("y_hat", np.nan)):
                all_x.append(float(t["x_hat"]))
                all_y.append(float(t["y_hat"]))

            roi_pts = t.get("roi_pts", None)
            if roi_pts is not None and getattr(roi_pts, "shape", (0, 0))[0] > 0:
                all_x.extend(np.asarray(roi_pts[:, 0], dtype=float).tolist())
                all_y.extend(np.asarray(roi_pts[:, 1], dtype=float).tolist())

    if not all_x or not all_y:
        return (-10.0, 10.0, -10.0, 10.0)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    dx = x_max - x_min
    dy = y_max - y_min

    pad_x = max(1.0, 0.08 * dx if dx > 0 else 1.0)
    pad_y = max(1.0, 0.08 * dy if dy > 0 else 1.0)

    return (
        x_min - pad_x,
        x_max + pad_x,
        y_min - pad_y,
        y_max + pad_y,
    )


def _resolve_fixed_limits(frame_ids, frame_cache, params):
    has_manual = all(
        key in params for key in ["VIEW_XMIN", "VIEW_XMAX", "VIEW_YMIN", "VIEW_YMAX"]
    )

    if has_manual:
        return (
            float(params["VIEW_XMIN"]),
            float(params["VIEW_XMAX"]),
            float(params["VIEW_YMIN"]),
            float(params["VIEW_YMAX"]),
        )

    return _compute_global_limits(frame_ids, frame_cache)


def _configure_axes(ax_left, ax_right, state, params):
    x0, x1, y0, y1 = state["fixed_limits"]

    for ax in (ax_left, ax_right):
        ax.grid(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_autoscale_on(False)

        xstep = params.get("VIEW_XTICK_STEP", None)
        ystep = params.get("VIEW_YTICK_STEP", None)

        if xstep is not None and float(xstep) > 0:
            ax.set_xticks(np.arange(x0, x1 + 1e-9, float(xstep)))
        if ystep is not None and float(ystep) > 0:
            ax.set_yticks(np.arange(y0, y1 + 1e-9, float(ystep)))


def _estimation_mode_text(params):
    mode = str(params.get("ESTIMATION_MODE", "cv_recursive")).lower()
    if mode == "cv_recursive":
        return "Recursive CV"
    if mode == "raw":
        return "Raw Measurement"
    return mode


def _count_valid_estimates(targets):
    ok_count = 0
    for t in targets:
        if int(t.get("ok", 0)) == 1 and np.isfinite(t.get("x_hat", np.nan)) and np.isfinite(t.get("y_hat", np.nan)):
            ok_count += 1
    return ok_count


def _count_clusters(labels):
    if labels is None:
        return 0
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    return int(np.sum(uniq >= 1))


def _build_suptitle(fid, frame_idx, n_frames, params, state, targets, labels):
    mode_text = _estimation_mode_text(params)
    viewer_mode = "ANIM" if state["enable_animation"] else "STATIC"
    play_state = "PLAY" if state["playing"] else "PAUSE"
    ok_count = _count_valid_estimates(targets)
    n_clusters = _count_clusters(labels)

    return (
        f"Frame {fid} [{frame_idx + 1}/{n_frames}] | "
        f"mode={mode_text} | viewer={viewer_mode} {play_state} | "
        f"valid_estimates={ok_count}/{len(targets)} | clusters={n_clusters}"
    )


def _draw_measurements(ax, meas_df, alpha=0.75, color="blue", size=12):
    if meas_df is None or len(meas_df) == 0:
        return
    ax.scatter(
        meas_df["X"].values,
        meas_df["Y"].values,
        s=size,
        alpha=alpha,
        color=color,
    )


def _draw_clustered_measurements(ax, meas_df, labels, show_noise=True):
    if meas_df is None or len(meas_df) == 0:
        return

    x = meas_df["X"].values
    y = meas_df["Y"].values

    if labels is None or len(labels) != len(meas_df):
        ax.scatter(x, y, s=12, alpha=0.35, color="gray")
        return

    labels = np.asarray(labels)

    if show_noise:
        ax.scatter(x, y, c=labels, s=12, alpha=0.60)
        return

    uniq = np.unique(labels)
    for cid in uniq:
        cid = int(cid)
        mask = labels == cid
        if np.sum(mask) == 0:
            continue

        if cid == -1:
            ax.scatter(x[mask], y[mask], s=10, alpha=0.15, color="gray")
        else:
            ax.scatter(x[mask], y[mask], s=12, alpha=0.80)


def _draw_cluster_text(ax, meas_df, labels):
    if meas_df is None or len(meas_df) == 0:
        return
    if labels is None or len(labels) != len(meas_df):
        return

    x = meas_df["X"].values
    y = meas_df["Y"].values
    labels = np.asarray(labels)

    for cid in np.unique(labels):
        cid = int(cid)
        mask = labels == cid
        if np.sum(mask) == 0:
            continue

        cx = float(np.mean(x[mask]))
        cy = float(np.mean(y[mask]))
        tag = "noise" if cid == -1 else f"C{cid}"
        ax.text(cx + 0.05, cy + 0.05, tag, fontsize=8, color="red")


def _draw_left_targets(ax, targets, params, show_gt_box=True, show_roi_band=True, show_roi_pts=True, show_gt_id=True):
    for t in targets:
        gx = t["gx"]
        gy = t["gy"]
        yaw_rad = t["yaw_rad"]
        length = t["length"]
        width = t["width"]
        side_info = t["side_info"]
        outward_sign = t["outward_sign"]
        roi_pts = t["roi_pts"]

        if show_gt_box:
            box_corners = make_box_corners(gx, gy, length, width, yaw_rad)
            ax.add_patch(
                Polygon(
                    box_corners,
                    closed=True,
                    fill=False,
                    linewidth=2.0,
                    edgecolor="green"
                )
            )

        ax.scatter(gx, gy, marker="x", s=60, linewidths=2, color="green")

        if show_gt_id:
            ax.text(gx + 0.08, gy + 0.08, f"ID={t['gid']}", fontsize=9, color="green")

        if show_roi_band:
            band_corners = make_side_band_polygon(
                center_x=gx,
                center_y=gy,
                yaw_rad=yaw_rad,
                side_info=side_info,
                outward_sign=outward_sign,
                outer_margin=params["ROI_OUTER"],
                inner_margin=params["ROI_INNER"],
            )
            ax.add_patch(
                Polygon(
                    band_corners,
                    closed=True,
                    fill=False,
                    linewidth=1.8,
                    edgecolor="orange",
                    linestyle="--"
                )
            )

        if show_roi_pts and roi_pts.shape[0] > 0:
            ax.scatter(
                roi_pts[:, 0],
                roi_pts[:, 1],
                s=34,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=1.2
            )


def _draw_right_targets(ax, targets, params, show_roi_pts=True, show_gt=True, show_est=True):
    estimation_mode = str(params.get("ESTIMATION_MODE", "cv_recursive")).lower()

    for t in targets:
        roi_pts = t["roi_pts"]

        if show_roi_pts and roi_pts.shape[0] > 0:
            ax.scatter(
                roi_pts[:, 0],
                roi_pts[:, 1],
                s=34,
                marker="o",
                facecolors="none",
                edgecolors="orange",
                linewidths=1.2
            )

        if show_gt and np.isfinite(t["x_gt"]) and np.isfinite(t["y_gt"]):
            ax.scatter(
                t["x_gt"],
                t["y_gt"],
                marker="x",
                s=90,
                linewidths=2,
                color="green"
            )
            ax.text(
                t["x_gt"] + 0.08,
                t["y_gt"] + 0.08,
                f"GT-{t['gid']}",
                fontsize=9,
                color="green"
            )

        if show_est and int(t.get("ok", 0)) == 1 and np.isfinite(t.get("x_hat", np.nan)) and np.isfinite(t.get("y_hat", np.nan)):
            ax.scatter(
                t["x_hat"],
                t["y_hat"],
                marker="x",
                s=90,
                linewidths=2,
                color="red"
            )

            if estimation_mode == "cv_recursive":
                tag = f"CV-{t['gid']}"
            elif estimation_mode == "raw":
                tag = f"RAW-{t['gid']}"
            else:
                tag = f"OUT-{t['gid']}"

            ax.text(
                t["x_hat"] + 0.08,
                t["y_hat"] + 0.08,
                tag,
                fontsize=9,
                color="red"
            )


def render_frame(fig, axes, frame_ids, state, frame_cache, params):
    i = int(np.clip(state["idx"], 0, len(frame_ids) - 1))
    state["idx"] = i

    fid = frame_ids[i]
    cache = frame_cache[fid]

    meas_df = cache["meas_df"]
    gt_frame_df = cache["gt_frame_df"]
    targets = cache["targets"]
    labels = cache.get("cluster_labels", None)

    ax_left, ax_right = axes
    ax_left.clear()
    ax_right.clear()

    _configure_axes(ax_left, ax_right, state, params)

    fig.suptitle(
        _build_suptitle(fid, i, len(frame_ids), params, state, targets, labels),
        fontsize=12
    )

    if state["show_measurements_left"]:
        _draw_measurements(
            ax_left,
            meas_df,
            alpha=0.75,
            color="blue",
            size=12,
        )

    _draw_left_targets(
        ax=ax_left,
        targets=targets,
        params=params,
        show_gt_box=state["show_gt_box_left"],
        show_roi_band=state["show_roi_band_left"],
        show_roi_pts=state["show_roi_pts_left"],
        show_gt_id=state["show_gt_id_left"],
    )

    if state["show_measurements_right"]:
        if state["show_clusters_right"]:
            _draw_clustered_measurements(
                ax=ax_right,
                meas_df=meas_df,
                labels=labels,
                show_noise=state["show_noise_right"],
            )
        else:
            _draw_measurements(
                ax_right,
                meas_df,
                alpha=0.35,
                color="gray",
                size=12,
            )

    if state["show_cluster_text_right"]:
        _draw_cluster_text(ax_right, meas_df, labels)

    _draw_right_targets(
        ax=ax_right,
        targets=targets,
        params=params,
        show_roi_pts=state["show_roi_pts_right"],
        show_gt=state["show_gt_right"],
        show_est=state["show_est_right"],
    )

    ax_left.set_title(
        f"Frame {fid} | Left: Measurements + GT Box + ROI\n"
        f"meas={len(meas_df)}, gt={len(gt_frame_df)}, roi_targets={len(targets)}"
    )

    ax_right.set_title(
        f"Frame {fid} | Right: Clusters + ROI + GT + Estimates\n"
        f"labels={'yes' if labels is not None else 'no'}, targets={len(targets)}"
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.canvas.draw_idle()


def launch_viewer(frame_ids, frame_cache, params):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    plt.subplots_adjust(bottom=0.08)

    fixed_limits = _resolve_fixed_limits(frame_ids, frame_cache, params)

    state = {
        "idx": 0,
        "playing": False,
        "enable_animation": True,
        "loop": True,
        "fixed_limits": fixed_limits,

        "show_measurements_left": True,
        "show_gt_box_left": True,
        "show_roi_band_left": True,
        "show_roi_pts_left": True,
        "show_gt_id_left": True,

        "show_measurements_right": True,
        "show_clusters_right": True,
        "show_noise_right": True,
        "show_cluster_text_right": False,
        "show_roi_pts_right": True,
        "show_gt_right": True,
        "show_est_right": True,

        "_ani": None,
    }

    def redraw(new_idx=None):
        if new_idx is not None:
            state["idx"] = int(np.clip(new_idx, 0, len(frame_ids) - 1))
        render_frame(fig, axes, frame_ids, state, frame_cache, params)

    def on_key(event):
        key = (event.key or "").lower()

        if key == "n":
            state["playing"] = False
            redraw(state["idx"] + 1)

        elif key == "p":
            state["playing"] = False
            redraw(state["idx"] - 1)

        elif key in ("q", "escape"):
            plt.close(fig)

        elif key == " ":
            if state["enable_animation"]:
                state["playing"] = not state["playing"]
                redraw(state["idx"])

        elif key == "a":
            state["enable_animation"] = not state["enable_animation"]
            if not state["enable_animation"]:
                state["playing"] = False
            redraw(state["idx"])

        elif key == "x":
            state["show_clusters_right"] = not state["show_clusters_right"]
            redraw(state["idx"])

        elif key == "z":
            state["show_noise_right"] = not state["show_noise_right"]
            redraw(state["idx"])

        elif key == "k":
            state["show_cluster_text_right"] = not state["show_cluster_text_right"]
            redraw(state["idx"])

        elif key == "g":
            state["show_gt_right"] = not state["show_gt_right"]
            redraw(state["idx"])

        elif key == "e":
            state["show_est_right"] = not state["show_est_right"]
            redraw(state["idx"])

        elif key == "r":
            state["show_roi_pts_right"] = not state["show_roi_pts_right"]
            redraw(state["idx"])

        elif key == "1":
            state["show_measurements_left"] = not state["show_measurements_left"]
            redraw(state["idx"])

        elif key == "2":
            state["show_gt_box_left"] = not state["show_gt_box_left"]
            redraw(state["idx"])

        elif key == "3":
            state["show_roi_band_left"] = not state["show_roi_band_left"]
            redraw(state["idx"])

        elif key == "4":
            state["show_roi_pts_left"] = not state["show_roi_pts_left"]
            redraw(state["idx"])

        elif key == "5":
            state["show_gt_id_left"] = not state["show_gt_id_left"]
            redraw(state["idx"])

    def update(_frame_idx):
        if not (state["enable_animation"] and state["playing"]):
            return []

        if state["idx"] >= len(frame_ids) - 1:
            if state["loop"]:
                redraw(0)
            else:
                state["playing"] = False
                redraw(state["idx"])
        else:
            redraw(state["idx"] + 1)

        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=int(params.get("ANIM_INTERVAL_MS", 250)),
        blit=False,
        cache_frame_data=False,
    )
    state["_ani"] = ani

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw(0)
    plt.show()
