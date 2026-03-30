import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from geometry import make_box_corners
from roi_analysis import make_side_band_polygon


def compute_plot_limits(meas_df, gt_frame_df):
    xs = []
    ys = []

    if len(meas_df) > 0:
        xs.extend(meas_df["X"].values.tolist())
        ys.extend(meas_df["Y"].values.tolist())

    if len(gt_frame_df) > 0:
        xs.extend(gt_frame_df["X"].values.tolist())
        ys.extend(gt_frame_df["Y"].values.tolist())

    if not xs or not ys:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    dx = x_max - x_min
    dy = y_max - y_min

    pad_x = max(1.0, 0.1 * dx if dx > 0 else 1.0)
    pad_y = max(1.0, 0.1 * dy if dy > 0 else 1.0)

    return (
        x_min - pad_x, x_max + pad_x,
        y_min - pad_y, y_max + pad_y,
    )


def render_frame(fig, axes, frame_ids, state, frame_cache, params):
    fid = frame_ids[state["idx"]]
    cache = frame_cache[fid]

    meas_df = cache["meas_df"]
    gt_frame_df = cache["gt_frame_df"]
    targets = cache["targets"]

    ax_left, ax_right = axes
    ax_left.clear()
    ax_right.clear()

    if len(meas_df) > 0:
        ax_left.scatter(meas_df["X"].values, meas_df["Y"].values, s=12, alpha=0.75, color="blue")
        ax_right.scatter(meas_df["X"].values, meas_df["Y"].values, s=12, alpha=0.35, color="gray")

    ok_count = 0

    for t in targets:
        gx = t["gx"]
        gy = t["gy"]
        yaw_rad = t["yaw_rad"]
        length = t["length"]
        width = t["width"]
        side_info = t["side_info"]
        outward_sign = t["outward_sign"]
        roi_pts = t["roi_pts"]

        box_corners = make_box_corners(gx, gy, length, width, yaw_rad)
        ax_left.add_patch(
            Polygon(box_corners, closed=True, fill=False, linewidth=2.0, edgecolor="green")
        )
        ax_left.scatter(gx, gy, marker="x", s=60, linewidths=2, color="green")
        ax_left.text(gx + 0.08, gy + 0.08, f"ID={t['gid']}", fontsize=9, color="green")

        band_corners = make_side_band_polygon(
            center_x=gx,
            center_y=gy,
            yaw_rad=yaw_rad,
            side_info=side_info,
            outward_sign=outward_sign,
            outer_margin=params["ROI_OUTER"],
            inner_margin=params["ROI_INNER"],
        )
        ax_left.add_patch(
            Polygon(band_corners, closed=True, fill=False, linewidth=1.8, edgecolor="orange", linestyle="--")
        )

        if roi_pts.shape[0] > 0:
            ax_left.scatter(
                roi_pts[:, 0], roi_pts[:, 1],
                s=34, marker="o",
                facecolors="none", edgecolors="red", linewidths=1.2
            )
            ax_right.scatter(
                roi_pts[:, 0], roi_pts[:, 1],
                s=34, marker="o",
                facecolors="none", edgecolors="orange", linewidths=1.2
            )

        if np.isfinite(t["x_gt"]) and np.isfinite(t["y_gt"]):
            ax_right.scatter(t["x_gt"], t["y_gt"], marker="x", s=90, linewidths=2, color="green")
            ax_right.text(t["x_gt"] + 0.08, t["y_gt"] + 0.08, f"GT-{t['gid']}", fontsize=9, color="green")

        if t["ok"] == 1 and np.isfinite(t["x_hat"]) and np.isfinite(t["y_hat"]):
            ok_count += 1
            ax_right.scatter(t["x_hat"], t["y_hat"], marker="x", s=90, linewidths=2, color="red")
            ax_right.text(t["x_hat"] + 0.08, t["y_hat"] + 0.08, f"KF-{t['gid']}", fontsize=9, color="red")

    ax_left.set_title(
        f"Frame {fid} | GT Boxes + ROI + ROI Points\n"
        f"meas={len(meas_df)}, gt={len(gt_frame_df)}"
    )
    ax_right.set_title(
        f"Frame {fid} | Tail ROI Points + KF Estimate vs GT Edge Midpoint\n"
        f"valid_estimates={ok_count}/{len(targets)}"
    )

    for ax in (ax_left, ax_right):
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

    limits = compute_plot_limits(meas_df, gt_frame_df)
    if limits is not None:
        x0, x1, y0, y1 = limits
        ax_left.set_xlim(x0, x1)
        ax_left.set_ylim(y0, y1)
        ax_right.set_xlim(x0, x1)
        ax_right.set_ylim(y0, y1)

    fig.suptitle(
        f"Tail ROI + Kalman CV | Frames {frame_ids[0]}-{frame_ids[-1]} | "
        f"outer={params['ROI_OUTER']:.1f}m inner={params['ROI_INNER']:.1f}m\n"
        "Keyboard: n=next, p=prev, q/esc=quit",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.canvas.draw_idle()


def launch_viewer(frame_ids, frame_cache, params):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    state = {"idx": 0}

    def on_key(event):
        key = (event.key or "").lower()
        if key == "n":
            state["idx"] = min(state["idx"] + 1, len(frame_ids) - 1)
            render_frame(fig, axes, frame_ids, state, frame_cache, params)
        elif key == "p":
            state["idx"] = max(state["idx"] - 1, 0)
            render_frame(fig, axes, frame_ids, state, frame_cache, params)
        elif key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render_frame(fig, axes, frame_ids, state, frame_cache, params)
    plt.show()
