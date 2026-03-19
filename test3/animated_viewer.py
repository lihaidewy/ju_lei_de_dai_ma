import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


def _safe_mean_error_str(metrics):
    mean_err = metrics.get("mean_center_error", float("nan"))
    return "{:.2f}".format(mean_err) if np.isfinite(mean_err) else "nan"


def _build_title(fid, frame_idx, n_frames, fit_mode, metrics, state):
    mode_str = "ANIM" if state["enable_animation"] else "STATIC"
    play_str = "PLAY" if state["playing"] else "PAUSE"
    return (
        "Frame {} [{}/{}] | mode={} | viewer={} {} | "
        "TP={} FP={} FN={} P={:.3f} R={:.3f} F1={:.3f} mean_err={}".format(
            fid,
            frame_idx + 1,
            n_frames,
            fit_mode,
            mode_str,
            play_str,
            int(metrics.get("TP", 0)),
            int(metrics.get("FP", 0)),
            int(metrics.get("FN", 0)),
            float(metrics.get("precision", 0.0)),
            float(metrics.get("recall", 0.0)),
            float(metrics.get("f1", 0.0)),
            _safe_mean_error_str(metrics),
        )
    )


def _configure_axes(axes, cfg):
    x0, x1 = getattr(cfg, "VIEW_XLIM", (-30, 30))
    y0, y1 = getattr(cfg, "VIEW_YLIM", (0, 250))
    xstep = float(getattr(cfg, "VIEW_XTICK_STEP", 5.0))
    ystep = float(getattr(cfg, "VIEW_YTICK_STEP", 5.0))

    for ax in axes:
        ax.set_xlim(float(x0), float(x1))
        ax.set_ylim(float(y0), float(y1))
        ax.set_autoscale_on(False)
        ax.set_xticks(np.arange(float(x0), float(x1) + 1e-9, xstep))
        ax.set_yticks(np.arange(float(y0), float(y1) + 1e-9, ystep))
        ax.grid(True)
        ax.set_xlabel("X (lateral)")
        ax.set_ylabel("Y (forward)")


def _normalize_cluster_centers(cluster_centers):
    out = {}
    for cid, center in cluster_centers.items():
        out[int(cid)] = np.asarray(center, dtype=float).reshape(2)
    return out


def _normalize_track_assignments(track_assignments):
    out = {}
    for cid, tid in track_assignments.items():
        out[int(cid)] = int(tid)
    return out


def _gt_box_xywh(g, cfg):
    model = int(g["model"])
    priors = getattr(cfg, "GT_MODEL_PRIORS", {})
    item = priors.get(model, None)
    if item is None:
        return None

    # 当前统一坐标系：
    # X方向宽 W，Y方向长 L
    L = float(item["L"])
    W = float(item["W"])
    cx = float(g["x"])
    cy = float(g["y"])
    return cx - W / 2.0, cy - L / 2.0, W, L


def _draw_gt(ax, gt_list, cfg, show_boxes=True, show_ids=True):
    for g in gt_list:
        gx = float(g["x"])
        gy = float(g["y"])
        gid = int(g["id"])
        model = int(g["model"])

        ax.scatter([gx], [gy], marker="x", s=60)

        if show_boxes:
            rect_args = _gt_box_xywh(g, cfg)
            if rect_args is not None:
                rect = Rectangle(
                    (rect_args[0], rect_args[1]),
                    rect_args[2],
                    rect_args[3],
                    fill=False,
                    linewidth=1.5,
                )
                ax.add_patch(rect)

        if show_ids:
            ax.text(gx, gy + 1.5, f"GT{gid}/M{model}", fontsize=8)


def _draw_measurements(ax, pts):
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.75)


def _draw_clusters(ax, pts, labels, show_noise=False):
    pts = np.asarray(pts, dtype=float)
    labels = np.asarray(labels)

    if pts.size == 0:
        return

    uniq = np.unique(labels)
    for cid in uniq:
        cid = int(cid)
        if cid < 1 and not show_noise:
            continue

        mask = labels == cid
        cpts = pts[mask]
        if cpts.size == 0:
            continue

        if cid < 1:
            ax.scatter(cpts[:, 0], cpts[:, 1], s=8, alpha=0.25)
        else:
            ax.scatter(cpts[:, 0], cpts[:, 1], s=12, alpha=0.85)


def _draw_centers(ax, centers, track_assignments, show_text=True):
    for cid, center in centers.items():
        cx, cy = float(center[0]), float(center[1])
        tid = track_assignments.get(int(cid), -1)

        ax.scatter([cx], [cy], marker="o", s=60)
        if show_text:
            ax.text(cx, cy + 1.5, f"C{cid}/T{tid}", fontsize=8)


def _draw_matches(ax, metrics, centers, track_assignments):
    for mmatch in metrics.get("matches", []):
        cid = int(mmatch["cid"])
        gid = int(mmatch["gid"])
        if cid not in centers:
            continue

        cx, cy = centers[cid]
        tid = track_assignments.get(cid, -1)
        dist = float(mmatch.get("center_dist", float("nan")))
        ax.text(cx, cy + 3.0, f"C{cid}/T{tid}->GT{gid} d={dist:.1f}", fontsize=8)

    for cid in metrics.get("unmatched_clusters", []):
        cid = int(cid)
        if cid not in centers:
            continue
        cx, cy = centers[cid]
        tid = track_assignments.get(cid, -1)
        ax.text(cx, cy + 3.0, f"C{cid}/T{tid}->FP", fontsize=8)

    gt_pos = {
        int(g["id"]): (float(g["x"]), float(g["y"]))
        for g in metrics.get("gt_list", [])
    }
    for gid in metrics.get("unmatched_gts", []):
        gid = int(gid)
        if gid not in gt_pos:
            continue
        gx, gy = gt_pos[gid]
        ax.text(gx, gy + 3.0, f"GT{gid}(FN)", fontsize=8)


def build_track_history(cache, frame_ids):
    history = {}
    for fid in frame_ids:
        item = cache[fid]
        centers = _normalize_cluster_centers(item.get("cluster_centers", {}))
        track_assignments = _normalize_track_assignments(item.get("track_assignments", {}))

        for cid, tid in track_assignments.items():
            if cid not in centers:
                continue
            center = centers[cid]
            history.setdefault(tid, []).append(
                {
                    "fid": int(fid),
                    "x": float(center[0]),
                    "y": float(center[1]),
                }
            )
    return history


def _draw_track_trails(ax, history, current_fid, trail_len=20, show_id=True):
    for tid, seq in history.items():
        past = [p for p in seq if p["fid"] <= int(current_fid)]
        if not past:
            continue

        past = past[-int(trail_len):]
        xs = [p["x"] for p in past]
        ys = [p["y"] for p in past]

        ax.plot(xs, ys, linewidth=1.5)
        if show_id:
            ax.text(xs[-1], ys[-1] + 1.0, f"T{tid}", fontsize=8)


def render_animated_frame(fig, axes, cache, frame_ids, n_frames, state, i, fit_mode, cfg):
    i = int(np.clip(i, 0, n_frames - 1))
    state["i"] = i

    fid = frame_ids[i]
    item = cache[fid]
    metrics = dict(item["metrics"])
    metrics["gt_list"] = item.get("gt_list", [])

    ax_left, ax_right = axes
    ax_left.clear()
    ax_right.clear()
    _configure_axes(axes, cfg)

    fig.suptitle(_build_title(fid, i, n_frames, fit_mode, metrics, state))

    pts = np.asarray(item["pts"], dtype=float)
    labels = np.asarray(item["labels"])
    gt_list = item.get("gt_list", [])
    centers = _normalize_cluster_centers(item.get("cluster_centers", {}))
    track_assignments = _normalize_track_assignments(item.get("track_assignments", {}))

    # 左图：真值框 + 量测
    if state["show_measurements_left"]:
        _draw_measurements(ax_left, pts)
    if state["show_gt"]:
        _draw_gt(
            ax_left,
            gt_list,
            cfg,
            show_boxes=state["show_gt_box"],
            show_ids=state["show_gt_id"],
        )
    ax_left.set_title("GT + Measurements")

    # 右图：量测 + 聚类 + 中心点 + 轨迹
    if state["show_measurements_right"]:
        if state["show_clusters"]:
            _draw_clusters(ax_right, pts, labels, show_noise=state["show_noise"])
        else:
            _draw_measurements(ax_right, pts)

    if state["show_centers"]:
        _draw_centers(
            ax_right,
            centers,
            track_assignments,
            show_text=state["show_center_text"],
        )

    if state["show_tracks"]:
        _draw_track_trails(
            ax_right,
            state["track_history"],
            current_fid=fid,
            trail_len=state["trail_len"],
            show_id=state["show_track_id"],
        )

    if state["show_match_text"]:
        _draw_matches(ax_right, metrics, centers, track_assignments)

    ax_right.set_title("Measurements + Clusters + Tracks + Centers")

    help_text = (
        "[n] next  [p] prev  [q] quit\n"
        "[space] play/pause  [a] anim on/off\n"
        "[t] tracks  [c] centers  [g] gt  [b] gt_box\n"
        "[m] match text  [x] clusters  [z] noise"
    )
    ax_right.text(
        0.02,
        0.98,
        help_text,
        transform=ax_right.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    fig.canvas.draw_idle()

def export_animation(fig, ani, state, frame_ids, cfg):
    if not bool(getattr(cfg, "EXPORT_ANIMATION", False)):
        return

    export_fmt = str(getattr(cfg, "EXPORT_ANIMATION_FORMAT", "gif")).lower()
    export_path = str(getattr(cfg, "EXPORT_ANIMATION_PATH", "data/animation_result.gif"))
    export_fps = int(getattr(cfg, "EXPORT_ANIMATION_FPS", 5))
    export_dpi = int(getattr(cfg, "EXPORT_ANIMATION_DPI", 120))

    print(f"[Animation] exporting to {export_path} ...")

    old_playing = state.get("playing", False)
    old_exporting = state.get("exporting", False)

    try:
        state["playing"] = False
        state["exporting"] = True

        if export_fmt == "gif":
            ani.save(export_path, writer="pillow", fps=export_fps, dpi=export_dpi)
        elif export_fmt == "mp4":
            ani.save(export_path, writer="ffmpeg", fps=export_fps, dpi=export_dpi)
        else:
            raise ValueError(f"Unsupported EXPORT_ANIMATION_FORMAT: {export_fmt}")

        print(f"[Animation] export done: {export_path}")

    finally:
        state["playing"] = old_playing
        state["exporting"] = old_exporting


def launch_animated_viewer(cache, frame_ids, cfg, fit_mode="center"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharex=True, sharey=True)
    plt.subplots_adjust(bottom=0.08)

    state = {
        "i": 0,
        "playing": bool(getattr(cfg, "ANIM_AUTOPLAY", False)),
        "enable_animation": True,
        "loop": bool(getattr(cfg, "ANIM_LOOP", True)),
        "trail_len": int(getattr(cfg, "ANIM_TRAIL_LEN", 20)),
        "show_gt": bool(getattr(cfg, "VIEW_SHOW_GT", True)),
        "show_gt_box": bool(getattr(cfg, "VIEW_SHOW_GT_BOX", True)),
        "show_gt_id": bool(getattr(cfg, "VIEW_SHOW_GT_ID", True)),
        "show_measurements_left": bool(getattr(cfg, "VIEW_SHOW_MEASUREMENTS_LEFT", True)),
        "show_measurements_right": bool(getattr(cfg, "VIEW_SHOW_MEASUREMENTS_RIGHT", True)),
        "show_clusters": bool(getattr(cfg, "VIEW_SHOW_CLUSTERS", True)),
        "show_centers": bool(getattr(cfg, "VIEW_SHOW_CENTERS", True)),
        "show_center_text": bool(getattr(cfg, "VIEW_SHOW_CENTER_TEXT", True)),
        "show_tracks": bool(getattr(cfg, "VIEW_SHOW_TRACKS", True)),
        "show_track_id": bool(getattr(cfg, "VIEW_SHOW_TRACK_ID", True)),
        "show_match_text": bool(getattr(cfg, "VIEW_SHOW_MATCH_TEXT", False)),
        "show_noise": bool(getattr(cfg, "VIEW_SHOW_NOISE", False)),
        "track_history": build_track_history(cache, frame_ids),
        "exporting": False,
        "_ani": None,
    }

    def redraw(idx=None):
        if idx is None:
            idx = state["i"]
        render_animated_frame(
            fig=fig,
            axes=axes,
            cache=cache,
            frame_ids=frame_ids,
            n_frames=len(frame_ids),
            state=state,
            i=idx,
            fit_mode=fit_mode,
            cfg=cfg,
        )

    def on_key(event):
        key = (event.key or "").lower()

        # 保持以前按键不变
        if key == "n":
            state["playing"] = False
            redraw(state["i"] + 1)

        elif key == "p":
            state["playing"] = False
            redraw(state["i"] - 1)

        elif key in ("q", "escape"):
            plt.close(fig)

        # 新增动画和显示开关
        elif key == " ":
            if state["enable_animation"]:
                state["playing"] = not state["playing"]
                redraw(state["i"])

        elif key == "a":
            state["enable_animation"] = not state["enable_animation"]
            if not state["enable_animation"]:
                state["playing"] = False
            redraw(state["i"])

        elif key == "t":
            state["show_tracks"] = not state["show_tracks"]
            redraw(state["i"])

        elif key == "c":
            state["show_centers"] = not state["show_centers"]
            redraw(state["i"])

        elif key == "g":
            state["show_gt"] = not state["show_gt"]
            redraw(state["i"])

        elif key == "b":
            state["show_gt_box"] = not state["show_gt_box"]
            redraw(state["i"])

        elif key == "m":
            state["show_match_text"] = not state["show_match_text"]
            redraw(state["i"])

        elif key == "x":
            state["show_clusters"] = not state["show_clusters"]
            redraw(state["i"])

        elif key == "z":
            state["show_noise"] = not state["show_noise"]
            redraw(state["i"])

    def update(frame_idx):
        # 导出模式：严格按 frame_idx 逐帧渲染
        if state.get("exporting", False):
            idx = int(frame_idx)
            idx = max(0, min(idx, len(frame_ids) - 1))
            redraw(idx)
            return []

        # 交互模式：按播放状态推进
        if not (state["enable_animation"] and state["playing"]):
            return []

        if state["i"] >= len(frame_ids) - 1:
            if state["loop"]:
                redraw(0)
            else:
                state["playing"] = False
                redraw(state["i"])
        else:
            redraw(state["i"] + 1)
        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=int(getattr(cfg, "ANIM_INTERVAL_MS", 250)),
        blit=False,
        cache_frame_data=False,
    )

    state["_ani"] = ani

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw(0)
    
    export_animation(fig, ani, state, frame_ids, cfg)

    # if bool(getattr(cfg, "EXPORT_ANIMATION", False)):
    #     export_fmt = str(getattr(cfg, "EXPORT_ANIMATION_FORMAT", "gif")).lower()
    #     export_path = str(getattr(cfg, "EXPORT_ANIMATION_PATH", "data/animation_result.gif"))
    #     export_fps = int(getattr(cfg, "EXPORT_ANIMATION_FPS", 5))
    #     export_dpi = int(getattr(cfg, "EXPORT_ANIMATION_DPI", 120))

    #     print(f"[Animation] exporting to {export_path} ...")

    #     if export_fmt == "gif":
    #         ani.save(export_path, writer="pillow", fps=export_fps, dpi=export_dpi)
    #     elif export_fmt == "mp4":
    #         ani.save(export_path, writer="ffmpeg", fps=export_fps, dpi=export_dpi)
    #     else:
    #         raise ValueError(f"Unsupported EXPORT_ANIMATION_FORMAT: {export_fmt}")

    #     print(f"[Animation] export done: {export_path}")

    return fig, axes, ani

