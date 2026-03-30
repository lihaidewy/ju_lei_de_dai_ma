import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from config import Config
from data_pipeline import load_all_data
from online_tracker import KalmanTrackCV


# =====================================
# 参数区
# =====================================
START_FRAME = 80
END_FRAME = 129

# 不对称带状区域
ROI_OUTER = 1.0   # 朝 x 轴方向扩
ROI_INNER = 1.0   # 朝框内部扩

# GT 尺寸先验
_GT_DIM = {
    0: {"L": 5.06, "W": 2.22},
    1: {"L": 4.32, "W": 2.19},
    2: {"L": 3.55, "W": 2.58},
}

# Kalman CV 参数
KF_DT = 1.0
KF_Q_POS = 0.30
KF_Q_VEL = 0.50
KF_R_POS = 1.50
KF_INIT_POS_VAR = 4.0
KF_INIT_VEL_VAR = 9.0

# 至少需要多少个 ROI 点才认为当前帧有有效量测
MIN_ROI_POINTS = 1


# =====================================
# 基础几何函数
# =====================================
def rotation_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def world_to_local(points_xy, center_x, center_y, yaw_rad):
    pts = np.asarray(points_xy, dtype=float)
    shifted = pts - np.array([center_x, center_y], dtype=float)

    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)

    local_x = shifted[:, 0] * c + shifted[:, 1] * s
    local_y = -shifted[:, 0] * s + shifted[:, 1] * c
    return np.column_stack([local_x, local_y])


def local_to_world(points_xy, center_x, center_y, yaw_rad):
    pts = np.asarray(points_xy, dtype=float)
    rot = rotation_matrix(yaw_rad)
    world = pts @ rot.T
    world[:, 0] += center_x
    world[:, 1] += center_y
    return world


def make_box_corners(center_x, center_y, length, width, yaw_rad):
    half_l = length / 2.0
    half_w = width / 2.0

    local_corners = np.array([
        [-half_w, -half_l],
        [ half_w, -half_l],
        [ half_w,  half_l],
        [-half_w,  half_l],
    ], dtype=float)

    return local_to_world(local_corners, center_x, center_y, yaw_rad)


def build_valid_frames(radar_data, gt_df, start_frame, end_frame):
    target_frames = list(range(start_frame, end_frame + 1))
    gt_frames = set(gt_df["Frame"].astype(int).unique())
    valid_frames = [fid for fid in target_frames if fid in radar_data and fid in gt_frames]
    return valid_frames


# =====================================
# ROI 相关
# =====================================
def get_side_closest_to_x_axis(center_x, center_y, length, width, yaw_rad):
    """
    找到 GT 框四条边里，离 x 轴(y=0)最近的那一条边
    """
    half_l = length / 2.0
    half_w = width / 2.0

    candidate_sides = [
        {"name": "bottom", "const_axis": "y", "const_val": -half_l, "span_min": -half_w, "span_max": half_w},
        {"name": "top",    "const_axis": "y", "const_val":  half_l, "span_min": -half_w, "span_max": half_w},
        {"name": "left",   "const_axis": "x", "const_val": -half_w, "span_min": -half_l, "span_max": half_l},
        {"name": "right",  "const_axis": "x", "const_val":  half_w, "span_min": -half_l, "span_max": half_l},
    ]

    best_side = None
    best_abs_y = None

    for side in candidate_sides:
        if side["const_axis"] == "y":
            local_mid = np.array([[0.0, side["const_val"]]], dtype=float)
        else:
            local_mid = np.array([[side["const_val"], 0.0]], dtype=float)

        world_mid = local_to_world(local_mid, center_x, center_y, yaw_rad)[0]
        abs_y = abs(world_mid[1])

        if best_abs_y is None or abs_y < best_abs_y:
            best_abs_y = abs_y
            best_side = side

    return best_side


def get_outward_sign_toward_x_axis(center_x, center_y, yaw_rad, const_axis, const_val):
    """
    判断选中边的哪一侧更靠近 x 轴(y=0)
    返回：
      +1: 局部正方向更靠近 x 轴
      -1: 局部负方向更靠近 x 轴
    """
    if const_axis == "y":
        local_plus = np.array([[0.0, const_val + 1e-3]], dtype=float)
        local_minus = np.array([[0.0, const_val - 1e-3]], dtype=float)
    else:
        local_plus = np.array([[const_val + 1e-3, 0.0]], dtype=float)
        local_minus = np.array([[const_val - 1e-3, 0.0]], dtype=float)

    world_plus = local_to_world(local_plus, center_x, center_y, yaw_rad)[0]
    world_minus = local_to_world(local_minus, center_x, center_y, yaw_rad)[0]

    return +1 if abs(world_plus[1]) < abs(world_minus[1]) else -1


def get_roi_points(
    meas_df,
    gx,
    gy,
    yaw_rad,
    side_info,
    outward_sign,
    outer_margin=0.5,
    inner_margin=0.1,
):
    """
    返回落在不对称带状 ROI 内的点
    """
    if len(meas_df) == 0:
        return np.empty((0, 2), dtype=float), np.zeros(0, dtype=bool)

    pts = meas_df[["X", "Y"]].values
    local_pts = world_to_local(pts, gx, gy, yaw_rad)

    const_axis = side_info["const_axis"]
    const_val = side_info["const_val"]
    span_min = side_info["span_min"]
    span_max = side_info["span_max"]

    low = const_val - inner_margin * outward_sign
    high = const_val + outer_margin * outward_sign

    band_min = min(low, high)
    band_max = max(low, high)

    if const_axis == "y":
        mask = (
            (local_pts[:, 0] >= span_min) &
            (local_pts[:, 0] <= span_max) &
            (local_pts[:, 1] >= band_min) &
            (local_pts[:, 1] <= band_max)
        )
    else:
        mask = (
            (local_pts[:, 1] >= span_min) &
            (local_pts[:, 1] <= span_max) &
            (local_pts[:, 0] >= band_min) &
            (local_pts[:, 0] <= band_max)
        )

    return pts[mask], mask


def make_side_band_polygon(
    center_x,
    center_y,
    yaw_rad,
    const_axis,
    const_val,
    span_min,
    span_max,
    outward_sign,
    outer_margin=0.5,
    inner_margin=0.1,
):
    low = const_val - inner_margin * outward_sign
    high = const_val + outer_margin * outward_sign

    band_min = min(low, high)
    band_max = max(low, high)

    if const_axis == "y":
        local_corners = np.array([
            [span_min, band_min],
            [span_max, band_min],
            [span_max, band_max],
            [span_min, band_max],
        ], dtype=float)
    else:
        local_corners = np.array([
            [band_min, span_min],
            [band_max, span_min],
            [band_max, span_max],
            [band_min, span_max],
        ], dtype=float)

    return local_to_world(local_corners, center_x, center_y, yaw_rad)


def get_side_midpoint_world(gx, gy, yaw_rad, side_info):
    """
    获取选中边中点的世界坐标
    """
    if side_info["const_axis"] == "y":
        local_mid = np.array([[0.0, side_info["const_val"]]], dtype=float)
    else:
        local_mid = np.array([[side_info["const_val"], 0.0]], dtype=float)

    world_mid = local_to_world(local_mid, gx, gy, yaw_rad)[0]
    return float(world_mid[0]), float(world_mid[1])


# =====================================
# Kalman 部分
# =====================================
def make_cv_track(track_id, init_center):
    return KalmanTrackCV(
        track_id=track_id,
        center=init_center,
        dt=KF_DT,
        q_pos=KF_Q_POS,
        q_vel=KF_Q_VEL,
        r_pos=KF_R_POS,
        init_pos_var=KF_INIT_POS_VAR,
        init_vel_var=KF_INIT_VEL_VAR,
        enable_output_ema=False,
        use_adaptive_r=False,
        use_quality_aware_r=False,
    )


def measurement_from_roi_points(roi_pts):
    """
    用 ROI 点的中位数作为当前帧量测
    """
    if roi_pts.shape[0] == 0:
        return None
    z = np.median(roi_pts, axis=0)
    return np.asarray(z, dtype=float).reshape(2)


# =====================================
# 主逻辑
# =====================================
def run_tail_kf_y_estimation(radar_data, gt_df, frame_ids):
    """
    对每个 gid 建一个 CV-KF，使用 ROI 内点的中位数做量测
    输出：
      filtered_y 与 GT 选中边中点 y_gt 的比较
    同时缓存可视化数据
    """
    tracks = {}   # gid -> KalmanTrackCV
    rows = []
    frame_cache = {}

    for fid in frame_ids:
        meas_df = radar_data[fid].copy()
        gt_frame_df = gt_df[gt_df["Frame"].astype(int) == fid].copy()

        print(f"\n===== Frame {fid} =====")

        vis_targets = []

        for gt_row in gt_frame_df.itertuples(index=False):
            gid = int(gt_row.ID)
            model = int(gt_row.model)

            if model not in _GT_DIM:
                continue

            gx = float(gt_row.X)
            gy = float(gt_row.Y)

            yaw_deg = float(gt_row.YAW) if hasattr(gt_row, "YAW") and gt_row.YAW is not None else 0.0
            yaw_rad = math.radians(yaw_deg)
            # 如果你的 YAW 本身就是弧度，改成：
            # yaw_rad = float(gt_row.YAW) if hasattr(gt_row, "YAW") and gt_row.YAW is not None else 0.0

            length = float(_GT_DIM[model]["L"])
            width = float(_GT_DIM[model]["W"])

            side_info = get_side_closest_to_x_axis(
                center_x=gx,
                center_y=gy,
                length=length,
                width=width,
                yaw_rad=yaw_rad,
            )

            outward_sign = get_outward_sign_toward_x_axis(
                center_x=gx,
                center_y=gy,
                yaw_rad=yaw_rad,
                const_axis=side_info["const_axis"],
                const_val=side_info["const_val"],
            )

            roi_pts, roi_mask = get_roi_points(
                meas_df=meas_df,
                gx=gx,
                gy=gy,
                yaw_rad=yaw_rad,
                side_info=side_info,
                outward_sign=outward_sign,
                outer_margin=ROI_OUTER,
                inner_margin=ROI_INNER,
            )

            n_roi = int(roi_pts.shape[0])
            z = measurement_from_roi_points(roi_pts) if n_roi >= MIN_ROI_POINTS else None

            # GT 选中边中点真值
            x_gt, y_gt = get_side_midpoint_world(gx, gy, yaw_rad, side_info)

            # Kalman 跟踪
            if gid not in tracks:
                if z is not None:
                    tracks[gid] = make_cv_track(track_id=gid, init_center=z)
                    filtered = tracks[gid].output_center.copy()
                    used_measurement = 1
                else:
                    filtered = None
                    used_measurement = 0
            else:
                track = tracks[gid]
                track.predict()

                if z is not None:
                    filtered = track.update(z).copy()
                    used_measurement = 1
                else:
                    track.mark_missed()
                    filtered = track.output_center.copy()
                    used_measurement = 0

            if filtered is None:
                x_hat = np.nan
                y_hat = np.nan
                y_error = np.nan
                abs_y_error = np.nan
                fit_ok = 0
            else:
                x_hat = float(filtered[0])
                y_hat = float(filtered[1])
                y_error = y_hat - y_gt
                abs_y_error = abs(y_error)
                fit_ok = 1

            rows.append({
                "Frame": fid,
                "gid": gid,
                "model": model,
                "side": side_info["name"],
                "n_roi": n_roi,
                "used_measurement": used_measurement,
                "track_exists": 1 if gid in tracks else 0,
                "x_gt": x_gt,
                "y_gt": y_gt,
                "x_hat": x_hat,
                "y_hat": y_hat,
                "y_error": y_error,
                "abs_y_error": abs_y_error,
                "ok": fit_ok,
            })

            vis_targets.append({
                "gid": gid,
                "model": model,
                "gx": gx,
                "gy": gy,
                "yaw_rad": yaw_rad,
                "length": length,
                "width": width,
                "side_info": side_info,
                "outward_sign": outward_sign,
                "roi_pts": roi_pts,
                "n_roi": n_roi,
                "used_measurement": used_measurement,
                "x_gt": x_gt,
                "y_gt": y_gt,
                "x_hat": x_hat,
                "y_hat": y_hat,
                "ok": fit_ok,
            })

            if np.isfinite(y_hat):
                print(
                    f"gid={gid}, side={side_info['name']}, "
                    f"n_roi={n_roi}, meas_used={used_measurement}, "
                    f"y_gt={y_gt:.3f}, y_hat={y_hat:.3f}, err={y_error:.3f}"
                )
            else:
                print(
                    f"gid={gid}, side={side_info['name']}, "
                    f"n_roi={n_roi}, meas_used={used_measurement}, "
                    f"y_gt={y_gt:.3f}, y_hat=NaN"
                )

        frame_cache[fid] = {
            "meas_df": meas_df,
            "gt_frame_df": gt_frame_df,
            "targets": vis_targets,
        }

    return pd.DataFrame(rows), frame_cache


def print_summary(df):
    print("\n===== 尾部 ROI 点 + Kalman CV 滤波结果 =====")

    if df.empty:
        print("没有结果。")
        return

    ok_df = df[df["ok"] == 1].copy()

    print(f"\n总记录数: {len(df)}")
    print(f"有效估计数: {len(ok_df)}")

    if len(ok_df) == 0:
        print("没有有效估计。")
        return

    mae = ok_df["abs_y_error"].mean()
    medae = ok_df["abs_y_error"].median()
    rmse = np.sqrt(np.mean(ok_df["y_error"].values ** 2))
    bias = ok_df["y_error"].mean()

    print("\n===== 全局误差统计 =====")
    print(f"MAE   = {mae:.4f} m")
    print(f"MedAE = {medae:.4f} m")
    print(f"RMSE  = {rmse:.4f} m")
    print(f"Bias  = {bias:.4f} m")

    print("\n===== 每帧平均绝对误差 =====")
    frame_group = ok_df.groupby("Frame")["abs_y_error"].mean()
    for fid, val in frame_group.items():
        print(f"Frame {fid}: mean_abs_y_error = {val:.4f} m")


# =====================================
# 可视化
# =====================================
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


def render_frame(fig, axes, frame_ids, state, frame_cache):
    fid = frame_ids[state["idx"]]
    cache = frame_cache[fid]

    meas_df = cache["meas_df"]
    gt_frame_df = cache["gt_frame_df"]
    targets = cache["targets"]

    ax_left, ax_right = axes
    ax_left.clear()
    ax_right.clear()

    # 左图：原始量测
    if len(meas_df) > 0:
        ax_left.scatter(
            meas_df["X"].values,
            meas_df["Y"].values,
            s=12,
            alpha=0.75,
            color="blue"
        )

    # 右图：原始量测作为背景
    if len(meas_df) > 0:
        ax_right.scatter(
            meas_df["X"].values,
            meas_df["Y"].values,
            s=12,
            alpha=0.35,
            color="gray"
        )

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

        # 左图：GT 框
        box_corners = make_box_corners(gx, gy, length, width, yaw_rad)
        ax_left.add_patch(
            Polygon(
                box_corners,
                closed=True,
                fill=False,
                linewidth=2.0,
                edgecolor="green"
            )
        )
        ax_left.scatter(gx, gy, marker="x", s=60, linewidths=2, color="green")
        ax_left.text(gx + 0.08, gy + 0.08, f"ID={t['gid']}", fontsize=9, color="green")

        # 左图：ROI 带状区域
        band_corners = make_side_band_polygon(
            center_x=gx,
            center_y=gy,
            yaw_rad=yaw_rad,
            const_axis=side_info["const_axis"],
            const_val=side_info["const_val"],
            span_min=side_info["span_min"],
            span_max=side_info["span_max"],
            outward_sign=outward_sign,
            outer_margin=ROI_OUTER,
            inner_margin=ROI_INNER,
        )
        ax_left.add_patch(
            Polygon(
                band_corners,
                closed=True,
                fill=False,
                linewidth=1.8,
                edgecolor="orange",
                linestyle="--"
            )
        )

        # 左图：ROI 点高亮
        if roi_pts.shape[0] > 0:
            ax_left.scatter(
                roi_pts[:, 0],
                roi_pts[:, 1],
                s=34,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=1.2
            )

        # 右图：ROI 点
        if roi_pts.shape[0] > 0:
            ax_right.scatter(
                roi_pts[:, 0],
                roi_pts[:, 1],
                s=34,
                marker="o",
                facecolors="none",
                edgecolors="orange",
                linewidths=1.2
            )

        # 右图：GT 边中点
        if np.isfinite(t["x_gt"]) and np.isfinite(t["y_gt"]):
            ax_right.scatter(
                t["x_gt"], t["y_gt"],
                marker="x", s=90, linewidths=2, color="green"
            )
            ax_right.text(
                t["x_gt"] + 0.08, t["y_gt"] + 0.08,
                f"GT-{t['gid']}",
                fontsize=9, color="green"
            )

        # 右图：KF 估计点
        if t["ok"] == 1 and np.isfinite(t["x_hat"]) and np.isfinite(t["y_hat"]):
            ok_count += 1
            ax_right.scatter(
                t["x_hat"], t["y_hat"],
                marker="x", s=90, linewidths=2, color="red"
            )
            ax_right.text(
                t["x_hat"] + 0.08, t["y_hat"] + 0.08,
                f"KF-{t['gid']}",
                fontsize=9, color="red"
            )

    ax_left.set_title(
        f"Frame {fid} | GT Boxes + ROI + ROI Points\n"
        f"meas={len(meas_df)}, gt={len(gt_frame_df)}"
    )
    ax_left.set_xlabel("X")
    ax_left.set_ylabel("Y")
    ax_left.grid(True)
    ax_left.set_aspect("equal", adjustable="box")

    ax_right.set_title(
        f"Frame {fid} | Tail ROI Points + KF Estimate vs GT Edge Midpoint\n"
        f"valid_estimates={ok_count}/{len(targets)}"
    )
    ax_right.set_xlabel("X")
    ax_right.set_ylabel("Y")
    ax_right.grid(True)
    ax_right.set_aspect("equal", adjustable="box")

    limits = compute_plot_limits(meas_df, gt_frame_df)
    if limits is not None:
        x0, x1, y0, y1 = limits
        ax_left.set_xlim(x0, x1)
        ax_left.set_ylim(y0, y1)
        ax_right.set_xlim(x0, x1)
        ax_right.set_ylim(y0, y1)

    fig.suptitle(
        f"Tail ROI + Kalman CV | Frames {frame_ids[0]}-{frame_ids[-1]} | outer={ROI_OUTER:.1f}m inner={ROI_INNER:.1f}m\n"
        "Keyboard: n=next, p=prev, q/esc=quit",
        fontsize=12
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.canvas.draw_idle()


def launch_viewer(frame_ids, frame_cache):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    state = {"idx": 0}

    def on_key(event):
        key = (event.key or "").lower()
        if key == "n":
            state["idx"] = min(state["idx"] + 1, len(frame_ids) - 1)
            render_frame(fig, axes, frame_ids, state, frame_cache)
        elif key == "p":
            state["idx"] = max(state["idx"] - 1, 0)
            render_frame(fig, axes, frame_ids, state, frame_cache)
        elif key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render_frame(fig, axes, frame_ids, state, frame_cache)
    plt.show()


def main():
    cfg = Config()

    radar_data, gt_df = load_all_data(cfg)

    frame_ids = build_valid_frames(radar_data, gt_df, START_FRAME, END_FRAME)
    if not frame_ids:
        raise ValueError(f"在 {START_FRAME} 到 {END_FRAME} 帧之间没有找到公共帧。")

    print("有效帧:")
    print(frame_ids)

    result_df, frame_cache = run_tail_kf_y_estimation(radar_data, gt_df, frame_ids)

    print_summary(result_df)

    launch_viewer(frame_ids, frame_cache)


if __name__ == "__main__":
    main()
