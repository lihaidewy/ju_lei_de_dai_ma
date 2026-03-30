import math
import numpy as np
import pandas as pd

from config import Config
from data_pipeline import load_all_data


# =====================================
# 参数区
# =====================================
START_FRAME = 80
END_FRAME = 129

# 不对称带状区域
ROI_OUTER = 0.5   # 朝 x 轴方向扩
ROI_INNER = 0.1   # 朝框内部扩

# 最少拟合点数
MIN_POINTS_FOR_FIT = 2

# GT 尺寸先验
_GT_DIM = {
    0: {"L": 5.06, "W": 2.22},
    1: {"L": 4.32, "W": 2.19},
    2: {"L": 3.55, "W": 2.58},
}


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

    rot = rotation_matrix(yaw_rad)

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


def count_points_in_side_band(
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
    取出落在不对称带状 ROI 内的点
    """
    if len(meas_df) == 0:
        return 0, np.zeros(0, dtype=bool)

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

    return int(np.sum(mask)), mask


# =====================================
# 最小二乘拟合
# =====================================
def fit_line_least_squares(points_xy):
    """
    最小二乘拟合 y = a x + b
    返回:
      a, b
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    A = np.column_stack([x, np.ones_like(x)])
    theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = theta[0], theta[1]
    return float(a), float(b)


def get_side_midpoint_world(gx, gy, yaw_rad, side_info):
    """
    获取选中边的中点世界坐标
    """
    if side_info["const_axis"] == "y":
        local_mid = np.array([[0.0, side_info["const_val"]]], dtype=float)
    else:
        local_mid = np.array([[side_info["const_val"], 0.0]], dtype=float)

    world_mid = local_to_world(local_mid, gx, gy, yaw_rad)[0]
    return float(world_mid[0]), float(world_mid[1])


def process_one_target(fid, meas_df, gt_row):
    """
    对单个目标：
      1) 取 ROI 内点
      2) 最小二乘拟合 y = ax + b
      3) 用边中点 x_ref 估计 y_hat
      4) 与 GT 边中点 y_gt 比较
    """
    gid = int(gt_row.ID)
    model = int(gt_row.model)

    if model not in _GT_DIM:
        return None

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

    n_roi, roi_mask = count_points_in_side_band(
        meas_df=meas_df,
        gx=gx,
        gy=gy,
        yaw_rad=yaw_rad,
        side_info=side_info,
        outward_sign=outward_sign,
        outer_margin=ROI_OUTER,
        inner_margin=ROI_INNER,
    )

    if n_roi < MIN_POINTS_FOR_FIT:
        return {
            "Frame": fid,
            "gid": gid,
            "model": model,
            "side": side_info["name"],
            "n_roi": n_roi,
            "fit_ok": 0,
            "a": np.nan,
            "b": np.nan,
            "x_ref": np.nan,
            "y_gt": np.nan,
            "y_hat": np.nan,
            "y_error": np.nan,
            "abs_y_error": np.nan,
        }

    roi_pts = meas_df.loc[roi_mask, ["X", "Y"]].values

    # 最小二乘拟合
    a, b = fit_line_least_squares(roi_pts)

    # GT 边中点真值
    x_ref, y_gt = get_side_midpoint_world(gx, gy, yaw_rad, side_info)

    # 用拟合线估计该位置的 y
    y_hat = a * x_ref + b

    y_error = y_hat - y_gt
    abs_y_error = abs(y_error)

    return {
        "Frame": fid,
        "gid": gid,
        "model": model,
        "side": side_info["name"],
        "n_roi": n_roi,
        "fit_ok": 1,
        "a": a,
        "b": b,
        "x_ref": x_ref,
        "y_gt": y_gt,
        "y_hat": y_hat,
        "y_error": y_error,
        "abs_y_error": abs_y_error,
    }


def run_fit_all_frames(radar_data, gt_df, frame_ids):
    rows = []

    for fid in frame_ids:
        meas_df = radar_data[fid].copy()
        gt_frame_df = gt_df[gt_df["Frame"].astype(int) == fid].copy()

        for gt_row in gt_frame_df.itertuples(index=False):
            result = process_one_target(fid, meas_df, gt_row)
            if result is not None:
                rows.append(result)

    return pd.DataFrame(rows)


def print_summary(df):
    print("\n===== ROI 点最小二乘拟合 y 值结果 =====")

    if df.empty:
        print("没有结果。")
        return

    fit_df = df[df["fit_ok"] == 1].copy()
    fail_df = df[df["fit_ok"] == 0].copy()

    print(f"总目标数: {len(df)}")
    print(f"成功拟合数: {len(fit_df)}")
    print(f"拟合失败数(ROI点不足): {len(fail_df)}")

    if len(fit_df) > 0:
        mae = fit_df["abs_y_error"].mean()
        medae = fit_df["abs_y_error"].median()
        rmse = np.sqrt(np.mean(fit_df["y_error"].values ** 2))
        bias = fit_df["y_error"].mean()

        print("\n--- 误差统计（单位：m）---")
        print(f"MAE   = {mae:.4f}")
        print(f"MedAE = {medae:.4f}")
        print(f"RMSE  = {rmse:.4f}")
        print(f"Bias  = {bias:.4f}")

        print("\n--- 每帧每目标结果 ---")
        for row in fit_df.itertuples(index=False):
            print(
                f"Frame={row.Frame}, gid={row.gid}, side={row.side}, "
                f"n_roi={row.n_roi}, y_gt={row.y_gt:.4f}, y_hat={row.y_hat:.4f}, "
                f"y_error={row.y_error:.4f}"
            )
    else:
        print("没有成功拟合的目标。")


def main():
    cfg = Config()

    radar_data, gt_df = load_all_data(cfg)

    frame_ids = build_valid_frames(radar_data, gt_df, START_FRAME, END_FRAME)
    if not frame_ids:
        raise ValueError(f"在 {START_FRAME} 到 {END_FRAME} 帧之间没有找到公共帧。")

    print("有效帧:")
    print(frame_ids)

    result_df = run_fit_all_frames(radar_data, gt_df, frame_ids)

    print_summary(result_df)

    out_path = "roi_least_squares_y_compare.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
