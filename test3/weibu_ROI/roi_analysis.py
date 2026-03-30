import numpy as np
from geometry import world_to_local, local_to_world

"""
1. 只考虑车辆前后两个端边：top / bottom
2. 从这两个端边中选择“近端边”
3. 近端默认定义为：边中点到传感器原点 (0, 0) 的欧式距离更小
4. outward_sign 也按“离传感器更近的方向”为 +1 来确定
"""


def resolve_target_side_geometry(
    center_x,
    center_y,
    length,
    width,
    yaw_rad,
    sensor_x=0.0,
    sensor_y=0.0,
):
    """
    只在 top / bottom 两个端边中选择近端边。
    参数:
        center_x, center_y : 目标中心（世界坐标）
        length, width      : 目标尺寸
        yaw_rad            : 朝向（弧度）
        sensor_x, sensor_y : 传感器位置（世界坐标），默认原点
    返回:
        {
            "side_info": best_side,
            "outward_sign": outward_sign,
            "midpoint_world": best_midpoint_world,
        }
    """
    half_l = length / 2.0
    half_w = width / 2.0
    candidate_sides = [
        {
            "name": "bottom",
            "const_axis": "y",
            "const_val": -half_l,
            "span_min": -half_w,
            "span_max": half_w,
        },
        {
            "name": "top",
            "const_axis": "y",
            "const_val": half_l,
            "span_min": -half_w,
            "span_max": half_w,
        },
    ]

    best_side = None
    best_dist = None
    best_midpoint_world = None

    for side in candidate_sides:
        # 端边中点在局部坐标中一定是 (0, const_val)
        local_mid = np.array([[0.0, side["const_val"]]], dtype=float)
        world_mid = local_to_world(local_mid, center_x, center_y, yaw_rad)[0]

        dist = np.hypot(world_mid[0] - sensor_x, world_mid[1] - sensor_y)

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_side = dict(side)
            best_midpoint_world = (float(world_mid[0]), float(world_mid[1]))

    const_val = best_side["const_val"]

    local_plus = np.array([[0.0, const_val + 1e-3]], dtype=float)
    local_minus = np.array([[0.0, const_val - 1e-3]], dtype=float)

    world_plus = local_to_world(local_plus, center_x, center_y, yaw_rad)[0]
    world_minus = local_to_world(local_minus, center_x, center_y, yaw_rad)[0]

    dist_plus = np.hypot(world_plus[0] - sensor_x, world_plus[1] - sensor_y)
    dist_minus = np.hypot(world_minus[0] - sensor_x, world_minus[1] - sensor_y)

    outward_sign = +1 if dist_plus < dist_minus else -1

    return {
        "side_info": best_side,
        "outward_sign": outward_sign,
        "midpoint_world": best_midpoint_world,
    }


def compute_side_band_range(side_info, outward_sign, outer_margin=0.5, inner_margin=0.1):
    """
    计算带状 ROI 在局部坐标中沿法向的范围。
    """
    const_val = side_info["const_val"]
    low = const_val - inner_margin * outward_sign
    high = const_val + outer_margin * outward_sign
    return min(low, high), max(low, high)


def get_roi_points(
    meas_df,
    gx,
    gy,
    yaw_rad,
    side_info,
    outward_sign,
    outer_margin=1.0,
    inner_margin=1.0,
):

    if len(meas_df) == 0:
        return np.empty((0, 2), dtype=float), np.zeros(0, dtype=bool)

    pts = meas_df[["X", "Y"]].values
    local_pts = world_to_local(pts, gx, gy, yaw_rad)

    const_axis = side_info["const_axis"]
    span_min = side_info["span_min"]
    span_max = side_info["span_max"]

    band_min, band_max = compute_side_band_range(
        side_info=side_info,
        outward_sign=outward_sign,
        outer_margin=outer_margin,
        inner_margin=inner_margin,
    )

    if const_axis == "y":
        mask = (
            (local_pts[:, 0] >= span_min) &
            (local_pts[:, 0] <= span_max) &
            (local_pts[:, 1] >= band_min) &
            (local_pts[:, 1] <= band_max)
        )
    else:
        # 理论上这个分支现在不会走到，但保留以兼容接口
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
    side_info,
    outward_sign,
    outer_margin=1.0,
    inner_margin=1.0,
):
    """
    生成可视化 ROI 带的四边形顶点。
    """
    span_min = side_info["span_min"]
    span_max = side_info["span_max"]
    const_axis = side_info["const_axis"]

    band_min, band_max = compute_side_band_range(
        side_info=side_info,
        outward_sign=outward_sign,
        outer_margin=outer_margin,
        inner_margin=inner_margin,
    )

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
