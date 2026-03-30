import numpy as np
from geometry import world_to_local, local_to_world


def resolve_target_side_geometry(center_x, center_y, length, width, yaw_rad):
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
    best_midpoint_world = None

    for side in candidate_sides:
        if side["const_axis"] == "y":
            local_mid = np.array([[0.0, side["const_val"]]], dtype=float)
        else:
            local_mid = np.array([[side["const_val"], 0.0]], dtype=float)

        world_mid = local_to_world(local_mid, center_x, center_y, yaw_rad)[0]
        abs_y = abs(world_mid[1])

        if best_abs_y is None or abs_y < best_abs_y:
            best_abs_y = abs_y
            best_side = dict(side)
            best_midpoint_world = (float(world_mid[0]), float(world_mid[1]))

    const_axis = best_side["const_axis"]
    const_val = best_side["const_val"]

    if const_axis == "y":
        local_plus = np.array([[0.0, const_val + 1e-3]], dtype=float)
        local_minus = np.array([[0.0, const_val - 1e-3]], dtype=float)
    else:
        local_plus = np.array([[const_val + 1e-3, 0.0]], dtype=float)
        local_minus = np.array([[const_val - 1e-3, 0.0]], dtype=float)

    world_plus = local_to_world(local_plus, center_x, center_y, yaw_rad)[0]
    world_minus = local_to_world(local_minus, center_x, center_y, yaw_rad)[0]
    outward_sign = +1 if abs(world_plus[1]) < abs(world_minus[1]) else -1

    return {
        "side_info": best_side,
        "outward_sign": outward_sign,
        "midpoint_world": best_midpoint_world,
    }


def compute_side_band_range(side_info, outward_sign, outer_margin=0.5, inner_margin=0.1):
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
        side_info, outward_sign, outer_margin, inner_margin
    )

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
    side_info,
    outward_sign,
    outer_margin=1.0,
    inner_margin=1.0,
):
    span_min = side_info["span_min"]
    span_max = side_info["span_max"]
    const_axis = side_info["const_axis"]

    band_min, band_max = compute_side_band_range(
        side_info, outward_sign, outer_margin, inner_margin
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
