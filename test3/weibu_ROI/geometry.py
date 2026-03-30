import math
import numpy as np
"""几何相关的工具函数，主要是坐标变换和边界框计算。"""

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
