import numpy as np
from mylib.mydbscan_vel import mydbscan_ellipse_vel

def cluster_frame_dbscan(frame_data, fid, eps_x, eps_y, eps_v, min_pts):
    x = frame_data[fid]["X"]
    y = frame_data[fid]["Y"]
    v = frame_data[fid]["V"]

    points = np.column_stack([x, y])
    labels, core_mask = mydbscan_ellipse_vel(points, v, eps_x, eps_y, eps_v, min_pts)
    return labels
