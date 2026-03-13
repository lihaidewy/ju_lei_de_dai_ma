import numpy as np
from mylib.load_data2 import load_data

from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from mylib.tracker import track_across_frames
import matplotlib.pyplot as plt

def plot_tracks_xy(tracks, xlim=(0, 400), ylim=(-40, 40), title="Tracks (X-Y)"):
    plt.figure(figsize=(8, 7))
    for tr in tracks:
        xy = np.array([(h[1], h[2]) for h in tr.history], dtype=float)
        if xy.shape[0] < 2:
            continue
        plt.plot(xy[:, 0], xy[:, 1], marker="o", linewidth=1)
        plt.text(xy[-1, 0], xy[-1, 1], str(tr.tid), fontsize=8)  # 末端标注track id

    plt.xlim(*xlim); plt.ylim(*ylim)
    plt.xlabel("X (m)"); plt.ylabel("Y (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

path = "a.csv"
# path = "数据.xlsx"

frame_data = load_data(path)

frame_ids = sorted(frame_data.keys())[:200]

# eps_x, eps_y, eps_v, min_pts = 2.0, 4.0, 1.5, 2
eps_x, eps_y, eps_v, min_pts = 4.0, 1.5, 1.5, 2
def get_labels(fid):
    out = cluster_frame_dbscan(frame_data, fid, eps_x=eps_x, eps_y=eps_y, eps_v=eps_v, min_pts=min_pts)
    return out[0] if isinstance(out, tuple) else out

real_tracks, ghost_tracks = track_across_frames(
    frame_data, frame_ids,
    get_labels_fn=get_labels,
    gate_dist=3.0, max_misses=2, min_hits=3, dt=1.0
)

print("real tracks:", len(real_tracks), "ghost tracks:", len(ghost_tracks))


# 画真实轨迹
plot_tracks_xy(real_tracks, title="Real tracks")
plot_tracks_xy(ghost_tracks, title="Ghost tracks")