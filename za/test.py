from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path

from mylib.plot2 import plot_raw_and_clusters
from mylib.cluster_frame_dbscan import cluster_frame_dbscan
from mylib.load_data2 import load_data

# =========================
# 参数
path = "radar.csv"
# path = "radar2.csv"
# DBSCAN参数（与你原来一致）
EPS_X = 4.0
EPS_Y = 1.5
EPS_V = 1.0
MIN_PTS = 2

# =========================
# 读取数据（带X/Y）
frame_data = load_data(path)
frame_ids = sorted(frame_data.keys())
n_frames = len(frame_ids)

# =========================
# 1) 预先对所有帧做聚类，把Label存入结构体
# 2) 拼接成新表，输出CSV
rows = []
for fid in frame_ids:
    labels = cluster_frame_dbscan(
        frame_data, fid,
        eps_x=EPS_X, eps_y=EPS_Y, eps_v=EPS_V, min_pts=MIN_PTS
    )

    # 存入结构体（每一帧新增Label数组）
    frame_data[fid]["Label"] = labels

    # 输出用的DataFrame（每帧一段）
    gdf = pd.DataFrame({
        "Frame": frame_data[fid]["Frame"],
        "V": frame_data[fid]["V"],
        "R": frame_data[fid]["R"],
        "A": frame_data[fid]["A"],
        "SNR": frame_data[fid]["SNR"],
        "X": frame_data[fid]["X"],
        "Y": frame_data[fid]["Y"],
        "Label": labels,
    })
    rows.append(gdf)

df_out = pd.concat(rows, ignore_index=True)

out_path = Path(path).with_name(f"{Path(path).stem}_withXY_labels.csv")
df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
print("Saved:", out_path)

# =========================
# 可视化（动画 + slider + button）
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
plt.subplots_adjust(bottom=0.8)

state = {"playing": True, "i": 0, "lock": False}

def update(i):
    if state["playing"]:
        state["i"] = i
    else:
        i = state["i"]

    fid = frame_ids[i]
    x = frame_data[fid]["X"]
    y = frame_data[fid]["Y"]
    v = frame_data[fid]["V"]
    pts = np.column_stack([x, y])

    # 直接读缓存的标签（不再每次聚类）
    labels = frame_data[fid]["Label"]

    plot_raw_and_clusters(
        pts, v, labels,
        xlim=(-40, 40), ylim=(0, 300),
        title=f"Frame {fid}",
        show=False,
        fig=fig, axes=axes, clear=True
    )

    if state["playing"]:
        state["lock"] = True
        frame_slider.set_val(i)
        state["lock"] = False

    return []

ani = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False, repeat=True)

# =========================
# 进度条 Slider
ax_slider = plt.axes([0.18, 0.0, 0.62, 0.04])  # [left, bottom, width, height]
frame_slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

def on_slider(val):
    if state["lock"]:
        return
    state["i"] = int(val)
    update(state["i"])
    fig.canvas.draw_idle()

frame_slider.on_changed(on_slider)

# 播放/暂停 Button
ax_btn = plt.axes([0.82, 0.0, 0.12, 0.06])
btn = Button(ax_btn, "Pause")  # 初始 playing=True

def on_btn(event):
    state["playing"] = not state["playing"]
    btn.label.set_text("Pause" if state["playing"] else "Play")
    fig.canvas.draw_idle()

btn.on_clicked(on_btn)

plt.show()
