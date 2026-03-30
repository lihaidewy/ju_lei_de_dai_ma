from config import Config
from data_pipeline import load_all_data, get_frame_ids, build_gt_list_for_frame
import matplotlib.pyplot as plt

def plot_one_frame(radar_data, gt_df, fid):
    meas_df = radar_data[fid].copy()
    gt_frame_df = gt_df[gt_df["Frame"].astype(int) == fid].copy()

    plt.figure(figsize=(8, 6))
    plt.scatter(meas_df["X"], meas_df["Y"], s=10, label="Measurement")
    plt.scatter(gt_frame_df["X"], gt_frame_df["Y"], s=60, marker="x", label="GT")
    plt.title(f"Frame {fid}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

class DummyArgs:
    max_frames = 1000

cfg = Config()

# 你自己指定想看的帧
cfg.FRAMES_TO_SHOW = [100, 101, 105]

# 读数据
radar_data, gt_df = load_all_data(cfg)

# 取公共帧（只保留你设定且同时存在于 radar / gt 的帧）
frame_ids = get_frame_ids(radar_data, gt_df, cfg, DummyArgs())

for fid in frame_ids:
    # 这一帧的量测数据
    meas_df = radar_data[fid]

    # 这一帧的 GT
    gt_list = build_gt_list_for_frame(gt_df, fid)
    gt_frame_df = gt_df[gt_df["Frame"] == fid].copy()

    print(f"\n===== Frame {fid} =====")
    print("量测数据:")
    print(meas_df.head())

    print("GT数据:")
    print(gt_frame_df[["Frame", "ID", "X", "Y", "model"]].head())

plot_one_frame(radar_data, gt_df, 120)
