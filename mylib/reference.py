from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

RAW = ["Frame", "ID", "V", "R", "A"]  # 按你的数据列顺序
DIM = {
    0: {"L": 5.06, "W": 2.22, "H": 1.78},
    1: {"L": 4.32, "W": 2.19, "H": 1.44},
    2: {"L": 3.55, "W": 2.58, "H": 1.33},
}

def load_data(path, H=4.0, A_is_degree=True, save_csv=False, out_path=None):
    path = Path(path)
    df = (pd.read_csv(path, header=None) if path.suffix.lower()==".csv"
          else pd.read_excel(path, engine="openpyxl"))
    df.columns = RAW

    for c in RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Frame","R","A"])
    df["Frame"] = df["Frame"].astype(int)
    df["ID"] = df["ID"].astype(int)

    # A -> rad
    a = np.deg2rad(df["A"].to_numpy()) if A_is_degree else df["A"].to_numpy()

    # slant range R -> horizontal distance D
    r = df["R"].to_numpy()
    d = np.sqrt(np.maximum(r*r - H*H, 0.0))

    df["X"] = d * np.cos(a)
    df["Y"] = d * np.sin(a)

    # model = ID % 3
    df["model"] = df["ID"] % 3

    if save_csv:
        if out_path is None:
            out_path = path.with_name(path.stem + "_xy.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("saved:", out_path)

    return df


def rectangle_corners_fixed(cx, cy, L, W):
    """
    不旋转：长L沿X轴、宽W沿Y轴。返回四角点(4,2)。
    """
    halfL, halfW = L/2.0, W/2.0
    return np.array([
        [cx + halfL, cy + halfW],
        [cx - halfL, cy + halfW],
        [cx - halfL, cy - halfW],
        [cx + halfL, cy - halfW],
    ], dtype=float)


def plot_frame(df, frame_id, draw_id=True):
    g = df[df["Frame"] == frame_id]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # 雷达原点
    ax.scatter([0], [0], s=60, marker="x")

    for _, row in g.iterrows():
        cx, cy = float(row["X"]), float(row["Y"])
        m = int(row["model"])
        L, W = DIM[m]["L"], DIM[m]["W"]

        poly_xy = rectangle_corners_fixed(cx, cy, L, W)
        ax.add_patch(Polygon(poly_xy, closed=True, fill=False, linewidth=1.8))

        if draw_id:
            ax.text(cx, cy, str(int(row["ID"])), fontsize=8, ha="center", va="center")

    # # 自适应范围
    # if len(g) > 0:
    #     pad = 5.0
    #     ax.set_xlim(g["X"].min() - pad, g["X"].max() + pad)
    #     ax.set_ylim(g["Y"].min() - pad, g["Y"].max() + pad)
    # else:
    #     ax.set_xlim(-10, 10)
    #     ax.set_ylim(-10, 10)

    # 固定坐标轴范围
    ax.set_xlim(0, 400)
    ax.set_ylim(-40, 40)

    ax.set_title(f"Frame={frame_id}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.show()


# ===== 用法 =====
# df = load_data("/mnt/data/reference.csv", H=4.0, A_is_degree=True, save_csv=False)
# frames = np.sort(df["Frame"].unique())
# plot_frame(df, frames[0])
