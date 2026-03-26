import numpy as np
import pandas as pd
from pathlib import Path

RAW = ["Frame", "V", "R", "A", "SNR"]
COLS = RAW + ["X", "Y"]


def load_data(path, H=7.0, A_is_degree=True, save_csv=True, out_path=None):
    path = Path(path)

    df = (
        pd.read_csv(path, header=None, skiprows=1)
        if path.suffix.lower() == ".csv"
        else pd.read_excel(path, header=None, skiprows=1, engine="openpyxl")
    )
    df.columns = RAW

    # 强制转数值
    for c in RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Frame", "R", "A"])
    df["Frame"] = df["Frame"].astype(int)

    # 角度转弧度
    a = np.deg2rad(df["A"].values) if A_is_degree else df["A"].values

    # 原始斜距
    r = df["R"].values

    # 按 MATLAB 公式计算
    # x = -dist * sin(ang)
    # y = sqrt((dist * cos(ang))^2 - H^2)
    x = -r * np.sin(a)
    y_sq = (r * np.cos(a)) ** 2 - H ** 2
    y = np.sqrt(np.maximum(y_sq, 0.0))

    df["X"] = x
    df["Y"] = y

    # 保存新增了 X/Y 的 CSV
    if save_csv:
        if out_path is None:
            out_path = path.with_name(f"{path.stem}_withXY.csv")
        else:
            out_path = Path(out_path)

        df[COLS].to_csv(out_path, index=False, encoding="utf-8-sig")
        print("Saved:", out_path)

    frame_data = {
        int(k): g[COLS].copy().reset_index(drop=True)
        for k, g in df.groupby("Frame", sort=True)
    }

    print("Frames:", sorted(frame_data))
    return frame_data
