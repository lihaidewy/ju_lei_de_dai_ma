import numpy as np
import pandas as pd
from pathlib import Path

RAW  = ["Frame","V","R","A","SNR"]
COLS = RAW + ["X","Y"]

def load_data(path, H=6.0, A_is_degree=True, save_csv=True, out_path=None):
    path = Path(path)
    df = (pd.read_csv(path, header=None) if path.suffix.lower()==".csv"
          else pd.read_excel(path, engine="openpyxl"))
    df.columns = RAW

    # 强制转数值
    for c in RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Frame","R","A"])
    df["Frame"] = df["Frame"].astype(int)

    # 角度 -> 弧度
    a = np.deg2rad(df["A"].to_numpy()) if A_is_degree else df["A"].to_numpy()

    # 斜距R -> 水平距离D
    r = df["R"].to_numpy()
    d = np.sqrt(np.maximum(r*r - H*H, 0.0))

    # 计算 X, Y
    df["X"] = d * np.cos(a)
    df["Y"] = d * np.sin(a)

    # 保存新增了X/Y的CSV
    if save_csv:
        if out_path is None:
            out_path = path.with_name(f"{path.stem}_withXY.csv")
        else:
            out_path = Path(out_path)
        df[COLS].to_csv(out_path, index=False, encoding="utf-8-sig")
        print("Saved:", out_path)

    frame_data = {
        int(k): {c: g[c].to_numpy() for c in COLS}
        for k, g in df[COLS].groupby("Frame", sort=True)
    }
    print("Frames:", sorted(frame_data))
    return frame_data
