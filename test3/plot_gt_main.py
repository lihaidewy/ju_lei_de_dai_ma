import pandas as pd
import numpy as np

def load_gt_reference(reference_path, H=6.0):
    """
    reference.csv: 含 Frame/ID/R/A（或中文列名）

    使用与 MATLAB 一致的坐标解算：
        X = -R * sin(A)
        Y = sqrt((R*cos(A))^2 - H^2)

    坐标系定义：
        X : 横向
        Y : 前向
    """

    # 跳过第一行，并手动指定列名
    gt = pd.read_csv(reference_path, skiprows=1, header=None)
    gt.columns = ["Frame", "ID","V", "R", "A","YAW"]

    for c in ["Frame", "ID","V", "R", "A","YAW"]:
        gt[c] = pd.to_numeric(gt[c], errors="coerce")

    gt = gt.dropna(subset=["Frame", "ID", "R", "A"])
    gt["Frame"] = gt["Frame"].astype(int)
    gt["ID"] = gt["ID"].astype(int)

    # 角度转弧度
    a = np.deg2rad(gt["A"].values)

    r = gt["R"].values

    # 与 MATLAB 相同的解算
    x = -r * np.sin(a)

    y_sq = (r * np.cos(a))**2 - H**2
    y = np.sqrt(np.maximum(y_sq, 0.0))

    gt["X"] = x
    gt["Y"] = y

    # 车型分类
    gt["model"] = gt["ID"] % 3

    return gt
