import numpy as np
import pandas as pd
from mylib.cluster_frame_dbscan import cluster_frame_dbscan

def add_cluster_and_export(path_in, path_out,
                           eps_x=2.0, eps_y=4.0, eps_v=1.5, min_pts=2):
    # 读原始数据
    df = pd.read_excel(path_in, engine="openpyxl")
    df.columns = ["R","V","A","SNR","Frame","Onframe","X","Y","SNR_db"]

    # 新列：默认 -1（噪声/未聚类）
    df["Cluster"] = -1

    # 按 Frame 分组，把每帧 labels 写回原 df
    for fid, g in df.groupby("Frame", sort=True):
        idx = g.index.to_numpy()  # 该帧在原 df 中的行号
        pts = g[["X", "Y"]].to_numpy()
        vel = g["V"].to_numpy()

        # 跑聚类：兼容返回 (labels, core_mask) 或只返回 labels
        out = cluster_frame_dbscan(
            {int(fid): {"X": g["X"].to_numpy(), "Y": g["Y"].to_numpy(), "V": g["V"].to_numpy()}},
            int(fid),
            eps_x=eps_x, eps_y=eps_y, eps_v=eps_v, min_pts=min_pts
        )
        if isinstance(out, tuple):
            labels = out[0]
        else:
            labels = out

        df.loc[idx, "Cluster"] = labels

    # 输出
    df.to_excel(path_out, index=False, engine="openpyxl")
    return df
