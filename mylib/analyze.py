import numpy as np
import pandas as pd

def analyze_max_snr_point_in_clusters(frame_data, fid, labels):
    """
    frame_data: load_data() 的输出 dict
    fid: 帧号
    labels: 该帧聚类标签 (N,)
    return: pandas DataFrame，每行对应一个 cluster
    """
    d = frame_data[fid]
    labels = np.asarray(labels).reshape(-1)

    R = np.asarray(d["R"])
    SNR = np.asarray(d["SNR"])
    X = np.asarray(d["X"])
    Y = np.asarray(d["Y"])
    V = np.asarray(d["V"])

    rows = []
    cluster_ids = np.unique(labels[labels > 0])

    for cid in cluster_ids:
        mask = labels == cid
        idxs = np.where(mask)[0]

        # 该簇最大 SNR 点（若并列取第一个）
        local_i = np.argmax(SNR[mask])
        i = idxs[local_i]

        # 该簇内 Range 分布位置
        r_cluster = R[mask]
        r0 = R[i]

        r_min = float(r_cluster.min())
        r_max = float(r_cluster.max())

        # 排名（从小到大）：1..N
        # 这里用“严格小于”的个数 +1，简单稳定
        rank = int(np.sum(r_cluster < r0) + 1)
        npts = int(r_cluster.size)

        # 百分位（0~100）：有多少点的 R <= r0
        percentile = float(np.mean(r_cluster <= r0) * 100.0)

        # 归一化位置（0~1），0=最小range，1=最大range
        norm_pos = float((r0 - r_min) / (r_max - r_min)) if (r_max > r_min) else 0.0

        # 解释性标签
        if np.isclose(r0, r_min):
            r_pos_tag = "near(min R)"
        elif np.isclose(r0, r_max):
            r_pos_tag = "far(max R)"
        else:
            r_pos_tag = "middle"

        rows.append({
            "Frame": fid,
            "ClusterID": int(cid),
            "N": npts,

            "MaxSNR": float(SNR[i]),
            "IndexInFrame": int(i),

            "X": float(X[i]),
            "Y": float(Y[i]),
            "R": float(R[i]),
            "V": float(V[i]),

            "R_min": r_min,
            "R_max": r_max,
            "R_rank(1=smallest)": rank,
            "R_percentile": percentile,
            "R_norm_pos(0~1)": norm_pos,
            "R_position": r_pos_tag
        })

    return pd.DataFrame(rows).sort_values(["Frame", "ClusterID"]).reset_index(drop=True)

def get_max_snr_indices(frame_data, fid, labels):
    d = frame_data[fid]
    labels = np.asarray(labels).reshape(-1)
    snr = np.asarray(d["SNR"])

    star_idx = []
    for cid in np.unique(labels[labels > 0]):
        mask = labels == cid
        idxs = np.where(mask)[0]
        i = idxs[np.argmax(snr[mask])]
        star_idx.append(int(i))
    return star_idx