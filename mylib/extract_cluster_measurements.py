import numpy as np

def extract_cluster_measurements(frame_data, fid, labels):
    """
    返回该帧所有簇的观测列表，每个观测是一个 dict：
    {cid, x, y, v, n, snr_mean}
    """
    d = frame_data[fid]
    x = d["X"]; y = d["Y"]; v = d["V"]; snr = d["SNR"]
    labels = np.asarray(labels).reshape(-1)

    meas = []
    for cid in np.unique(labels[labels > 0]):
        m = labels == cid
        meas.append({
            "cid": int(cid),
            "x": float(np.mean(x[m])),
            "y": float(np.mean(y[m])),
            "v": float(np.mean(v[m])),
            "n": int(np.sum(m)),
            "snr_mean": float(np.mean(snr[m])),
        })
    return meas
