import pandas as pd
import numpy as np

def load_gt_reference(reference_path, H=4.0):
    """
    reference.csv: 含 Frame/ID/R/A（或中文列名）
    用 D = sqrt(R^2 - H^2), X=D*cos(A), Y=D*sin(A)
    """
    gt = pd.read_csv(reference_path)

    # 兼容中文列名（你之前用过）
    gt = gt.rename(columns={
        "帧号": "Frame",
        "车辆ID": "ID",
        "距离(m)": "R",
        "角度(deg)": "A",
    })

    for c in ["Frame", "ID", "R", "A"]:
        gt[c] = pd.to_numeric(gt[c], errors="coerce")
    gt = gt.dropna(subset=["Frame", "ID", "R", "A"])
    gt["Frame"] = gt["Frame"].astype(int)
    gt["ID"] = gt["ID"].astype(int)

    a = np.deg2rad(gt["A"].to_numpy())
    r = gt["R"].to_numpy()
    d = np.sqrt(np.maximum(r*r - H*H, 0.0))

    gt["X"] = d * np.cos(a)
    gt["Y"] = d * np.sin(a)
    gt["model"] = gt["ID"] % 3
    return gt
