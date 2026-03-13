import pandas as pd
import numpy as np

def load_data(path):
    """
    输出：
        frame_data     - list[dict]，每个元素对应一个帧的数据
        unique_frames  - 不同帧编号的 numpy 数组
        data           - 读取到的原始 DataFrame
    """

    # 读取 Excel
    data = pd.read_excel(path)

    # 按列取数据（假设 Excel 列顺序与 MATLAB 一致）
    R = data.iloc[:, 0].to_numpy()
    v = data.iloc[:, 1].to_numpy()
    A_deg = data.iloc[:, 2].to_numpy()
    SNR = data.iloc[:, 3].to_numpy()
    Frame = data.iloc[:, 4].to_numpy()
    Onframe = data.iloc[:, 5].to_numpy()
    X = data.iloc[:, 6].to_numpy()
    Y = data.iloc[:, 7].to_numpy()
    SNR_db = data.iloc[:, 8].to_numpy()

    # 不同帧编号
    unique_frames = np.unique(Frame)
    print(f"数据中共有 {len(unique_frames)} 个不同的帧")
    print("帧编号:", unique_frames)

    frame_data = []

    for frame_num in unique_frames:
        idx = (Frame == frame_num)

        frame_dict = {
            "frame_num": frame_num,
            "indices": np.where(idx)[0],   # 等价于 MATLAB 的 find
            "count": np.sum(idx),

            "R": R[idx],
            "v": v[idx],
            "A_deg": A_deg[idx],
            "SNR": SNR[idx],
            "Onframe": Onframe[idx],
            "X": X[idx],
            "Y": Y[idx],
            "SNR_db": SNR_db[idx]
        }

        frame_data.append(frame_dict)

        print(f"帧 {frame_num}: 有 {frame_dict['count']} 个数据点")

    return frame_data, unique_frames, data
