import numpy as np
import pandas as pd

def load_data(path):
    df = pd.read_excel(path,engine='openpyxl')
    # print(df)
    df.columns = ["R","V","A","SNR","Frame","Onframe","X","Y","SNR_db"]
    cols = df.columns.tolist()
    frame_data = {
        int(fid): {c: g[c].to_numpy() for c in cols}
        for fid, g in df[cols].groupby("Frame", sort=True)
    }
    print("Frames:",sorted(frame_data.keys()))
    
    return frame_data



    # R = df.iloc[:, 0].to_numpy()
    # v = df.iloc[:, 1].to_numpy()
    # A_deg = df.iloc[:, 2].to_numpy()
    # SNR = df.iloc[:, 3].to_numpy()
    # Frame = df.iloc[:, 4].to_numpy()
    # Onframe = df.iloc[:, 5].to_numpy()
    # X = df.iloc[:, 6].to_numpy()
    # Y = df.iloc[:, 7].to_numpy()
    # SNR_db = df.iloc[:, 8].to_numpy()
    # unique_frames = np.unique(Frame)
    # print("帧编号:", unique_frames)


