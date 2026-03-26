from mylib.reference import load_data,plot_frame
import numpy as np

df = load_data("reference.csv", H=4.0, A_is_degree=True, save_csv=False)

frames = np.sort(df["Frame"].unique())
print("总帧数:", len(frames))
print("前20个帧号:", frames[:20])

plot_frame(df, frames[9])
