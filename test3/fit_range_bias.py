import numpy as np
import pandas as pd

df = pd.read_csv("data/tp_matches_for_bias.csv")

pred_y = df["pred_y"].values
dy = df["dy"].values

# 拟合 dy = a + b * pred_y
coef = np.polyfit(pred_y, dy, deg=1)

b = float(coef[0])
a = float(coef[1])

print("Learned linear bias model:")
print(f"bias_y = {a:.6f} + {b:.6f} * pred_y")
