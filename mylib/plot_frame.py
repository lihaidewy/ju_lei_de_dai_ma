import matplotlib.pyplot as plt

def plot_frame(frame_data, fid=0, xlim=(-60, 60), ylim=(0, 400), s=10):
    x = frame_data[fid]["X"]
    y = frame_data[fid]["Y"]
    v = frame_data[fid]["V"]   # 速度作为颜色

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(x, y, c=v, s=s)
    plt.colorbar(sc, label="V(m/s)")

    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Frame {fid}")
    plt.grid(True, alpha=0.3)
    plt.show()
