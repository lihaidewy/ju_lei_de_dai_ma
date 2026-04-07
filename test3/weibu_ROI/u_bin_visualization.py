import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import matplotlib.pyplot as plt

# ===== 设置中文字体 =====
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


REQUIRED_PROB_COLS = {
    "u_bin",
    "u_left",
    "u_right",
    "n_points",
    "n_roi_points",
    "p_roi_given_u",
}

REQUIRED_POINT_COLS = {
    "u",
    "u_bin",
    "is_roi",
}


def _validate_prob_df(prob_df: pd.DataFrame) -> None:
    missing = REQUIRED_PROB_COLS.difference(prob_df.columns)
    if missing:
        raise ValueError(f"prob_df 缺少字段: {sorted(missing)}")


def _validate_point_df(point_df: pd.DataFrame) -> None:
    missing = REQUIRED_POINT_COLS.difference(point_df.columns)
    if missing:
        raise ValueError(f"point_df 缺少字段: {sorted(missing)}")


def _make_bin_labels(prob_df: pd.DataFrame):
    labels = []
    for row in prob_df.itertuples(index=False):
        labels.append(f"[{row.u_left:.2f},\n{row.u_right:.2f})")
    return labels


def plot_u_bin_summary(prob_df: pd.DataFrame, title_prefix: str = "u 分 bin 可视化"):
    _validate_prob_df(prob_df)

    plot_df = prob_df.copy().sort_values(["u_left", "u_right"]).reset_index(drop=True)
    x = np.arange(len(plot_df))
    labels = _make_bin_labels(plot_df)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    ax1 = axes[0]
    width = 0.38
    ax1.bar(x - width / 2, plot_df["n_points"].values, width=width, label="总点数")
    ax1.bar(x + width / 2, plot_df["n_roi_points"].values, width=width, label="ROI 点数")
    ax1.set_title(f"{title_prefix} - 各 bin 点数分布")
    ax1.set_ylabel("点数")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()

    for i, row in enumerate(plot_df.itertuples(index=False)):
        ax1.text(i - width / 2, row.n_points, f"{int(row.n_points)}", ha="center", va="bottom", fontsize=8)
        ax1.text(i + width / 2, row.n_roi_points, f"{int(row.n_roi_points)}", ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    probs = plot_df["p_roi_given_u"].fillna(0.0).values
    ymax = float(np.nanmax(probs)) if len(probs) > 0 else 1.0
    ax2.bar(x, probs)
    ax2.set_title(f"{title_prefix} - P(ROI|u)")
    ax2.set_xlabel("u bin 区间")
    ax2.set_ylabel("概率")
    ax2.set_ylim(0.0, max(1.0, ymax * 1.15))
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0)
    ax2.grid(True, axis="y", alpha=0.3)

    for i, p in enumerate(probs):
        ax2.text(i, p, f"{p:.2f}", ha="center", va="bottom", fontsize=8)

    return fig, axes


def plot_u_histogram_from_points(point_df: pd.DataFrame, num_bins: int = 20, title: str = "u 原始分布直方图"):
    _validate_point_df(point_df)

    u = point_df["u"].astype(float).values
    is_roi = point_df["is_roi"].astype(int).values

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.hist(u, bins=num_bins, range=(0.0, 1.0), alpha=0.55, label="全部点")
    if np.any(is_roi == 1):
        ax.hist(u[is_roi == 1], bins=num_bins, range=(0.0, 1.0), alpha=0.55, label="ROI 点")

    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("频数")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    return fig, ax


def plot_u_bin_combined(prob_df: pd.DataFrame, point_df: Optional[pd.DataFrame] = None, title_prefix: str = "u 分 bin 可视化"):
    _validate_prob_df(prob_df)
    if point_df is not None:
        _validate_point_df(point_df)

    plot_df = prob_df.copy().sort_values(["u_left", "u_right"]).reset_index(drop=True)
    x = np.arange(len(plot_df))
    labels = _make_bin_labels(plot_df)

    nrows = 3 if point_df is not None else 2
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(12, 10 if point_df is not None else 8),
        constrained_layout=True
    )

    axes = np.atleast_1d(axes)

    width = 0.38
    axes[0].bar(x - width / 2, plot_df["n_points"].values, width=width, label="总点数")
    axes[0].bar(x + width / 2, plot_df["n_roi_points"].values, width=width, label="ROI 点数")
    axes[0].set_title(f"{title_prefix} - 各 bin 点数")
    axes[0].set_ylabel("点数")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    probs = plot_df["p_roi_given_u"].fillna(0.0).values
    axes[1].bar(x, probs)
    axes[1].set_title(f"{title_prefix} - P(ROI|u)")
    axes[1].set_ylabel("概率")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", alpha=0.3)

    if point_df is not None:
        u = point_df["u"].astype(float).values
        is_roi = point_df["is_roi"].astype(int).values
        axes[2].hist(u, bins=len(plot_df), range=(0.0, 1.0), alpha=0.55, label="全部点")
        if np.any(is_roi == 1):
            axes[2].hist(u[is_roi == 1], bins=len(plot_df), range=(0.0, 1.0), alpha=0.55, label="ROI 点")
        axes[2].set_title(f"{title_prefix} - u 原始直方图")
        axes[2].set_xlabel("u")
        axes[2].set_ylabel("频数")
        axes[2].grid(True, axis="y", alpha=0.3)
        axes[2].legend()

    return fig, axes


def save_u_bin_figures(prob_df: pd.DataFrame,
                       point_df: Optional[pd.DataFrame] = None,
                       summary_path: str = "u_bin_summary.png",
                       combined_path: Optional[str] = "u_bin_combined.png",
                       show: bool = False):
    fig1, _ = plot_u_bin_summary(prob_df)
    fig1.savefig(summary_path, dpi=160, bbox_inches="tight")

    if combined_path is not None:
        fig2, _ = plot_u_bin_combined(prob_df, point_df=point_df)
        fig2.savefig(combined_path, dpi=160, bbox_inches="tight")
        plt.close(fig2)

    if show:
        plt.show()
    else:
        plt.close(fig1)


if __name__ == "__main__":
    print("这是一个 u 分 bin 可视化模块，请在主流程中导入调用。")
