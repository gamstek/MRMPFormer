"""色谱峰检测可视化"""

import matplotlib.pyplot as plt


def plot_roi_with_peaks(rt, intensity, peaks, title="ROI Peak Detection", save_path=None):
    """画出一条 ROI 的色谱图并在检测到的峰上画橙色框和红色峰顶点线"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rt, intensity, color="black", lw=1)

    for i, p in enumerate(peaks):
        ax.axvspan(p["rt_start"], p["rt_end"], color="orange", alpha=0.3)
        ax.axvline(p["apex_rt"], color="red", linestyle="--", lw=0.8)
        ax.annotate(f"SNR={p['snr']:.1f}",
                    xy=(p["apex_rt"], p["apex_intensity"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="red")

    ax.set_xlabel("Retention Time (s)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
