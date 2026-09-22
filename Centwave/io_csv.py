"""CSV 文件读取与批量 ROI 峰检测处理"""

import os

import pandas as pd
import matplotlib.pyplot as plt

from detector import detect_peaks_centwave
from plot import plot_roi_with_peaks


def process_csv(csv_path, rt_col="rt", intensity_col="intensity",
                 roi_col=None, snr_thresh=3, out_dir="."):
    """
    批量处理CSV里的ROI。
    - 若 roi_col=None：假设整个CSV是一条ROI（两列：rt, intensity）
    - 若指定 roi_col：CSV里有多条ROI，按该列分组，每组单独出图
    """
    df = pd.read_csv(csv_path)
    results = []

    groups = df.groupby(roi_col) if roi_col else [("ROI", df)]

    for roi_name, sub in groups:
        sub = sub.sort_values(rt_col)
        rt = sub[rt_col].values
        intensity = sub[intensity_col].values

        peaks, baseline, noise = detect_peaks_centwave(
            rt, intensity, snr_thresh=snr_thresh
        )
        for p in peaks:
            p["roi"] = roi_name
        results.extend(peaks)

        fig = plot_roi_with_peaks(rt, intensity, peaks, title=f"ROI: {roi_name}")
        safe_name = str(roi_name).replace("/", "_")
        fig.savefig(os.path.join(out_dir, f"roi_{safe_name}_peaks.png"), dpi=150)
        plt.close(fig)

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(out_dir, "detected_peaks.csv"), index=False)
    return result_df
