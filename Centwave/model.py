"""
centWave 色谱峰检测 — 统一入口

模块结构:
  wavelet.py   小波变换（ricker / cwt）
  detector.py  核心峰检测算法（centWave）
  plot.py       可视化
  io_csv.py     CSV 文件读写
  io_mzml.py    .mzML 文件读写与批处理
  model.py      本文件 — 统一入口 & 命令行接口
"""

# 向后兼容：从各子模块重新导出所有公共接口
from wavelet import ricker, cwt                                          # noqa: F401
from detector import detect_peaks_centwave                                 # noqa: F401
from plot import plot_roi_with_peaks                                      # noqa: F401
from io_csv import process_csv                                            # noqa: F401
from io_mzml import read_mzml_chromatograms, process_mzml, process_mzml_batch  # noqa: F401


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # 命令行模式：python model.py <mzml_dir> [snr_thresh]
        mzml_dir = sys.argv[1]
        snr = float(sys.argv[2]) if len(sys.argv) > 2 else 3
        process_mzml_batch(mzml_dir, snr_thresh=snr)
    #测试demo
    else:
        # ---- 合成数据：3个不同宽度的峰 + 噪声，用于验证峰边界划定 ----
        import numpy as np

        rt = np.linspace(0, 200, 400)
        intensity = (
            3000 * np.exp(-0.5 * ((rt - 40) / 3) ** 2)
            + 8000 * np.exp(-0.5 * ((rt - 100) / 8) ** 2)
            + 1500 * np.exp(-0.5 * ((rt - 150) / 2) ** 2)
            + np.random.normal(0, 100, len(rt))
        )

        peaks, baseline, noise = detect_peaks_centwave(rt, intensity)
        print(f"baseline={baseline:.1f}, noise={noise:.1f}")
        for p in peaks:
            print(p)

        fig = plot_roi_with_peaks(rt, intensity, peaks, title="Demo ROI")
        fig.savefig("demo_roi_peaks.png", dpi=150)
        print("Saved demo_roi_peaks.png")
