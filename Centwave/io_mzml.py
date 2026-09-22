""".mzML 文件读取与批量峰检测处理"""

import os
import glob
import base64
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from detector import detect_peaks_centwave


def read_mzml_chromatograms(mzml_path):
    """
    从 .mzML 文件中读取所有色谱图的 (rt, intensity) 数组。

    返回 list of dict: [{"chrom_id": str, "rt": np.array, "intensity": np.array}, ...]
    """
    NS = "{http://psi.hupo.org/ms/mzml}"

    with open(mzml_path, "rb") as f:
        raw = f.read()

    for enc in ["utf-8", "gbk", "gb2312", "gb18030"]:
        try:
            tree = ET.ElementTree(ET.fromstring(raw.decode(enc)))
            break
        except (ET.ParseError, UnicodeDecodeError):
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
        tree = ET.ElementTree(ET.fromstring(text))

    chromatograms = tree.findall(f".//{NS}chromatogram")
    results = []

    for c in chromatograms:
        chrom_id = c.get("id", c.get("index", "0"))
        rt = None
        intensity = None

        for ba in c.findall(f".//{NS}binaryDataArray"):
            names = [cv.get("name") for cv in ba.findall(f"{NS}cvParam")]
            binary_text = ba.find(f"{NS}binary").text.strip()
            arr = np.frombuffer(base64.b64decode(binary_text), dtype=np.float64)

            if "time array" in names:
                rt = arr
            elif "intensity array" in names:
                intensity = arr

        if rt is not None and intensity is not None:
            results.append({"chrom_id": chrom_id, "rt": rt, "intensity": intensity})

    return results


def process_mzml(mzml_path, out_dir=".", snr_thresh=3, plot=True):
    """
    处理单个 .mzML 文件：读取所有色谱图 → 峰检测 → 输出结果表并可选出图。

    输出结果表字段：
      file, chrom, rt_start, rt_end, apex_rt, apex_intensity, snr
    """
    basename = os.path.splitext(os.path.basename(mzml_path))[0]
    chromatograms = read_mzml_chromatograms(mzml_path)
    all_peaks = []

    if plot and chromatograms:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        n_chroms = len(chromatograms)
        fig_height = min(3 * n_chroms, 30)  # 最多30英寸，超出则自动压缩
        fig, axes = plt.subplots(n_chroms, 1,
                                 figsize=(10, fig_height), squeeze=False)
        for i, chrom in enumerate(chromatograms):
            rt_c = chrom["rt"]
            intensity_c = chrom["intensity"]
            peaks_c, baseline, noise = detect_peaks_centwave(
                rt_c, intensity_c, snr_thresh=snr_thresh
            )

            for p in peaks_c:
                p["file"] = basename
                p["chrom"] = chrom["chrom_id"]
            all_peaks.extend(peaks_c)

            ax = axes[i][0]
            ax.plot(rt_c, intensity_c, color="black", lw=0.6)
            ax.set_title(f"{basename}  {chrom['chrom_id']}  "
                         f"(baseline={baseline:.0f}, noise={noise:.1f}, peaks={len(peaks_c)})")
            for p in peaks_c:
                ax.axvspan(p["rt_start"], p["rt_end"], color="orange", alpha=0.3)
                ax.axvline(p["apex_rt"], color="red", linestyle="--", lw=0.8)
                ax.annotate(f"SNR={p['snr']:.1f}", xy=(p["apex_rt"], p["apex_intensity"]),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=6, color="red")
            ax.set_xlabel("Retention Time (s)")
            ax.set_ylabel("Intensity")

        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{basename}_peaks.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    else:
        for chrom in chromatograms:
            rt_c = chrom["rt"]
            intensity_c = chrom["intensity"]
            peaks_c, _, _ = detect_peaks_centwave(
                rt_c, intensity_c, snr_thresh=snr_thresh
            )
            for p in peaks_c:
                p["file"] = basename
                p["chrom"] = chrom["chrom_id"]
            all_peaks.extend(peaks_c)

    df = pd.DataFrame(all_peaks)
    return df


def process_mzml_batch(mzml_dir, out_dir=None, snr_thresh=3, plot=True):
    """
    批量处理目录下所有 .mzML 文件，汇总结果并出图。

    输出:
      - results.csv      汇总结果表（file, chrom, rt_start, rt_end, apex_rt, apex_intensity, snr）
      - *_peaks.png      每个文件的色谱出峰图
    """
    mzml_dir = os.path.abspath(mzml_dir)
    if out_dir is None:
        out_dir = os.path.join(mzml_dir, "output")

    os.makedirs(out_dir, exist_ok=True)
    mzml_files = sorted(glob.glob(os.path.join(mzml_dir, "*.mzML")))

    all_results = []
    for mzml_path in mzml_files:
        print(f"Processing: {os.path.basename(mzml_path)} ...")
        df = process_mzml(mzml_path, out_dir=out_dir, snr_thresh=snr_thresh, plot=plot)
        all_results.append(df)

    if all_results:
        result_df = pd.concat(all_results, ignore_index=True)
        result_df = result_df[["file", "chrom", "rt_start", "rt_end",
                               "apex_rt", "apex_intensity", "snr"]]
        csv_path = os.path.join(out_dir, "results.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"\nDone. {len(mzml_files)} files processed, "
              f"{len(result_df)} peaks detected.")
        print(f"Results saved to: {csv_path}")
        return result_df
    else:
        print("No .mzML files found.")
        return pd.DataFrame()
