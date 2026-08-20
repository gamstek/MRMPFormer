# testXIC.py (with Gaussian Smoothing)
"""
提取 XIC 并生成 ROI 图像。支持两种输入模式：
1. mzML 文件：从 mzML 提取 chromatogram
2. 外部数组：由 (m/z名称、RT数组、强度数组) 生成，可选 (m/z, 预期RT) 替代 feature.csv 中的 RT

用法:
  # mzML 单文件：结果写入 output_dir/<mzML 文件名(无扩展名)>/（与批量模式子文件夹命名一致）
  python testXIC.py --mzml "path/to/file.mzML" --output_dir xic-roi-batch
  # 若需直接写入 output_dir（不建子文件夹）：加 --flat_output

  # 外部数组模式（JSON）
  python testXIC.py --from_json compounds.json --output_dir results/my_output

  # Python API
  from preprocessing.xic_extraction import extract_xic_from_arrays
  compounds = [{"mz_name": 142.0, "rt": [...], "intensity": [...]}]  # RT 以峰顶为准
  extract_xic_from_arrays("output_dir", compounds)
"""
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TRUEDATA_TEST_DIR = ROOT_DIR / "truedata" / "test"

from pyopenms import MzMLFile, MSExperiment
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
from scipy.ndimage import gaussian_filter1d  # 新增导入
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def roi_safe_name_base(n: int, mz_parent, q3_product, compound_name=None) -> str:
    """
    ROI 文件名主体（无扩展名）：N_mz{母离子}[_q3{子离子}][_{化合物名}]。
    compound_name 通常来自 mzML chromatogram id（如 阿维菌素-1）。
    """
    try:
        mz_ok = mz_parent is not None and np.isfinite(float(mz_parent))
    except (TypeError, ValueError):
        mz_ok = False
    base = f"{n}_mz{float(mz_parent):.4f}" if mz_ok else f"{n}_mznan"
    try:
        q_ok = q3_product is not None and np.isfinite(float(q3_product))
    except (TypeError, ValueError):
        q_ok = False
    if q_ok:
        base = f"{base}_q3{float(q3_product):.4f}"
    if compound_name is not None:
        name = str(compound_name).strip()
        if name:
            from utils.mzml_chromatogram_ids import filesystem_slug_for_native_id

            slug = filesystem_slug_for_native_id(name, max_len=48)
            if slug and slug != "empty":
                base = f"{base}_{slug}"
    return base


def _load_standard_rt_refs(standard_refs_csv):
    """
    Load standard RT references from standard_refs.csv.
    Returns:
      key_to_rt: {"mz{:.4f}_q3{:.4f}": ref_rt_peak}
      mz_to_rt: {round(mz,4): median_ref_rt_peak}
    """
    if not standard_refs_csv:
        return {}, {}
    p = Path(standard_refs_csv).resolve()
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] standard_refs.csv not found: {p}")
    df = pd.read_csv(p)
    if not {"compound_key", "ref_rt_peak"}.issubset(df.columns):
        raise ValueError("[ERROR] standard_refs.csv must contain columns: compound_key, ref_rt_peak")
    key_to_rt = {}
    mz_map = {}
    use_mz_expected = "mz_expected_rt" in df.columns
    for _, r in df.iterrows():
        key = str(r.get("compound_key", "")).strip()
        rt_ref = pd.to_numeric(r.get("ref_rt_peak"), errors="coerce")
        if (not key) or pd.isna(rt_ref):
            continue
        key_to_rt[key] = float(rt_ref)
        if key.startswith("mz") and "_q3" in key:
            try:
                mz_val = round(float(key[2:].split("_q3", 1)[0]), 4)
                if use_mz_expected:
                    rt_mz = pd.to_numeric(r.get("mz_expected_rt"), errors="coerce")
                    if not pd.isna(rt_mz):
                        mz_map.setdefault(mz_val, []).append(float(rt_mz))
                    else:
                        mz_map.setdefault(mz_val, []).append(float(rt_ref))
                else:
                    mz_map.setdefault(mz_val, []).append(float(rt_ref))
            except Exception:
                pass
    mz_to_rt = {k: float(np.nanmedian(v)) for k, v in mz_map.items() if len(v) > 0}
    print(f"[INFO] Loaded standard RT refs: keys={len(key_to_rt)}, mz_fallback={len(mz_to_rt)} from {p}")
    return key_to_rt, mz_to_rt

def _to_loadable_path(mzml_path):
    """
    返回 pyopenms 可加载的路径。

    Windows 下若路径含非 ASCII 字符，pyopenms 在部分环境会加载失败，
    则复制到临时 ASCII 文件名后再加载。
    """
    path = Path(mzml_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"mzML not found: {path}")
    path_str = str(path)
    if sys.platform != "win32":
        return path_str, None

    # 避免使用 8.3 短路径：某些情况下会把 .mzML 截断成 .MZM。
    if not any(ord(c) > 127 for c in path_str):
        return path_str, None

    # 含非 ASCII 时复制到临时文件（ASCII 名）供 pyopenms 加载
    try:
        import tempfile
        import shutil
        suffix = path.suffix or ".mzML"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="mzml_")
        os.close(fd)
        shutil.copy2(path, tmp_path)
        return tmp_path, tmp_path
    except Exception:
        return path_str, None


def _native_id_to_str(native_id):
    if native_id is None:
        return ""
    if isinstance(native_id, bytes):
        return native_id.decode("utf-8", errors="replace")
    return str(native_id)


def _label_key_from_channel(compound, channel):
    """标注行 → mzML native_id 键：定量离子→-1，定性离子→-2（与 coco_annotation.label_key 一致，内联避免循环依赖）。"""
    ch = (channel or "").strip()
    suffix = "1" if "定量" in ch else ("2" if "定性" in ch else None)
    if suffix is None:
        return None
    return f"{(compound or '').strip()}-{suffix}"


def _parse_label_rt(s):
    """解析标注 rt 字段 '16.428(0.000)' / '16.428' → 分钟；空/非法 → None（与 label_qc 一致）。"""
    s = (s or "").strip()
    if not s:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else None


def _parse_q1_q3_from_text(native_id_text):
    text = str(native_id_text)
    q1 = q3 = None
    for pat in (r"Q1=([\d\.]+)", r"q1=([\d\.]+)", r"precursor[=:_ ]([\d\.]+)"):
        m1 = re.search(pat, text)
        if m1:
            q1 = float(m1.group(1))
            break
    for pat in (r"Q3=([\d\.]+)", r"q3=([\d\.]+)", r"product[=:_ ]([\d\.]+)"):
        m3 = re.search(pat, text)
        if m3:
            q3 = float(m3.group(1))
            break
    return q1, q3


def _q1_q3_from_chrom_metadata(chrom):
    q1 = q3 = None
    try:
        mz_pre = float(chrom.getPrecursor().getMZ())
        if np.isfinite(mz_pre) and mz_pre > 0:
            q1 = mz_pre
    except Exception:
        pass
    try:
        mz_pro = float(chrom.getProduct().getMZ())
        if np.isfinite(mz_pro) and mz_pro > 0:
            q3 = mz_pro
    except Exception:
        pass
    return q1, q3


def _extract_q1_q3(chrom, native_id_text):
    q1, q3 = _parse_q1_q3_from_text(native_id_text)
    mq1, mq3 = _q1_q3_from_chrom_metadata(chrom)
    if q1 is None:
        q1 = mq1
    if q3 is None:
        q3 = mq3
    return q1, q3


def extract_xic_with_pyopenms(
    mzml_path,
    output_dir,
    smooth_sigma=1.0,
    standard_rt_key_to_rt=None,
    standard_rt_mz_to_rt=None,
    min_chrom_points=0,
    min_max_intensity=0.0,
    rt_center_overrides=None,
    exclude_tic=False,
    exclude_native_ids=None,
    labels=None,
):
    """
    提取 XIC 并生成 ROI 图像，支持高斯平滑。
    Parameters:
        mzml_path (str): 输入 .mzML 文件路径
        output_dir (str): 输出目录
        smooth_sigma (float): 高斯平滑 sigma 参数（设为 0 则禁用平滑）
        standard_rt_key_to_rt / standard_rt_mz_to_rt: 已弃用，保留参数仅为兼容旧调用；ROI 与 feature 的 RT
            均以该通道 XIC 在（可选）平滑后的最高峰 RT 为准。
        min_chrom_points (int): >0 时跳过数据点数少于此值的 Transition（流程图：数据点过少不参与后续）
        min_max_intensity (float): >0 时跳过整条 XIC 最大强度（平滑后）低于此值的通道（流程图：低强度剔除）
        rt_center_overrides (dict): {native_id: RT分钟}，ROI 窗口中心覆盖表 —— 命中 native_id 时以
            指定 RT（如人工标注 RT）为窗口中心，替代默认的谱图最高强度点；未命中的通道（TIC 等）
            仍用最高强度点。用于训练数据生成时与标注对齐（improve.md 第 7 项）。
        exclude_tic (bool): True 时跳过无 (Q1,Q3) 数值的通道（TIC 等），不生成 ROI、不进 feature/xic_matrix；
            剔除记录写入 pipeline_qc_excluded.csv（reason=tic_excluded）。
        exclude_native_ids (dict/set): 标注 RT 一致性 QC 命中剔除的 native_id 集合；dict 时 value 为剔除原因
            （label_rt_cross_sample / label_rt_ion_pair），set 时原因记为 label_rt。命中通道不生成 ROI、
            不进 feature/xic_matrix，剔除记录写入 pipeline_qc_excluded.csv（reason 区分检查类型）。
        labels (list[dict]): 标注行列表（键含 compound/channel/rt）。提供时 ROI 由标注驱动（B 范式）：
            仅标注命中（native_id == label_key(compound, channel)）的通道生成 ROI，窗口中心 = 标注 rt
            （替代最高强度点）；标注了但 mzML 无对应通道的行记 pipeline_qc_excluded.csv（reason=label_no_channel）；
            标注 rt 无法解析的行 reason=label_rt_missing；未标注的 mzML 通道不生成 ROI。
            不提供时维持通道驱动（默认最高强度点居中，rt_center_overrides 可覆盖）。
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Reading mzML with pyopenms: {mzml_path}")
    if smooth_sigma > 0:
        print(f"[INFO] Applying Gaussian smoothing with sigma={smooth_sigma}")

    from utils.mzml_load import load_ms_experiment
    from utils.mzml_chromatogram_ids import resolve_native_ids_for_chromatograms

    exp = load_ms_experiment(mzml_path)
    chromatograms = exp.getChromatograms()
    print(f"[INFO] Found {len(chromatograms)} chromatograms.")
    if len(chromatograms) == 0:
        raise ValueError("No chromatograms found in mzML file!")

    native_ids = resolve_native_ids_for_chromatograms(
        mzml_path, chromatograms, _native_id_to_str
    )

    # === label 驱动模式（B 范式）：ROI 由标注行决定，窗口中心 = 标注 rt ===
    label_rt_map = None  # {native_id: rt_min|None}
    matched_nids = set()
    if labels:
        label_rt_map = {}
        for rec in labels:
            if rec.get("_qc_excluded"):
                continue
            kid = _label_key_from_channel(rec.get("compound"), rec.get("channel"))
            if not kid:
                continue
            label_rt_map.setdefault(kid, _parse_label_rt(rec.get("rt")))

    features = []
    intensity_matrix = []  # 存储对齐后的强度 (N, S)
    common_rt = None  # 公共 RT 轴（分钟）
    roi_windows_rows = []  # 每张 ROI 实际绘图窗口 [rt_lo, rt_hi]（分钟），供积分时像素→RT 映射一致
    # 按 (Q1, Q3) 去重：一个母离子只对应有限子离子（如 Quantifier+Qualifier=2），避免 mzML 重复 chromatogram 导致 4 条
    seen_q1_q3 = set()
    qc_excluded = []

    for i, chrom in enumerate(chromatograms):
        rt_sec = np.array([p.getRT() for p in chrom])  # 单位：秒
        intensity = np.array([p.getIntensity() for p in chrom])
        if len(rt_sec) == 0:
            print(f"[WARN] Chromatogram {i} is empty. Skipping.")
            continue

        native_id = native_ids[i] if i < len(native_ids) else _native_id_to_str(chrom.getNativeID())
        q1, q3 = _extract_q1_q3(chrom, native_id)

        # === label 驱动（B 范式）：仅标注命中的通道生成 ROI；标注 rt 缺失的通道剔除 ===
        label_rt_center = None
        if label_rt_map is not None:
            matched_key = native_id if native_id in label_rt_map else None
            if matched_key is None:
                # 回退：mzML 通道 id 不带 -1/-2 后缀（如 traindata1）时，按化合物裸名唯一匹配标注键；
                # 裸名同时命中 -1/-2（歧义）则不匹配，保持严格
                _cands = [k for k in label_rt_map
                          if k.endswith(("-1", "-2")) and k[: -2] == native_id]
                if len(_cands) == 1:
                    matched_key = _cands[0]
                    print(f"[INFO] label 匹配回退（裸名）: mzML「{native_id}」→ 标注键「{matched_key}」")
            if matched_key is None:
                # mzML 有通道但标注未覆盖 → 不生成 ROI（ROI 由标注驱动）
                continue
            matched_nids.add(matched_key)
            label_rt_center = label_rt_map[matched_key]
            if label_rt_center is None:
                print(
                    f"[WARN] Skip chrom {i}: 标注 rt 字段缺失/非法（label_rt_missing）; "
                    f"native_id={native_id[:72]}..."
                )
                qc_excluded.append({
                    "chrom_index": i,
                    "native_id": native_id,
                    "q1": q1,
                    "q3": q3,
                    "reason": "label_rt_missing",
                    "n_points": int(len(rt_sec)),
                    "max_intensity_smoothed": float(np.max(intensity)) if len(intensity) else 0.0,
                })
                continue

        # 数值型 (Q1,Q3) 去重；若仍缺（如 TIC），用 native_id 避免全部为 (None,None) 误判重复
        if q1 is not None and q3 is not None:
            key = (round(q1, 4), round(q3, 2))
        else:
            key = ("native_id", native_id)
        if key in seen_q1_q3:
            print(f"[WARN] Skip duplicate (Q1,Q3) chromatogram: Q1={q1}, Q3={q3}, native_id={native_id[:60]}...")
            continue
        seen_q1_q3.add(key)

        # === TIC 等无 (Q1,Q3) 数值通道剔除（--exclude_tic / exclude_tic=True）===
        if exclude_tic and q1 is None and q3 is None:
            print(
                f"[WARN] Skip chrom {i}: no numeric (Q1,Q3) (TIC-like); native_id={native_id[:72]}..."
            )
            qc_excluded.append({
                "chrom_index": i,
                "native_id": native_id,
                "q1": q1,
                "q3": q3,
                "reason": "tic_excluded",
                "n_points": int(len(rt_sec)),
                "max_intensity_smoothed": float(np.max(intensity)) if len(intensity) else 0.0,
            })
            continue

        # === 【新增】高斯平滑处理 ===
        if smooth_sigma > 0:
            intensity = gaussian_filter1d(intensity, sigma=smooth_sigma)
        # ===========================

        n_pts = int(len(rt_sec))
        imax = float(np.max(intensity)) if n_pts else 0.0

        # === 【标注QC】命中 exclude_native_ids 的通道不生成 ROI（reason 区分检查类型）===
        if exclude_native_ids and native_id in exclude_native_ids:
            _label_reason = (
                exclude_native_ids.get(native_id, "label_rt")
                if isinstance(exclude_native_ids, dict)
                else "label_rt"
            )
            print(
                f"[WARN] Skip chrom {i}: 标注 RT 一致性 QC 剔除（{_label_reason}）; "
                f"native_id={native_id[:72]}..."
            )
            qc_excluded.append({
                "chrom_index": i,
                "native_id": native_id,
                "q1": q1,
                "q3": q3,
                "reason": _label_reason,
                "n_points": n_pts,
                "max_intensity_smoothed": imax,
            })
            continue

        if min_chrom_points > 0 and n_pts < int(min_chrom_points):
            print(
                f"[WARN] Skip chrom {i}: too few RT points ({n_pts} < {min_chrom_points}); "
                f"native_id={native_id[:72]}..."
            )
            qc_excluded.append({
                "chrom_index": i,
                "native_id": native_id,
                "q1": q1,
                "q3": q3,
                "reason": "too_few_points",
                "n_points": n_pts,
                "max_intensity_smoothed": imax,
            })
            continue
        if min_max_intensity > 0.0 and imax < float(min_max_intensity):
            print(
                f"[WARN] Skip chrom {i}: max intensity {imax:.4g} < {min_max_intensity} "
                f"(after smooth_sigma={smooth_sigma}); native_id={native_id[:72]}..."
            )
            qc_excluded.append({
                "chrom_index": i,
                "native_id": native_id,
                "q1": q1,
                "q3": q3,
                "reason": "low_max_intensity",
                "n_points": n_pts,
                "max_intensity_smoothed": imax,
            })
            continue

        # === 计算 apex RT (分钟) ===
        max_idx = np.argmax(intensity)
        rt_apex_min = rt_sec[max_idx] / 60.0  # 转为分钟
        rt_apex_sec = rt_sec[max_idx]
        # ROI 窗口中心：label 驱动（B 范式）时以标注 rt 为源头（标注即正确答案，无需与 apex 对比报差异）；
        # 否则外部覆盖表（rt_center_overrides）；再否则以（平滑后）强度最高点对应 RT 为中心
        rt_center_min = rt_apex_min
        if label_rt_center is not None:
            rt_center_min = float(label_rt_center)
        elif rt_center_overrides and native_id in rt_center_overrides:
            override_rt = float(rt_center_overrides[native_id])
            if np.isfinite(override_rt) and rt_sec[0] / 60.0 <= override_rt <= rt_sec[-1] / 60.0:
                if abs(override_rt - rt_apex_min) > 1e-6:
                    print(
                        f"[INFO] ROI center override {native_id}: 标注 RT {override_rt:.3f} min "
                        f"替代最高强度点 {rt_apex_min:.3f} min"
                    )
                rt_center_min = override_rt

        features.append({
            'Compound Name': len(features) + 1,
            'native_id': native_id,
            'mz': q1 if q1 is not None else np.nan,
            'q3': q3 if q3 is not None else np.nan,
            'RT': round(rt_center_min, 3)
        })

        # === 初始化公共 RT 轴（分钟，用第一个非空色谱）===
        if common_rt is None:
            common_rt = (rt_sec / 60.0).copy()

        # === 对齐强度到 common_rt（用于 xic_matrix.npy）===
        rt_min = rt_sec / 60.0
        if len(rt_min) == len(common_rt) and np.allclose(rt_min, common_rt, atol=1e-3):
            aligned_intensity = intensity
        else:
            from scipy.interpolate import interp1d
            f = interp1d(rt_min, intensity, bounds_error=False, fill_value=0.0)
            aligned_intensity = f(common_rt)
        intensity_matrix.append(aligned_intensity)

        # 使用 2 分钟窗口（±1 分钟）裁剪 ROI ===
        window_half_min = 1.0  # ±1 分钟 → 总宽 2 分钟
        rt_start_min = rt_center_min - window_half_min
        rt_end_min = rt_center_min + window_half_min

        # 转换为秒用于裁剪
        rt_start_sec = rt_start_min * 60.0
        rt_end_sec = rt_end_min * 60.0

        # 边界保护：不超出实际 RT 范围
        rt_start_sec = max(rt_start_sec, rt_sec[0])
        rt_end_sec = min(rt_end_sec, rt_sec[-1])

        mask = (rt_sec >= rt_start_sec) & (rt_sec <= rt_end_sec)
        plot_rt_sec = rt_sec[mask]
        plot_intensity = intensity[mask]

        # === ROI 命名：N_mz{母离子}_q3{子离子}.jpeg（N=feature 行号 1-based），与 feature.csv/xic_matrix 一一对应 ===
        n = len(features)
        safe_name_base = roi_safe_name_base(n, q1, q3, compound_name=native_id)

        # === 只保存 CNN 输入图像 (与原项目一致：固定 400x300, 无坐标轴) ===
        fig = Figure(figsize=(4, 3), dpi=100)  # 400x300
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(plot_rt_sec / 60.0, plot_intensity, color='blue', linewidth=1.5)
        # 固定 x 轴为裁剪窗口（分钟），使 400px 与 roi_rt_mapping 线性映射一致
        if rt_end_sec > rt_start_sec and len(plot_rt_sec) > 0:
            ax.set_xlim(rt_start_sec / 60.0, rt_end_sec / 60.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        roi_path = os.path.join(output_dir, f"{safe_name_base}.jpeg")  # 原项目默认 jpeg
        canvas.print_jpeg(roi_path)
        # 记录该 ROI 实际 x 轴窗口（与 set_xlim 一致），积分时用此窗口做像素→RT 映射，避免与 common_rt 裁剪不一致导致偏移
        rt_lo_actual = rt_start_sec / 60.0
        rt_hi_actual = rt_end_sec / 60.0
        roi_windows_rows.append({"image": f"{safe_name_base}.jpeg", "rt_lo": rt_lo_actual, "rt_hi": rt_hi_actual})
        print(f"[INFO] Saved CNN input ROI ({rt_start_min:.2f}~{rt_end_min:.2f} min): {roi_path}")

        # === 注释掉以下所有 debug 图像生成代码 ===
        # debug_dir = os.path.join(output_dir, "debug_plots")
        # Path(debug_dir).mkdir(parents=True, exist_ok=True)
        #
        # # 1. 保存原始 XIC 图像 (带坐标轴)
        # plt.figure(figsize=(8, 4))
        # plt.plot(rt_sec / 60.0, intensity, 'b-', linewidth=1, markersize=2)
        # plt.axvline(rt_apex_min, color='r', linestyle='--', label=f'Apex RT: {rt_apex_min:.2f} min')
        # if q1 is not None or q3 is not None:
        #     plt.title(f"Full XIC (Smoothed σ={smooth_sigma}) - Q1={q1_str_for_title}, Q3={q3_str_for_title}")
        # else:
        #     plt.title(f"Full XIC (Smoothed σ={smooth_sigma}) - Chrom {i}")
        # plt.xlabel("Retention Time (min)")
        # plt.ylabel("Intensity")
        # plt.grid(True, linestyle=':', alpha=0.7)
        # plt.legend()
        # full_xic_path = os.path.join(debug_dir, f"{safe_name_base}_full_xic.png")
        # plt.savefig(full_xic_path, dpi=150, bbox_inches='tight')
        # plt.close()
        # print(f"[DEBUG] Saved full XIC plot: {full_xic_path}")
        #
        # # 2. 保存裁剪范围示意图
        # plt.figure(figsize=(8, 4))
        # plt.plot(rt_sec / 60.0, intensity, 'b-', linewidth=1, label='Full XIC (Smoothed)')
        # plt.axvspan(rt_start_min, rt_end_min, color='yellow', alpha=0.3, label='ROI Window (±1 min)')
        # plt.axvline(rt_apex_min, color='r', linestyle='--', label=f'Apex RT: {rt_apex_min:.2f} min')
        # plt.title(f"ROI Selection (Smoothed σ={smooth_sigma}) - Q1={q1_str_for_title}, Q3={q3_str_for_title}")
        # plt.xlabel("Retention Time (min)")
        # plt.ylabel("Intensity")
        # plt.grid(True, linestyle=':', alpha=0.7)
        # plt.legend()
        # roi_selection_path = os.path.join(debug_dir, f"{safe_name_base}_roi_selection.png")
        # plt.savefig(roi_selection_path, dpi=150, bbox_inches='tight')
        # plt.close()
        # print(f"[DEBUG] Saved ROI selection plot: {roi_selection_path}")
        #
        # # 3. 额外：将 CNN 输入图也存一份到 debug 文件夹 (PNG, 更清晰)
        # cnn_input_debug_path = os.path.join(debug_dir, f"{safe_name_base}_cnn_input.png")
        # plt.figure(figsize=(4, 3), dpi=150)
        # plt.plot(plot_rt_sec, plot_intensity, 'b-', linewidth=1.5)
        # plt.axis('off')
        # plt.savefig(cnn_input_debug_path, dpi=150, bbox_inches='tight', pad_inches=0)
        # plt.close()
        # print(f"[DEBUG] Saved CNN input (debug PNG): {cnn_input_debug_path}")
        # ========================================

    # === 保存 feature.csv（与 main.py / io_utils.load_features 一致：Compound Name, mz, RT；可选 q3）===
    _feature_cols = ["Compound Name", "native_id", "mz", "q3", "RT"]
    df_features = pd.DataFrame(features) if features else pd.DataFrame(columns=_feature_cols)
    feature_csv = os.path.join(output_dir, "feature.csv")
    df_features.to_csv(feature_csv, index=False)
    if features:
        print(f"[INFO] Saved feature.csv (main.py compatible): {feature_csv}")
    else:
        print(
            "[WARN] Saved empty feature.csv (headers only): all chromatograms were skipped by QC "
            "(e.g. --pipeline_min_max_intensity). See pipeline_qc_excluded.csv"
        )

    # === 保存 roi_windows.csv：每张 ROI 的 x 轴窗口 [rt_lo, rt_hi]（分钟），积分时优先使用此窗口做像素→RT 映射 ===
    # 全部通道被 QC 剔除时也写 headers-only 空表（与 feature.csv 行为一致），下游 read_csv 不炸
    roi_windows_csv = os.path.join(output_dir, "roi_windows.csv")
    if roi_windows_rows:
        pd.DataFrame(roi_windows_rows).to_csv(roi_windows_csv, index=False)
        print(f"[INFO] Saved roi_windows.csv (integration mapping): {roi_windows_csv}")
    else:
        pd.DataFrame(columns=["image", "rt_lo", "rt_hi"]).to_csv(roi_windows_csv, index=False)
        print(f"[WARN] Saved empty roi_windows.csv (headers only): {roi_windows_csv}")

    # === 保存 XIC 矩阵: (N+1, S) ===
    if intensity_matrix and common_rt is not None:
        intensity_array = np.array(intensity_matrix)  # (N, S)
        xic_full = np.vstack([common_rt, intensity_array])  # 第0行: RT (分钟)
        npy_path = os.path.join(output_dir, "xic_matrix.npy")
        np.save(npy_path, xic_full)
        print(f"[INFO] Saved XIC matrix with RT row (minutes): {npy_path}, shape={xic_full.shape}")
    else:
        print("[WARN] No valid chromatograms to save XIC matrix.")

    # === label 驱动（B 范式）：标注了但 mzML 无对应通道的行 → 记剔除（label_no_channel）===
    if label_rt_map is not None:
        for kid, rt_val in label_rt_map.items():
            if kid in matched_nids or (exclude_native_ids and kid in exclude_native_ids):
                continue
            print(f"[WARN] 标注行在 mzML 无对应通道（label_no_channel）: native_id={kid[:72]}...")
            qc_excluded.append({
                "chrom_index": -1,
                "native_id": kid,
                "q1": None,
                "q3": None,
                "reason": "label_no_channel",
                "n_points": 0,
                "max_intensity_smoothed": 0.0,
            })

    if qc_excluded:
        excl_path = os.path.join(output_dir, "pipeline_qc_excluded.csv")
        try:
            pd.DataFrame(qc_excluded).to_csv(excl_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] QC 剔除条目已写入: {excl_path} ({len(qc_excluded)} 行)")
        except OSError as e:
            print(f"[WARN] 无法写入 pipeline_qc_excluded.csv: {e}")

    n_roi = len(features)
    n_chrom = len(chromatograms)
    n_excl = len(qc_excluded)
    print(
        f"[INFO] mzML「{os.path.basename(mzml_path)}」: 读取色谱 {n_chrom} 条 → "
        f"生成 ROI 图 {n_roi} 张（QC 剔除 {n_excl} 条）"
    )
    print(f"[DONE] Processed {n_roi} compounds.")
    return {
        "mzml": os.path.basename(mzml_path),
        "n_chromatograms": n_chrom,
        "n_roi_images": n_roi,
        "n_qc_excluded": n_excl,
    }


def extract_xic_from_arrays(output_dir, compounds, smooth_sigma=0.0):
    """
    由外部输入的 (m/z名称、RT数组、强度数组) 生成 XIC 矩阵和 ROI 图像。

    Parameters:
        output_dir (str): 输出目录
        compounds (list): 化合物列表，每个元素为 dict，包含:
            - mz_name: str 或 float，如 "142.0" 或 142.0
            - rt: array-like，保留时间数组（单位：分钟；若值>200 则视为秒并自动转换）
            - intensity: array-like，强度数组
            - q3: 可选，子离子 m/z
            - expected_rt: 已忽略；feature 与 ROI 窗口均以峰顶 RT（平滑后 argmax）为准。
        smooth_sigma (float): 高斯平滑 sigma（0 表示不平滑）

    Returns:
        str: 输出目录路径
    """
    from scipy.interpolate import interp1d

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 统一 RT 网格：取所有化合物的 RT 并集，采样
    all_rt_min, all_rt_max = float("inf"), float("-inf")
    for c in compounds:
        rt = np.asarray(c["rt"], dtype=np.float64)
        if rt.size == 0:
            continue
        if np.nanmax(rt) > 200:
            rt = rt / 60.0
        all_rt_min = min(all_rt_min, float(np.nanmin(rt)))
        all_rt_max = max(all_rt_max, float(np.nanmax(rt)))

    if all_rt_min >= all_rt_max:
        raise ValueError("No valid RT range from compounds.")

    # 采样步长：取各化合物最小间隔的合理值
    min_step = 0.01
    for c in compounds:
        rt = np.asarray(c["rt"], dtype=np.float64)
        if np.nanmax(rt) > 200:
            rt = rt / 60.0
        if rt.size >= 2:
            steps = np.diff(np.sort(rt))
            steps = steps[steps > 0]
            if steps.size > 0:
                min_step = min(min_step, float(np.min(steps)))
    n_pts = max(200, int((all_rt_max - all_rt_min) / min_step) + 1)
    common_rt = np.linspace(all_rt_min, all_rt_max, n_pts)

    features = []
    intensity_matrix = []
    roi_windows_rows = []

    for i, c in enumerate(compounds):
        rt_raw = np.asarray(c["rt"], dtype=np.float64)
        intensity_raw = np.asarray(c["intensity"], dtype=np.float64)
        if rt_raw.size < 2 or intensity_raw.size < 2 or rt_raw.size != intensity_raw.size:
            print(f"[WARN] Compound {i+1} invalid (rt/intensity length). Skipping.")
            continue

        rt_min = rt_raw.copy()
        if np.nanmax(rt_min) > 200:
            rt_min = rt_min / 60.0

        if smooth_sigma > 0:
            intensity_raw = gaussian_filter1d(intensity_raw.astype(np.float64), sigma=smooth_sigma)

        apex_idx = np.argmax(intensity_raw)
        rt_apex_min = float(rt_min[apex_idx])
        rt_for_feature = rt_apex_min

        mz_val = c.get("mz_name")
        if mz_val is None:
            mz_val = np.nan
        else:
            try:
                mz_val = float(mz_val)
            except (TypeError, ValueError):
                mz_val = np.nan
        q3_val = c.get("q3", np.nan)
        if q3_val is not None:
            try:
                q3_val = float(q3_val)
            except (TypeError, ValueError):
                q3_val = np.nan
        else:
            q3_val = np.nan

        compound_label = (
            c.get("compound_name")
            or c.get("name")
            or c.get("native_id")
            or ""
        )

        features.append({
            "Compound Name": len(features) + 1,
            "native_id": str(compound_label).strip(),
            "mz": mz_val,
            "q3": q3_val,
            "RT": round(float(rt_for_feature), 3),
        })

        f_interp = interp1d(rt_min, intensity_raw, kind="linear", bounds_error=False, fill_value=0.0)
        aligned_intensity = f_interp(common_rt)
        intensity_matrix.append(aligned_intensity)

        window_half_min = 1.0
        rt_start_min = rt_for_feature - window_half_min
        rt_end_min = rt_for_feature + window_half_min
        rt_start_min = max(rt_start_min, float(np.nanmin(rt_min)))
        rt_end_min = min(rt_end_min, float(np.nanmax(rt_min)))
        if rt_end_min <= rt_start_min:
            rt_start_min = float(np.nanmin(rt_min))
            rt_end_min = float(np.nanmax(rt_min))

        mask = (rt_min >= rt_start_min) & (rt_min <= rt_end_min)
        if np.sum(mask) < 2:
            plot_rt = rt_min
            plot_intensity = intensity_raw
        else:
            plot_rt = rt_min[mask]
            plot_intensity = intensity_raw[mask]

        n = len(features)
        safe_name_base = roi_safe_name_base(n, mz_val, q3_val, compound_name=compound_label or None)

        fig = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(plot_rt, plot_intensity, color="blue", linewidth=1.5)
        if rt_end_min > rt_start_min and len(plot_rt) > 0:
            ax.set_xlim(rt_start_min, rt_end_min)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        roi_path = os.path.join(output_dir, f"{safe_name_base}.jpeg")
        canvas.print_jpeg(roi_path)
        roi_windows_rows.append({"image": f"{safe_name_base}.jpeg", "rt_lo": rt_start_min, "rt_hi": rt_end_min})
        print(f"[INFO] Saved ROI ({rt_start_min:.2f}~{rt_end_min:.2f} min): {roi_path}")

    df_features = pd.DataFrame(features)
    feature_csv = os.path.join(output_dir, "feature.csv")
    df_features.to_csv(feature_csv, index=False)
    print(f"[INFO] Saved feature.csv: {feature_csv}")

    if roi_windows_rows:
        df_roi_windows = pd.DataFrame(roi_windows_rows)
        roi_windows_csv = os.path.join(output_dir, "roi_windows.csv")
        df_roi_windows.to_csv(roi_windows_csv, index=False)
        print(f"[INFO] Saved roi_windows.csv: {roi_windows_csv}")

    if intensity_matrix and common_rt is not None:
        intensity_array = np.array(intensity_matrix)
        xic_full = np.vstack([common_rt, intensity_array])
        npy_path = os.path.join(output_dir, "xic_matrix.npy")
        np.save(npy_path, xic_full)
        print(f"[INFO] Saved xic_matrix.npy, shape={xic_full.shape}")

    print(f"[DONE] Processed {len(features)} compounds from arrays.")
    return output_dir


def generate_prediction_plots(images_path, model_path, threshold=0.99, plot_dir="predicted_plots"):
    """ 使用 MRMPFormer 模型对刚生成的 ROI 图像进行推理，并输出带预测框的可视化图。 """
    from utils.predict_utils import build_predictor
    print("[INFO] Running predictor for ROI visualization...")
    results = build_predictor(
        model_path=model_path,
        images_path=images_path,
        threshold=threshold,
        plot=True,
        plot_dir=plot_dir
    )
    print(f"[INFO] Prediction visualization done. {len(results)} images with detections.")


def run_batch_mzml(
    batch_dir,
    output_base,
    smooth_sigma=0.0,
    model_path=None,
    plot_predictions=False,
    threshold=0.99,
    plot_dir="predicted_plots",
    standard_rt_key_to_rt=None,
    standard_rt_mz_to_rt=None,
    min_chrom_points=0,
    min_max_intensity=0.0,
):
    """
    批量处理 batch_dir 中所有 .mzml / .mzML 文件，每个文件的结果保存到 output_base/<文件名(无扩展名)>/
    """
    batch_path = Path(batch_dir)
    output_base_path = Path(output_base)
    if not batch_path.exists():
        raise FileNotFoundError(f"[ERROR] Batch directory not found: {batch_path}")

    mzml_files = list(batch_path.glob("*.mzml")) + list(batch_path.glob("*.mzML"))
    mzml_files = sorted(set(mzml_files))  # 去重（Windows 不区分大小写时可能重复）

    if not mzml_files:
        print(f"[WARN] No .mzml or .mzML files found in {batch_path}")
        return []

    print(f"[INFO] Found {len(mzml_files)} mzML file(s) in {batch_path}")
    output_base_path.mkdir(parents=True, exist_ok=True)
    batch_stats = []

    for mzml_path in mzml_files:
        stem = mzml_path.stem  # 文件名（无扩展名）
        out_dir = output_base_path / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        mzml_abs = mzml_path.resolve()
        out_abs = out_dir.resolve()
        print("=" * 60)
        print(f"[BATCH] Processing: {mzml_path.name} -> {out_dir}")
        print("=" * 60)
        st = extract_xic_with_pyopenms(
            str(mzml_abs),
            str(out_abs),
            smooth_sigma=smooth_sigma,
            standard_rt_key_to_rt=standard_rt_key_to_rt,
            standard_rt_mz_to_rt=standard_rt_mz_to_rt,
            min_chrom_points=min_chrom_points,
            min_max_intensity=min_max_intensity,
        )
        if st:
            batch_stats.append({"stem": stem, **st})
        if plot_predictions and model_path:
            generate_prediction_plots(
                images_path=str(out_dir),
                model_path=model_path,
                threshold=threshold,
                plot_dir=str(out_dir / plot_dir)
            )
    total_roi = sum(s.get("n_roi_images", 0) for s in batch_stats)
    print(
        f"[DONE] Batch completed. {len(mzml_files)} mzML file(s), "
        f"合计生成 ROI 图 {total_roi} 张."
    )
    return batch_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract XIC ROI images and optionally generate prediction plots with boxes."
    )
    parser.add_argument(
        "--mzml", type=str,
        default=None,
        help="Path to single input mzML file (ignored when --batch_dir is set)"
    )
    default_batch_dir = str(TRUEDATA_TEST_DIR)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: process all .mzml/.mzML in truedata/test, save each to output_dir/<mzml_filename>/"
    )
    parser.add_argument(
        "--batch_dir", type=str,
        default=None,
        help="Directory containing .mzml/.mzML files (default when --batch: truedata/test)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(ROOT_DIR.parent.parent / "output" / "inference" / "xic-roi-batch"),
        help="Base output directory (default: <repo>/output/inference/xic-roi-batch). Single mzML: results go under output_dir/<stem>/ unless --flat_output. Batch: one subfolder per mzML (stem = filename without extension)."
    )
    parser.add_argument(
        "--flat_output",
        action="store_true",
        help="Single mzML only: write ROI/csv/npy directly into --output_dir without creating a subfolder named after the mzML file.",
    )
    parser.add_argument(
        "--smooth_sigma", type=float, default=0.0,
        help="Gaussian smoothing sigma (0 means no smoothing)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model checkpoint path for optional prediction plot generation"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.99,
        help="Confidence threshold for optional prediction plot generation"
    )
    parser.add_argument(
        "--plot_predictions", action="store_true",
        help="Generate prediction plots with bounding boxes after ROI extraction"
    )
    parser.add_argument(
        "--plot_dir", type=str, default="predicted_plots",
        help="Directory name for prediction plots (relative to each output folder in batch)"
    )
    parser.add_argument(
        "--from_json", type=str, default=None,
        help="External输入：JSON 中每化合物含 mz_name, rt, intensity；可选 q3。expected_rt 已忽略。"
    )
    args = parser.parse_args()

    # 外部数组输入模式：--from_json 指定 JSON 文件
    if args.from_json:
        import json
        json_path = Path(args.from_json).resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"[ERROR] JSON file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        compounds = data if isinstance(data, list) else data.get("compounds", data.get("data", []))
        if not compounds:
            raise ValueError("[ERROR] No compounds in JSON. Expected list of {mz_name, rt, intensity, q3?, expected_rt?}")
        extract_xic_from_arrays(args.output_dir, compounds, smooth_sigma=args.smooth_sigma)
        if args.plot_predictions and args.model:
            generate_prediction_plots(
                images_path=args.output_dir,
                model_path=args.model,
                threshold=args.threshold,
                plot_dir=args.plot_dir,
            )
    else:
        # 批量模式：--batch 或显式指定 --batch_dir
        batch_dir = args.batch_dir
        if args.batch and batch_dir is None:
            batch_dir = default_batch_dir
        if batch_dir:
            output_base = args.output_dir
            run_batch_mzml(
                batch_dir=batch_dir,
                output_base=output_base,
                smooth_sigma=args.smooth_sigma,
                model_path=args.model,
                plot_predictions=args.plot_predictions,
                threshold=args.threshold,
                plot_dir=args.plot_dir,
            )
        else:
            # 单文件模式：默认取 truedata/test 下第一个 .mzML
            mzml = args.mzml
            if not mzml:
                mzml_files = list(TRUEDATA_TEST_DIR.glob("*.mzML")) + list(TRUEDATA_TEST_DIR.glob("*.mzml"))
                mzml = str(mzml_files[0]) if mzml_files else str(TRUEDATA_TEST_DIR / "sample.mzML")
            if not os.path.exists(mzml):
                raise FileNotFoundError(f"[ERROR] mzML file not found: {mzml}")
            mzml_path_obj = Path(mzml).resolve()
            if args.flat_output:
                out_dir = Path(args.output_dir)
            else:
                out_dir = Path(args.output_dir) / mzml_path_obj.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Output directory: {out_dir}")
            extract_xic_with_pyopenms(
                str(mzml_path_obj),
                str(out_dir),
                smooth_sigma=args.smooth_sigma,
            )
            if args.plot_predictions:
                if not args.model:
                    raise ValueError("[ERROR] --plot_predictions requires --model to be provided.")
                generate_prediction_plots(
                    images_path=str(out_dir),
                    model_path=args.model,
                    threshold=args.threshold,
                    plot_dir=str(out_dir / args.plot_dir),
                )