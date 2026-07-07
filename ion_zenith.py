"""
ion_zenith.py
=============
从 mzML 的 MS1 谱图中提取每个 m/z 的信号顶点（最高强度），
输出 (m/z, RT, intensity) 精简表。

"Zenith" = 天顶 / 顶点 —— 每个离子在整个色谱过程中强度最高的那一刻。

离子天顶算法
----
遍历 MS1 谱图 → 按 m/z 容差合并 → 保留最高强度 → 按强度范围过滤 → 输出 CSV。

所有参数在下方全局配置区修改，直接 `python extract_ms1_mz_rt.py` 运行。

依赖: pip install pymzml numpy
"""

import csv
import math
import os
import time
import numpy as np
import pymzml
import tqdm

# ============================================================
# ★ 全局配置 —— 修改这里即可，不需要命令行参数
# ============================================================

# --- 输入输出 ---
INPUT_MZML  = r"data/test2/mzML/20260423_001.mzML"   # 输入 mzML 文件路径
OUTPUT_CSV  = r"data/test2/ion_zenith_output.csv"     # 输出 CSV

# --- m/z 范围 ---
MZ_MIN  = 50.0       # 最小 m/z
MZ_MAX  = 2000.0     # 最大 m/z

# --- m/z 分箱容差（合并相近离子） ---
MZ_TOLERANCE_PPM = 10.0   # ppm 容差
MZ_TOLERANCE_DA  = 0.01   # Da 容差（低 m/z 区与 ppm 取较大者）

# --- 强度过滤 ---
# 三种模式（三选一，其余两个设为 None）：
#   - 只保留强度 >= INTENSITY_MIN  （设 INTENSITY_MIN=值, INTENSITY_MAX=None）
#   - 只保留强度 <= INTENSITY_MAX  （设 INTENSITY_MIN=None, INTENSITY_MAX=值）
#   - 只保留强度在 [MIN, MAX] 之间 （两个都设）
#   - 不过滤                      （两个都设 None）
INTENSITY_MIN = None   # 最小强度（含），None=不限制下限
INTENSITY_MAX = None   # 最大强度（含），None=不限制上限

# --- 扫描控制 ---
MAX_SPECTRA = 0        # 最大扫描谱图数，0=不限制（扫描全部）

# --- 其他 ---
BUILD_INDEX  = False   # 是否重建 mzML 索引
SHOW_PROGRESS = True   # 是否显示进度条（需 pip install tqdm）

# ============================================================
# 谱图读取（最小化，不用 XML 解析）
# ============================================================

def _parse_rt(spectrum):
    """从 pymzML spectrum 获取 RT，返回 (rt_min, rt_sec) 或 (None, None)。"""
    st = getattr(spectrum, "scan_time", None)
    if st is None:
        return None, None
    if isinstance(st, (tuple, list)) and len(st) >= 2:
        val, unit = float(st[0]), str(st[1]).lower()
    else:
        val, unit = float(st), ""
    if math.isnan(val) or math.isinf(val):
        return None, None
    if "min" in unit:
        return val, val * 60.0
    return val / 60.0, val  # 默认当秒处理


def _peaks(spectrum):
    """获取 (m/z, intensity) 二维 numpy 数组，失败返回 None。"""
    p = getattr(spectrum, "peaks", None)
    if p is not None and len(p) > 0:
        return np.asarray(p, dtype=np.float64)
    mz_arr = getattr(spectrum, "mz", None)
    int_arr = getattr(spectrum, "i", None)
    if mz_arr is not None and int_arr is not None:
        return np.column_stack((np.asarray(mz_arr, dtype=np.float64),
                                np.asarray(int_arr, dtype=np.float64)))
    return None


# ============================================================
# 主流程
# ============================================================

def main():
    t0 = time.perf_counter()

    # ---- 打印配置 ----
    print(f"输入:   {INPUT_MZML}")
    print(f"输出:   {OUTPUT_CSV}")
    print(f"m/z:    {MZ_MIN}–{MZ_MAX}  (容差 {MZ_TOLERANCE_PPM} ppm / {MZ_TOLERANCE_DA} Da)")
    if INTENSITY_MIN is not None and INTENSITY_MAX is not None:
        print(f"强度:   {INTENSITY_MIN} ≤ I ≤ {INTENSITY_MAX}")
    elif INTENSITY_MIN is not None:
        print(f"强度:   I ≥ {INTENSITY_MIN}")
    elif INTENSITY_MAX is not None:
        print(f"强度:   I ≤ {INTENSITY_MAX}")
    else:
        print(f"强度:   不过滤")
    print(f"谱图:   {'全部' if MAX_SPECTRA == 0 else f'前 {MAX_SPECTRA} 张'}")
    print(f"{'─'*50}")

    limit = MAX_SPECTRA if MAX_SPECTRA > 0 else None
    run = pymzml.run.Reader(INPUT_MZML, build_index_from_scratch=BUILD_INDEX)

    # best: mz_key -> [mz_center, rt_min, rt_sec, max_intensity, count]
    best = {}
    n_spec = 0     # 总谱图计数
    n_ms1 = 0      # MS1 谱图计数
    n_peaks = 0    # 扫描峰总数

    use_tqdm = SHOW_PROGRESS and tqdm is not None
    pbar = tqdm(desc="扫描中", unit="spectrum") if use_tqdm else None

    for spectrum in run:
        n_spec += 1

        # ---- 只处理 MS1 ----
        ms_level = getattr(spectrum, "ms_level", None)
        if ms_level != 1:
            if pbar: pbar.update(1)
            if limit and n_spec >= limit: break
            continue

        # ---- RT ----
        rt_min, rt_sec = _parse_rt(spectrum)
        if rt_sec is None:
            if pbar: pbar.update(1)
            if limit and n_spec >= limit: break
            continue

        # ---- 峰数组 ----
        arr = _peaks(spectrum)
        if arr is None or arr.size == 0:
            if pbar: pbar.update(1)
            if limit and n_spec >= limit: break
            continue

        n_ms1 += 1
        mz_vals = arr[:, 0]
        int_vals = arr[:, 1]

        # ---- 强度过滤（在聚合前过滤，减少内存） ----
        mask = np.ones(len(mz_vals), dtype=bool)
        mask &= (mz_vals >= MZ_MIN) & (mz_vals <= MZ_MAX)
        mask &= (int_vals > 0)
        if INTENSITY_MIN is not None:
            mask &= (int_vals >= INTENSITY_MIN)
        if INTENSITY_MAX is not None:
            mask &= (int_vals <= INTENSITY_MAX)

        idx = np.where(mask)[0]
        if len(idx) == 0:
            if pbar: pbar.update(1)
            if limit and n_spec >= limit: break
            continue

        mz_filt = mz_vals[idx]
        int_filt = int_vals[idx]

        # ---- 聚合：同一 m/z key 只保留最高强度 ----
        for j in range(len(mz_filt)):
            mz = float(mz_filt[j])
            intensity = float(int_filt[j])
            n_peaks += 1

            # m/z 分箱 key
            tol = max(mz * MZ_TOLERANCE_PPM * 1e-6, MZ_TOLERANCE_DA)
            key = round(mz / tol) * tol

            if key not in best or intensity > best[key][3]:
                best[key] = [mz, rt_min, rt_sec, intensity, 1]
            else:
                best[key][4] += 1

        if pbar: pbar.update(1)
        if limit and n_spec >= limit:
            break

    if pbar: pbar.close()

    # ---- 按 m/z 排序写入 CSV ----
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    rows = sorted(best.values(), key=lambda x: x[0])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["mz", "rt_min", "rt_sec", "max_intensity", "n_observations"])
        buf = []
        for mz_val, rt_m, rt_s, max_i, cnt in rows:
            buf.append([
                round(mz_val, 6),
                round(rt_m, 6) if rt_m is not None else "",
                round(rt_s, 3) if rt_s is not None else "",
                round(max_i, 1),
                cnt,
            ])
            if len(buf) >= 50000:
                w.writerows(buf); buf.clear()
        if buf:
            w.writerows(buf)

    elapsed = time.perf_counter() - t0

    print(f"总谱图:   {n_spec}")
    print(f"MS1:      {n_ms1}")
    print(f"扫描峰:   {n_peaks}")
    print(f"去重后:   {len(rows)} 个 m/z")
    print(f"耗时:     {elapsed:.2f} s")
    print(f"输出:     {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
