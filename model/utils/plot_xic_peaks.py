# -*- coding: utf-8 -*-
"""
共享 XIC 峰标注绘图核心（需求 2/7）。

供两处使用，保证图样式一致：
  1) predictor model_plots/（模型推理输出，--plot_style xic 默认）；
  2) peak_refinement refined_plots/（SNR 筛选 + 框修正后）。

规格：
  - 横轴 RT (min)、纵轴 Intensity（高斯平滑后非负裁剪）；
  - 每个 query 一个不同颜色 axvspan 阴影（第 1 个淡绿 #2ecc71，延续旧 Main interval 视觉）；
  - 置信度不进阴影文字，统一写入左上角信息框（每 query 一段）；
  - 信息框字段：保留时间 RT / 母离子 m/z(Q1) / 峰高 / 信噪比 / 峰区间扫描点数 / 置信度。
"""
import numpy as np


# 与 peak_refinement 现有 axvspan 色板对齐：第 1 个淡绿，其余按序循环
_QUERY_COLORS = [
    ("#2ecc71", 0.24),
    ("#f39c12", 0.22),
    ("#8e44ad", 0.18),
    ("#16a085", 0.16),
    ("#2980b9", 0.14),
    ("#c0392b", 0.12),
    ("#95a5a6", 0.20),
]


def _smooth_nonneg(y: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    s = float(sigma)
    if s > 0 and y.size > 2:
        s = min(s, y.size / 25.0)
        y = gaussian_filter1d(y.astype(np.float64), sigma=s)
    return np.clip(y, 0.0, None)


def _scan_points_in_interval(rt: np.ndarray, y: np.ndarray, rt_lo: float, rt_hi: float) -> int:
    """峰区间 [rt_lo, rt_hi] 内强度>0 的最长连续点数（不是整张 ROI 图）。"""
    mask = (rt >= rt_lo) & (rt <= rt_hi)
    vals = y[mask] > 0
    if vals.size == 0:
        return 0
    best = cur = 0
    for v in vals:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def plot_xic_with_queries(
    ax,
    rt: np.ndarray,
    intensity: np.ndarray,
    queries,
    q1=None,
    sigma: float = 1.0,
    title: str = "",
    roi_window=None,
    xlabel: str = "Retention Time (min)",
    ylabel: str = "Intensity",
):
    """在已有 Axes 上绘制 XIC 曲线与多 query 峰标注。

    参数:
        ax: matplotlib Axes（调用方创建，本函数只负责内容）
        rt: RT 数组（分钟；秒制由调用方先 /60）
        intensity: 原始强度数组（函数内平滑+非负）
        queries: list[dict]，每元素建议含
            {rt_lo, rt_hi, rt_peak, height, snr, n_points, score}
            （n_points 缺失时按区间内强度>0 最长连续点数计算）
        q1: 母离子 m/z（信息框显示用）
        roi_window: (rt_lo, rt_hi) 显示窗口；None 时用数据范围
    """
    import matplotlib

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    rt = np.asarray(rt, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    y = _smooth_nonneg(intensity, sigma)

    if roi_window is not None:
        w_lo, w_hi = float(roi_window[0]), float(roi_window[1])
    else:
        w_lo, w_hi = float(np.min(rt)), float(np.max(rt))
    mask = (rt >= w_lo) & (rt <= w_hi)
    x_plot = rt[mask]
    y_plot = y[mask]
    if x_plot.size < 2:
        x_plot, y_plot = rt, y

    ax.plot(x_plot, y_plot, color="blue", linewidth=1.5)
    if x_plot.size > 1:
        ax.set_xlim(float(x_plot[0]), float(x_plot[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)

    # 逐 query 阴影（不同颜色）+ 图例 label（含置信度）
    legend_handles = []
    qlist = list(queries) if queries else []
    for k, q in enumerate(qlist):
        color, alpha = _QUERY_COLORS[k % len(_QUERY_COLORS)]
        lo = float(q.get("rt_lo"))
        hi = float(q.get("rt_hi"))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        sc = q.get("score")
        sc_s = "%.3f" % float(sc) if sc is not None and np.isfinite(float(sc)) else "—"
        ax.axvspan(lo, hi, color=color, alpha=alpha,
                   label="query#%d (score=%s)" % (k + 1, sc_s))
        legend_handles.append((lo, hi, color, k + 1, sc_s))
        # 峰顶竖线（同色点线）
        pk = q.get("rt_peak")
        if pk is not None and np.isfinite(float(pk)):
            ax.axvline(float(pk), color=color, linestyle=":", linewidth=1.0, alpha=0.9)
    if qlist:
        ax.legend(loc="upper right", fontsize=8)

    # 左上角信息框：每 query 一段（RT/Q1/峰高/SNR/区间扫描点数/置信度）
    if qlist:
        blocks = []
        for k, q in enumerate(qlist):
            lo = float(q.get("rt_lo"))
            hi = float(q.get("rt_hi"))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue
            rt_pk = q.get("rt_peak")
            rt_s = "%.4f" % float(rt_pk) if rt_pk is not None and np.isfinite(float(rt_pk)) else "—"
            h = q.get("height")
            h_s = "%.6g" % float(h) if h is not None and np.isfinite(float(h)) else "—"
            snr = q.get("snr")
            snr_s = "%.4g" % float(snr) if snr is not None and np.isfinite(float(snr)) else "—"
            np_ = q.get("n_points")
            if np_ is None:
                np_ = _scan_points_in_interval(rt, y, lo, hi)
            np_s = "%d" % int(np_) if np_ is not None and np.isfinite(float(np_)) else "—"
            sc = q.get("score")
            sc_s = "%.3f" % float(sc) if sc is not None and np.isfinite(float(sc)) else "—"
            q1_s = "%.4f" % float(q1) if q1 is not None and np.isfinite(float(q1)) else "—"
            blocks.append(
                "query#%d  RT=%s min  Q1(m/z)=%s\n峰高=%s  SNR=%s  区间点数=%s  置信度=%s"
                % (k + 1, rt_s, q1_s, h_s, snr_s, np_s, sc_s)
            )
        info = "\n\n".join(blocks)
        ax.text(
            0.02, 0.98, info,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.92),
            zorder=10,
        )
