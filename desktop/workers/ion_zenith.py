"""
ion_zenith.py — 离子天顶算法后台线程（薄包装）
================================================

封装 IonZenithWorker(QThread)，调用 model/preprocessing/ion_zenith.py 中的
纯算法函数 extract_ions_from_ms1()，通过 Signal 将进度/统计/结果/错误转发给前端。

算法核心已迁移至 model/preprocessing/ion_zenith.py，本文件只负责 Qt 线程适配。

依赖: PySide6.QtCore, model.preprocessing.ion_zenith
"""

from PySide6.QtCore import QThread, Signal


class IonZenithWorker(QThread):
    """
    离子天顶算法后台线程。

    读取 mzML 文件，遍历 MS1 谱图，提取每个 m/z 信号顶点（最高强度），
    输出 (m/z, RT, intensity, n_observations) 的 CSV 文件。
    """

    # (scanned: int, total: int) — 已扫描谱图数（total=0 表示未知总数）
    progress = Signal(int, int)
    # (ms1_count: int, total_peaks: int) — MS1 谱图数, 累计扫描峰数
    stats = Signal(int, int)
    # (ion_count: int, elapsed_sec: float, output_path: str) — 任务完成
    finished = Signal(int, float, str)
    # (message: str) — 错误消息
    error = Signal(str)

    def __init__(self, params: dict, parent=None):
        """
        Args:
            params: 参数字典，包含以下键：
                input_mzml (str):      输入 mzML 文件路径（必需）
                output_csv (str):      输出 CSV 路径（必需）
                mz_min (float):        m/z 下限，默认 50.0
                mz_max (float):        m/z 上限，默认 2000.0
                ppm_tol (float):       ppm 容差，默认 10.0
                da_tol (float):        Da 容差，默认 0.01
                intensity_min (float|None): 强度下限，None=不限制
                intensity_max (float|None): 强度上限，None=不限制
                max_spectra (int):     最大扫描谱图数，0=全部，默认 0
                build_index (bool):    是否重建 mzML 索引，默认 False
                show_progress (bool):  是否通过 Signal 发送进度，默认 True
            parent: Qt parent object
        """
        super().__init__(parent)
        self._params = params
        self._cancelled = False  # 预留取消标志（后续版本实现真正的中断逻辑）

    def cancel(self):
        """请求取消当前运行。（当前版本仅设置标志，worker loop 检查此标志）"""
        self._cancelled = True

    def run(self):
        """在线程中执行离子天顶算法。"""
        # 延迟导入：避免 desktop 启动时强依赖 model 包
        import sys
        from pathlib import Path
        model_root = Path(__file__).resolve().parent.parent.parent / "model"
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))

        from preprocessing.ion_zenith import extract_ions_from_ms1

        p = self._params

        # ── 验证必需参数 ──
        input_path = Path(p.get("input_mzml", ""))
        output_path = Path(p.get("output_csv", ""))
        if not input_path.exists():
            self.error.emit(f"输入文件不存在: {input_path}")
            return
        if not output_path.parent.exists():
            self.error.emit(f"输出目录不存在: {output_path.parent}")
            return

        # ── 读取参数（带默认值） ──
        mz_min = float(p.get("mz_min", 50.0))
        mz_max = float(p.get("mz_max", 2000.0))
        ppm_tol = float(p.get("ppm_tol", 10.0))
        da_tol = float(p.get("da_tol", 0.01))
        intensity_min = p.get("intensity_min")  # None or float
        intensity_max = p.get("intensity_max")  # None or float
        max_spectra = int(p.get("max_spectra", 0))
        build_index = bool(p.get("build_index", False))
        show_progress = bool(p.get("show_progress", True))

        if intensity_min is not None:
            intensity_min = float(intensity_min)
        if intensity_max is not None:
            intensity_max = float(intensity_max)

        # ── 进度/统计回调（通过 Signal 转发） ──
        def _on_progress(scanned: int, total: int):
            if show_progress and not self._cancelled:
                self.progress.emit(scanned, total)

        def _on_stats(ms1_count: int, peaks: int):
            if show_progress and not self._cancelled:
                self.stats.emit(ms1_count, peaks)

        # ── 调用纯算法 ──
        try:
            result = extract_ions_from_ms1(
                mzml_path=str(input_path),
                output_csv=str(output_path),
                mz_min=mz_min,
                mz_max=mz_max,
                ppm_tol=ppm_tol,
                da_tol=da_tol,
                intensity_min=intensity_min,
                intensity_max=intensity_max,
                max_spectra=max_spectra,
                build_index=build_index,
                on_progress=_on_progress,
                on_stats=_on_stats,
            )
        except (FileNotFoundError, ValueError) as e:
            self.error.emit(str(e))
            return
        except Exception as e:
            self.error.emit(f"算法执行异常: {e}")
            return

        # ── 完成信号 ──
        self.finished.emit(
            result["n_ions"],
            result["elapsed_sec"],
            result["output_csv"],
        )
