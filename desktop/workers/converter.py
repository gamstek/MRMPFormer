"""
converter.py — msdata → mzML 格式转换后台线程
==============================================
封装 MsdataConverter(QThread)，异步调用内嵌的 bin/msdata2mzml.exe，
逐文件将 .msdata 转换为 .mzML。通过 Signal 驱动 UI 实时更新文件级状态。

核心流程:
  1. 设置 OPENMS_DATA_PATH → bin/share/OpenMS
  2. subprocess.run([msdata2mzml.exe, input_file])
  3. 检测输出目录是否有 .mzML 文件判定成功/失败
  4. 逐文件发出 progress + file_done 信号

依赖: subprocess, os, pathlib, PySide6.QtCore
"""

import os
import subprocess
from pathlib import Path
from PySide6.QtCore import QThread, Signal


# 定位 bin/ 目录（相对于本文件向上两级: workers/ → desktop/）
_BIN_DIR = Path(__file__).resolve().parent.parent / "bin"
_EXE_PATH = _BIN_DIR / "msdata2mzml.exe"
_SHARE_PATH = _BIN_DIR / "share" / "OpenMS"


class MsdataConverter(QThread):
    """
    格式转换后台线程。

    逐文件调用 msdata2mzml.exe 进行 msdata→mzML 转换。
    每个文件完成后发出 file_done 信号，整体完毕后发出 progress(total, total)。
    """

    # (current_index: int, total: int) — 当前处理到第几个文件（0-based）
    progress = Signal(int, int)
    # (index: int, success: bool, info: str) — 文件级结果，info 为文件大小或错误消息
    file_done = Signal(int, bool, str)
    # (message: str) — 全局致命错误（如 exe 不存在）
    error = Signal(str)

    def __init__(self, files: list[str], output_dir: str | None = None, parent=None):
        """
        Args:
            files: .msdata 文件绝对路径列表
            output_dir: 自定义输出目录，None=默认同输入目录
            parent: Qt parent object
        """
        super().__init__(parent)
        self._files = [Path(f) for f in files]
        self._output_dir = Path(output_dir) if output_dir else None

    def run(self):
        """在线程中执行批量转换。"""
        # 前置检查：exe 是否存在
        if not _EXE_PATH.exists():
            self.error.emit(f"未找到转换工具: {_EXE_PATH}\n请检查 bin/ 目录")
            return

        total = len(self._files)
        env = os.environ.copy()
        env["OPENMS_DATA_PATH"] = str(_SHARE_PATH)

        for i, file_path in enumerate(self._files):
            # 发出进度信号
            self.progress.emit(i, total)

            # 构建命令：以 bin/ 为工作目录，传相对路径
            try:
                rel_input = os.path.relpath(str(file_path), str(_BIN_DIR))
            except ValueError:
                # 不同盘符时 relpath 会抛异常，此时用绝对路径
                rel_input = str(file_path)

            cmd = [str(_EXE_PATH), rel_input]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    cwd=str(_BIN_DIR),
                    timeout=600,  # 单文件最多 10 分钟
                )
            except subprocess.TimeoutExpired:
                self.file_done.emit(i, False, "转换超时 (>10分钟)")
                continue
            except Exception as e:
                self.file_done.emit(i, False, f"进程异常: {e}")
                continue

            # 检查输出：预期在输入文件同级目录下的 <stem>/ 中生成 .mzML
            expected_dir = file_path.parent / file_path.stem
            if self._output_dir:
                expected_dir = self._output_dir / file_path.stem

            mzml_files = list(expected_dir.glob("*.mzML")) if expected_dir.exists() else []

            if mzml_files:
                total_bytes = sum(f.stat().st_size for f in mzml_files)
                size_mb = total_bytes / (1024 * 1024)
                info = f"{len(mzml_files)} 个 .mzML, {size_mb:.1f} MB"
                self.file_done.emit(i, True, info)
            else:
                stderr = result.stderr.strip() if result.stderr else ""
                stdout = result.stdout.strip() if result.stdout else ""
                reason = stderr or stdout or f"退码 {result.returncode}，未生成 mzML"
                # 检测中文路径问题
                if "path" in reason.lower() or "不存在" in reason or "not exist" in reason.lower():
                    reason += " (路径含中文字符可能导致 OpenMS 失败)"
                self.file_done.emit(i, False, reason)

        # 最终进度 (total, total) — 通知 UI 全部完成
        self.progress.emit(total, total)
