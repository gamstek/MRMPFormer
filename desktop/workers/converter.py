"""
converter.py — msdata / wiff → mzML 格式转换后台线程（Qt 薄包装）
==================================================================
封装 FormatConverter(QThread)，异步调用 converters/ 下的纯转换函数
（converters/msdata.py 与 converters/wiff.py），通过 Signal 驱动 UI 实时更新文件级状态。
本文件只负责 Qt 线程适配与信号转发，不包含任何转换逻辑
（bin 定位 / OPENMS_DATA_PATH / subprocess 均在 converters/ 中）。

核心流程:
  1. 在 run() 中确保仓库根目录在 sys.path，延迟导入 converters 模块
  2. 逐文件调用 convert_file()（按格式路由 msdata / wiff）
  3. 逐文件发出 progress + file_done 信号

依赖: sys, pathlib, PySide6.QtCore
"""

import logging
import sys
from pathlib import Path
from PySide6.QtCore import QThread, Signal


logger = logging.getLogger(__name__)


# 仓库根目录（desktop/workers/../..），converters/ 位于其中
ROOT = Path(__file__).resolve().parents[2]

# 各格式 → (转换模块, 可执行文件名)，exe 存在性由模块内 BIN_DIR 解析
_FORMAT_MODULES = {
    "msdata": ("converters.msdata", "msdata2mzml.exe"),
    "wiff": ("converters.wiff", "msconvert.exe"),
}


class FormatConverter(QThread):
    """
    格式转换后台线程（msdata / wiff → mzML）。

    逐文件调用 converters/ 中对应模块的 convert_file()。
    每个文件完成后发出 file_done 信号，整体完毕后发出 progress(total, total)。
    """

    # (current_index: int, total: int) — 当前处理到第几个文件（0-based）
    progress = Signal(int, int)
    # (index: int, success: bool, info: str) — 文件级结果，info 为文件大小或错误消息
    file_done = Signal(int, bool, str)
    # (message: str) — 全局致命错误（如转换工具不存在）
    error = Signal(str)

    def __init__(self, files: list[str], fmt: str = "msdata",
                 output_dir: str | None = None, parent=None):
        """
        Args:
            files: 待转换文件绝对路径列表
            fmt: 源格式 ("msdata" | "wiff")
            output_dir: 自定义输出目录，None=默认同输入目录
            parent: Qt parent object
        """
        super().__init__(parent)
        self._files = [Path(f) for f in files]
        self._fmt = fmt
        self._output_dir = Path(output_dir) if output_dir else None

    def run(self):
        """在线程中执行批量转换。"""
        logger.info("格式转换线程启动: fmt=%s, 文件数=%d, output_dir=%s",
                    self._fmt, len(self._files), self._output_dir)

        # 延迟导入：避免 desktop 启动时强依赖仓库根目录
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        module_name, exe_name = _FORMAT_MODULES.get(self._fmt, _FORMAT_MODULES["msdata"])
        try:
            mod = __import__(module_name, fromlist=["convert_file"])
        except ImportError as e:
            logger.error("无法加载转换模块 %s: %s", module_name, e)
            self.error.emit(f"无法加载转换模块 {module_name}: {e}")
            return

        # 前置检查：exe 是否存在
        exe_path = Path(mod.BIN_DIR) / exe_name
        if not exe_path.exists():
            logger.error("未找到转换工具: %s", exe_path)
            self.error.emit(f"未找到转换工具: {exe_path}\n请检查 {exe_path.parent} 目录")
            return
        logger.info("使用转换工具: %s", exe_path)

        convert_file = mod.convert_file
        total = len(self._files)

        for i, file_path in enumerate(self._files):
            # 发出进度信号
            self.progress.emit(i, total)
            logger.info("[%d/%d] 开始转换: %s", i + 1, total, file_path)

            try:
                ok, info = convert_file(
                    file_path,
                    output_dir=self._output_dir,
                    timeout=600,  # 单文件最多 10 分钟
                )
            except Exception as e:
                logger.error("[%d/%d] 转换异常: %s, 异常=%r", i + 1, total, file_path, e)
                self.file_done.emit(i, False, f"进程异常: {e}")
                continue

            # 中文路径可能导致底层 C++ 转换失败，补充提示
            if not ok and any(k in info.lower() for k in ("path", "not exist", "不存在")):
                info += " (路径含中文字符可能导致转换失败)"
            logger.info("[%d/%d] 转换结果: success=%s, %s", i + 1, total, ok, info)
            self.file_done.emit(i, ok, info)

        # 最终进度 (total, total) — 通知 UI 全部完成
        logger.info("格式转换线程完成: 共 %d 个文件", total)
        self.progress.emit(total, total)
