# -*- coding: utf-8 -*-
"""
MRMPFormer 运行时日志过滤工具。

用法:
    from mrmpformer.util.logutil import configure_log_level, LOG_LEVELS

    # 只显示 WARNING 与 ERROR（默认行为）
    configure_log_level("WARNING")

    # 显示所有日志（调试用）
    configure_log_level("INFO")

    # 环境变量控制（优先级最高）
    # export MRMPFORMER_LOG_LEVEL=INFO

设计思路：
    项目使用 print("[INFO] ...") / print("[WARN] ...") / print("[ERROR] ...") 模式。
    本模块通过替换 sys.stdout 为带缓冲的过滤器，按行前缀级别拦截输出，
    无需改动任何现有 print 调用。
"""

import os
import sys
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# 日志级别常量
# ---------------------------------------------------------------------------
LOG_LEVELS = {
    "DEBUG": 0,
    "INFO": 1,
    "WARNING": 2,
    "ERROR": 3,
    "NONE": 99,  # 静默全部
}

_PREFIX_LEVELS = [
    ("[ERROR]", "ERROR"),
    ("[WARN]", "WARNING"),
    ("[INFO]", "INFO"),
    ("[DEBUG]", "DEBUG"),
]

_current_level: int = LOG_LEVELS["WARNING"]  # 默认：抑制 INFO
_original_stdout: Optional[object] = None
_filter_installed: bool = False
_install_lock = threading.Lock()


def configure_log_level(level: str) -> None:
    """设置全局日志级别。

    Args:
        level: "DEBUG" | "INFO" | "WARNING" | "ERROR" | "NONE"

    环境变量 MRMPFORMER_LOG_LEVEL 优先级高于此函数调用。
    """
    global _current_level
    env_level = os.environ.get("MRMPFORMER_LOG_LEVEL", "").strip().upper()
    effective = env_level if env_level in LOG_LEVELS else level.upper()
    if effective not in LOG_LEVELS:
        effective = "WARNING"
    _current_level = LOG_LEVELS[effective]


def get_log_level() -> str:
    """返回当前日志级别名称。"""
    for name, val in LOG_LEVELS.items():
        if val == _current_level:
            return name
    return "WARNING"


# ---------------------------------------------------------------------------
# 带缓冲的 stdout 过滤器
# ---------------------------------------------------------------------------
class _FilteredStdout:
    """按行缓冲，根据前缀级别决定是否写入原始 stdout。"""

    def __init__(self, original):
        self._orig = original
        self._buf = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buf += text
        if "\n" in text:
            self._flush()
        # 如果缓冲区过长且无换行（不太可能），也强制刷新
        elif len(self._buf) > 4096:
            self._flush()
        return len(text)

    def _flush(self) -> None:
        if not self._buf:
            return
        lines = self._buf.split("\n")
        # 最后一段可能是未完成的行，保留在缓冲区
        self._buf = lines.pop()
        for line in lines:
            self._write_line(line + "\n")

    def _write_line(self, line: str) -> None:
        level_val = _classify_line(line)
        if level_val >= _current_level:
            self._orig.write(line)

    def flush(self) -> None:
        if self._buf:
            self._orig.write(self._buf)
            self._buf = ""
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


def _classify_line(line: str) -> int:
    """返回行对应的日志级别数值；无法识别时返回 DEBUG（放行）。"""
    for prefix, level_name in _PREFIX_LEVELS:
        if prefix in line:
            return LOG_LEVELS[level_name]
    # 无标签行（如纯输出、进度条、分隔线等）——放行
    return LOG_LEVELS["DEBUG"]


def install_filter() -> None:
    """全局安装 stdout 过滤器（幂等）。"""
    global _filter_installed, _original_stdout
    with _install_lock:
        if _filter_installed:
            return
        _original_stdout = sys.stdout
        sys.stdout = _FilteredStdout(_original_stdout)
        _filter_installed = True


def uninstall_filter() -> None:
    """卸载过滤器，恢复原始 stdout。"""
    global _filter_installed, _original_stdout
    with _install_lock:
        if not _filter_installed or _original_stdout is None:
            return
        # 刷新缓冲区
        if hasattr(sys.stdout, 'flush'):
            sys.stdout.flush()
        sys.stdout = _original_stdout
        _original_stdout = None
        _filter_installed = False
