"""
main.py — GAMSTEKPEAKing 应用入口
==================================
启动 QApplication，加载全局 QSS 样式，创建主窗口并注册所有功能页面。

用法:
    python main.py          # 正常启动
    python main.py --debug  # 调试模式（额外日志输出到 stdout）

依赖: PySide6, app, theme, pages.preprocessing, pages.peak_finding, pages.settings
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import QApplication, QMessageBox, QLabel

# 确保 gamstekpeaking 包在 sys.path 中（支持从任意目录运行）
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from app import GAMSTEKPEAKingWindow
from theme import global_stylesheet
from pages.preprocessing import PreprocessingPage
from pages.peak_finding import PeakFindingPage
from pages.settings import SettingsPage


def setup_exception_handler():
    """
    设置全局异常钩子。

    将所有未捕获异常写入 gamstekpeaking/error.log，
    并弹出 QMessageBox 提示用户，避免程序静默崩溃。
    """

    def _handler(exc_type, exc_value, exc_tb):
        # 键盘中断 → 正常退出
        if exc_type is KeyboardInterrupt:
            sys.exit(0)

        # 记录到 error.log
        log_path = _HERE / "error.log"
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"[{datetime.now().isoformat()}] 未捕获异常\n")
            f.writelines(tb_lines)

        # 弹窗提示
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("GAMSTEKPEAKing — 错误")
        msg.setText(f"发生未预期错误:\n{exc_value}\n\n详情已写入 error.log")
        msg.exec()

    sys.excepthook = _handler


def main():
    """应用主入口：创建 QApplication → 加载主题 → 构建窗口 → 注册页面 → 启动事件循环。"""
    setup_exception_handler()

    debug = "--debug" in sys.argv

    # 创建 QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("GAMSTEKPEAKing")
    app.setOrganizationName("LinShuhaiLAB")

    # 设置亮色调色板（确保按钮文字/箭头等系统绘制元素为深色）
    from PySide6.QtGui import QPalette, QColor
    palette = app.palette()
    palette.setColor(QPalette.ButtonText, QColor("#1F2937"))
    palette.setColor(QPalette.Button, QColor("#E5E7EB"))
    palette.setColor(QPalette.Base, QColor("#FFFFFF"))
    palette.setColor(QPalette.Text, QColor("#1F2937"))
    palette.setColor(QPalette.Window, QColor("#F8F9FA"))
    palette.setColor(QPalette.WindowText, QColor("#1F2937"))
    app.setPalette(palette)

    # 加载全局 QSS 样式
    app.setStyleSheet(global_stylesheet())

    # 创建主窗口
    window = GAMSTEKPEAKingWindow()

    # 注册功能页面（顺序决定侧边栏排列）
    window.add_page("前处理", PreprocessingPage(), enabled=True)
    window.add_page("寻峰", PeakFindingPage(), enabled=False)
    window.add_page("定量", QLabel("定量分析 — 敬请期待"), enabled=False)
    window.add_page("模型", QLabel("模型管理 — 敬请期待"), enabled=False)
    window.add_page("设置", SettingsPage(), enabled=False)

    window.show()

    if debug:
        print("[DEBUG] GAMSTEKPEAKing 已启动 (调试模式)")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
