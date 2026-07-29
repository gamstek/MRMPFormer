"""
peak_finding.py — 寻峰板块（占位）
==================================
未来整合 DETR 模型预测、EIC 可视化、峰面积定量等寻峰功能。
当前为占位页面，显示"敬请期待"。

依赖: PySide6, theme
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from theme import Colors, Fonts


class PeakFindingPage(QWidget):
    """寻峰板块占位页面。后续将加载 DETR 模型并实现交互式峰检测。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        icon = QLabel("📊")
        icon.setFont(QFont(Fonts.primary, 48))
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(f"color: {Colors.text_disabled}; border: none; background: none;")
        layout.addWidget(icon)

        text = QLabel("寻峰功能 — 敬请期待")
        text.setFont(QFont(Fonts.primary, 16))
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet(f"color: {Colors.text_disabled}; border: none; background: none;")
        layout.addWidget(text)

        hint = QLabel("将整合 DETR 模型预测、EIC 可视化、峰面积定量")
        hint.setFont(QFont(Fonts.primary, 12))
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {Colors.text_disabled}; border: none; background: none;")
        layout.addWidget(hint)
