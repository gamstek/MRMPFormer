"""
settings.py — 设置板块（占位）
==============================
未来提供全局偏好设置：主题切换、语言选择、默认路径、GPU 选择等。
当前为占位页面，显示"敬请期待"。

依赖: PySide6, theme
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from theme import Colors, Fonts


class SettingsPage(QWidget):
    """设置板块占位页面。后续将加载各项偏好设置控件。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        icon = QLabel("⚙")
        icon.setFont(QFont(Fonts.primary, 48))
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet(f"color: {Colors.text_disabled}; border: none; background: none;")
        layout.addWidget(icon)

        text = QLabel("设置 — 敬请期待")
        text.setFont(QFont(Fonts.primary, 16))
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet(f"color: {Colors.text_disabled}; border: none; background: none;")
        layout.addWidget(text)

        hint = QLabel("将提供主题 / 语言 / 默认路径 / GPU 等全局偏好设置")
        hint.setFont(QFont(Fonts.primary, 12))
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {Colors.text_disabled}; border: none; background: none;")
        layout.addWidget(hint)
