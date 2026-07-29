"""
theme.py — GAMSTEKPEAKing 全局主题系统
========================================
集中管理配色方案、字体族、QSS 样式表。
所有页面和组件通过导入 Colors/Fonts/global_stylesheet() 保持视觉一致。

设计语言: 红白蓝（Red / White / Blue）亮色专业风。
"""

import os


def _ensure_assets():
    """确保 assets/ 中存在所有自动生成的图标资源。

    返回 (up_arrow_path, down_arrow_path, check_icon_path) 的绝对路径元组（正斜杠），
    可直接嵌入 QSS url()。
    - spin_up_arrow.png  ← combo_down_arrow.png 旋转 180°
    - check_icon.png    ← QPainter 绘制的 ✓ 勾号
    """
    from PySide6.QtGui import QPixmap, QTransform, QPainter, QPen, QColor
    from PySide6.QtCore import Qt, QPoint

    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    up_path = os.path.join(assets_dir, "spin_up_arrow.png")
    down_path = os.path.join(assets_dir, "combo_down_arrow.png")
    check_path = os.path.join(assets_dir, "check_icon.png")

    # 上箭头：下箭头旋转 180°
    if not os.path.exists(up_path) and os.path.exists(down_path):
        pm = QPixmap(down_path)
        rotated = pm.transformed(QTransform().rotate(180), Qt.SmoothTransformation)
        rotated.save(up_path, "PNG")

    # 勾号图标：QPainter 绘制 ✓
    if not os.path.exists(check_path):
        pm = QPixmap(14, 14)
        pm.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("#DC2626"), 2.0)  # accent 红
        painter.setPen(pen)
        # 勾号折线：左下(3,7) → 中偏右(6,10) → 右上(12,3)
        painter.drawPolyline([QPoint(3, 7), QPoint(6, 10), QPoint(12, 3)])
        painter.end()
        pm.save(check_path, "PNG")

    return (up_path.replace("\\", "/"), down_path.replace("\\", "/"), check_path.replace("\\", "/"))


class Colors:
    """命名颜色令牌。直接引用 Colors.xxx 而非硬编码色值。"""
    # 背景层级（亮色调）
    bg_primary    = "#F8F9FA"   # 主窗口背景（浅灰白）
    bg_sidebar    = "#1E3A5F"   # 侧边栏背景（海军蓝）
    bg_card       = "#FFFFFF"   # 卡片背景（纯白）
    bg_card_hover = "#EEF2F7"   # 卡片/导航项悬停
    bg_input      = "#FFFFFF"   # 输入框背景

    # 文字层级
    text_primary   = "#1F2937"  # 主文字（深灰黑）
    text_secondary = "#6B7280"  # 次要文字 / 占位符
    text_disabled  = "#9CA3AF"  # 禁用态文字

    # 强调色（红色系，匹配 logo）
    accent       = "#DC2626"    # 主强调色（logo 红）
    accent_hover = "#EF4444"    # 强调色悬停

    # 语义色
    success = "#10B981"         # 成功 / 完成（翠绿）
    warning = "#F59E0B"         # 警告 / 进行中（琥珀）
    error   = "#EF4444"         # 错误 / 失败（柔红）

    # 边框
    border       = "#D1D5DB"    # 卡片/输入框边框
    border_focus = "#DC2626"    # 聚焦边框

    # 进度条
    progress_bg   = "#E5E7EB"   # 进度条背景
    progress_fill = "#DC2626"   # 进度条填充

    # 侧边栏内文字（白色系，因侧边栏为深蓝色背景）
    sidebar_text        = "#CBD5E1"  # 侧边栏普通文字
    sidebar_text_active = "#FFFFFF"  # 侧边栏激活文字


class Fonts:
    """字体族常量。优先使用系统自带字体，避免额外安装。
    QFont 和 QSS 均可直接使用 Fonts.primary（不含 CSS 引号以兼容 QFont 构造函数）。"""
    primary = "Microsoft YaHei, Segoe UI, sans-serif"
    mono    = "Cascadia Code, Consolas, monospace"


def global_stylesheet() -> str:
    """返回应用于 QApplication 的全局 QSS 样式表。"""
    # 确保图标资源存在，获取绝对路径
    _arrow_up, _arrow_down, _check_icon = _ensure_assets()
    return f"""
    /* === 全局 === */
    QMainWindow {{
        background-color: {Colors.bg_primary};
    }}
    QWidget {{
        font-family: {Fonts.primary};
        font-size: 13px;
        color: {Colors.text_primary};
    }}

    /* === 按钮基础 === */
    QPushButton {{
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 6px 14px;
        background-color: #E5E7EB;
        color: {Colors.text_primary};
    }}
    QPushButton:hover {{
        border-color: {Colors.accent};
        background-color: #D1D5DB;
    }}
    QPushButton:pressed {{
        background-color: #9CA3AF;
    }}
    QPushButton:disabled {{
        background-color: #F3F4F6;
        color: {Colors.text_disabled};
        border-color: {Colors.border};
    }}

    /* === 输入框 / 数字框 === */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {Colors.bg_input};
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 5px 8px;
        color: {Colors.text_primary};
        selection-background-color: {Colors.accent};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {Colors.border_focus};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        color: {Colors.text_disabled};
        background-color: {Colors.bg_card_hover};
    }}

    /* SpinBox 上下按钮 */
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border: none;
        border-left: 1px solid {Colors.border};
        border-bottom: 1px solid {Colors.border};
        border-top-right-radius: 6px;
        background-color: transparent;
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
        background-color: {Colors.bg_card_hover};
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 20px;
        border: none;
        border-left: 1px solid {Colors.border};
        border-bottom-right-radius: 6px;
        background-color: transparent;
    }}
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {Colors.bg_card_hover};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url({_arrow_up});
        width: 10px;
        height: 10px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url({_arrow_down});
        width: 10px;
        height: 10px;
    }}

    /* === 主按钮（红色填充） === */
    QPushButton[cssClass="primary"] {{
        background-color: {Colors.accent};
        border: none;
        border-radius: 6px;
        padding: 7px 22px;
        color: #FFFFFF;
        font-weight: bold;
    }}
    QPushButton[cssClass="primary"]:hover {{
        background-color: {Colors.accent_hover};
    }}
    QPushButton[cssClass="primary"]:pressed {{
        background-color: #B91C1C;
    }}
    QPushButton[cssClass="primary"]:disabled {{
        background-color: #FCA5A5;
        color: #7F1D1D;
    }}

    /* === 次按钮（描边） === */
    QPushButton[cssClass="secondary"] {{
        background: transparent;
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 7px 18px;
        color: {Colors.text_secondary};
    }}
    QPushButton[cssClass="secondary"]:hover {{
        border-color: {Colors.accent};
        color: {Colors.accent};
        background: transparent;
    }}

    /* === 文件选择按钮 === */
    QPushButton[cssClass="filePick"] {{
        background-color: {Colors.bg_card_hover};
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 5px 14px;
        color: {Colors.text_primary};
        font-weight: 500;
    }}
    QPushButton[cssClass="filePick"]:hover {{
        border-color: {Colors.accent};
        background-color: {Colors.bg_card};
    }}

    /* === 进度条 === */
    QProgressBar {{
        background-color: {Colors.progress_bg};
        border: none;
        border-radius: 3px;
        height: 4px;
        text-align: center;
        font-size: 10px;
    }}
    QProgressBar::chunk {{
        background-color: {Colors.accent};
        border-radius: 3px;
    }}

    /* === 滚动条（极简细条） === */
    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {Colors.border};
        border-radius: 3px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {Colors.text_secondary};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 6px;
    }}
    QScrollBar::handle:horizontal {{
        background: {Colors.border};
        border-radius: 3px;
        min-width: 24px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {Colors.text_secondary};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* === 下拉框 === */
    QComboBox {{
        background-color: {Colors.bg_input};
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 5px 28px 5px 10px;
        color: {Colors.text_primary};
    }}
    QComboBox:hover {{
        border-color: {Colors.accent};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: center right;
        width: 24px;
        border: none;
        border-left: 1px solid {Colors.border};
        border-top-right-radius: 5px;
        border-bottom-right-radius: 5px;
        background-color: {Colors.bg_card_hover};
    }}
    QComboBox::drop-down:hover {{
        background-color: {Colors.border};
    }}
    QComboBox::down-arrow {{
        image: url({_arrow_down});
        width: 10px;
        height: 10px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {Colors.bg_card};
        border: 1px solid {Colors.border};
        border-radius: 4px;
        outline: none;
        padding: 4px;
    }}
    QComboBox QAbstractItemView::item {{
        padding: 5px 10px;
        border-radius: 3px;
        color: {Colors.text_primary};
        background-color: transparent;
    }}
    QComboBox QAbstractItemView::item:hover {{
        background-color: {Colors.bg_card_hover};
    }}
    QComboBox QAbstractItemView::item:selected {{
        background-color: {Colors.accent};
        color: #FFFFFF;
    }}

    /* === 复选框 === */
    QCheckBox {{
        spacing: 8px;
        color: {Colors.text_secondary};
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {Colors.border};
        border-radius: 4px;
        background: {Colors.bg_input};
    }}
    QCheckBox::indicator:hover {{
        border-color: {Colors.accent};
    }}
    QCheckBox::indicator:checked {{
        background: {Colors.bg_input};
        border-color: {Colors.accent};
        image: url({_check_icon});
    }}

    /* === 工具提示 === */
    QToolTip {{
        background-color: {Colors.bg_card};
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 6px 10px;
        color: {Colors.text_primary};
        font-size: 12px;
    }}

    /* === 下拉菜单 === */
    QMenu {{
        background-color: {Colors.bg_card};
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 24px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background-color: {Colors.bg_card_hover};
    }}

    /* === 列表控件 === */
    QListWidget, QTableWidget {{
        background-color: {Colors.bg_card};
        border: 1px solid {Colors.border};
        border-radius: 6px;
        outline: none;
    }}
    QListWidget::item, QTableWidget::item {{
        padding: 4px 8px;
        border-radius: 3px;
    }}
    QListWidget::item:hover, QTableWidget::item:hover {{
        background-color: {Colors.bg_card_hover};
    }}
    QListWidget::item:selected, QTableWidget::item:selected {{
        background-color: {Colors.accent};
        color: #FFFFFF;
    }}

    /* === 分组框 === */
    QGroupBox {{
        border: 1px solid {Colors.border};
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 16px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 8px;
        color: {Colors.text_secondary};
    }}

    /* === 卡片样式 === */
    QFrame#conversionCard, QFrame#ionZenithCard {{
        background-color: {Colors.bg_card};
        border: 1px solid {Colors.border};
        border-radius: 8px;
    }}

    /* === 标签页 === */
    QTabWidget::pane {{
        border: none;
        background: {Colors.bg_primary};
    }}
    QTabBar::tab {{
        padding: 8px 16px;
        border: none;
        border-bottom: 2px solid transparent;
        color: {Colors.text_secondary};
    }}
    QTabBar::tab:hover {{
        color: {Colors.text_primary};
    }}
    QTabBar::tab:selected {{
        color: {Colors.accent};
        border-bottom-color: {Colors.accent};
    }}
    """
