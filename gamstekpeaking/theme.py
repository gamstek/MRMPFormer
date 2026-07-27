"""
theme.py — GAMSTEKPEAKing 全局主题系统
========================================
集中管理配色方案、字体族、QSS 样式表。
所有页面和组件通过导入 Colors/Fonts/global_stylesheet() 保持视觉一致。

设计语言: 深色科技风（Deep Tech），灵感来自现代 IDE 与科学计算平台。
"""


class Colors:
    """命名颜色令牌。直接引用 Colors.xxx 而非硬编码色值。"""
    # 背景层级
    bg_primary    = "#0F172A"   # 主窗口背景（深邃蓝黑）
    bg_sidebar    = "#1E293B"   # 侧边栏背景
    bg_card       = "#1E293B"   # 卡片背景
    bg_card_hover = "#273449"   # 卡片/导航项悬停
    bg_input      = "#0F172A"   # 输入框背景

    # 文字层级
    text_primary   = "#E2E8F0"  # 主文字（浅灰白）
    text_secondary = "#94A3B8"  # 次要文字 / 占位符
    text_disabled  = "#64748B"  # 禁用态文字

    # 强调色
    accent       = "#38BDF8"    # 主强调色（科技蓝）
    accent_hover = "#7DD3FC"    # 强调色悬停

    # 语义色
    success = "#34D399"         # 成功 / 完成（翠绿）
    warning = "#FBBF24"         # 警告 / 进行中（琥珀）
    error   = "#F87171"         # 错误 / 失败（柔红）

    # 边框
    border       = "#334155"    # 卡片/输入框边框
    border_focus = "#38BDF8"    # 聚焦边框

    # 进度条
    progress_bg   = "#1E293B"   # 进度条背景
    progress_fill = "#38BDF8"   # 进度条填充


class Fonts:
    """字体族常量。优先使用系统自带字体，避免额外安装。"""
    primary = '"Microsoft YaHei", "Segoe UI", sans-serif'
    mono    = '"Cascadia Code", "Consolas", monospace'


def global_stylesheet() -> str:
    """返回应用于 QApplication 的全局 QSS 样式表。"""
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

    /* === 输入框 === */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {Colors.bg_input};
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 6px 10px;
        color: {Colors.text_primary};
        selection-background-color: {Colors.accent};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {Colors.border_focus};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        color: {Colors.text_disabled};
        background-color: {Colors.bg_card};
    }}

    /* === 主按钮（填充渐变） === */
    QPushButton[cssClass="primary"] {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.accent},
            stop:1 #818CF8
        );
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        color: {Colors.bg_primary};
        font-weight: bold;
    }}
    QPushButton[cssClass="primary"]:hover {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.accent_hover},
            stop:1 #A5B4FC
        );
    }}
    QPushButton[cssClass="primary"]:disabled {{
        background: {Colors.border};
        color: {Colors.text_disabled};
    }}

    /* === 次按钮（幽灵/描边） === */
    QPushButton[cssClass="secondary"] {{
        background: transparent;
        border: 1px solid #475569;
        border-radius: 6px;
        padding: 8px 20px;
        color: {Colors.text_primary};
    }}
    QPushButton[cssClass="secondary"]:hover {{
        border-color: {Colors.accent};
        color: {Colors.accent};
    }}

    /* === 文件选择按钮 === */
    QPushButton[cssClass="filePick"] {{
        background: {Colors.bg_input};
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 6px 12px;
        color: {Colors.text_secondary};
    }}
    QPushButton[cssClass="filePick"]:hover {{
        border-color: {Colors.accent};
        color: {Colors.accent};
    }}

    /* === 进度条 === */
    QProgressBar {{
        background-color: {Colors.progress_bg};
        border: none;
        border-radius: 4px;
        height: 4px;
        text-align: center;
        font-size: 10px;
        color: {Colors.text_secondary};
    }}
    QProgressBar::chunk {{
        background: qlineargradient(
            x1:0, y1:0, x2:1, y2:0,
            stop:0 {Colors.accent},
            stop:1 #818CF8
        );
        border-radius: 4px;
    }}

    /* === 滚动条 === */
    QScrollBar:vertical {{
        background: {Colors.bg_primary};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {Colors.border};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {Colors.accent};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* === 下拉框 === */
    QComboBox {{
        background-color: {Colors.bg_input};
        border: 1px solid {Colors.border};
        border-radius: 6px;
        padding: 6px 10px;
        color: {Colors.text_primary};
    }}
    QComboBox:hover {{
        border-color: {Colors.accent};
    }}
    QComboBox QAbstractItemView {{
        background-color: {Colors.bg_card};
        border: 1px solid {Colors.border};
        selection-background-color: {Colors.bg_card_hover};
        color: {Colors.text_primary};
    }}

    /* === 复选框 === */
    QCheckBox {{
        spacing: 8px;
        color: {Colors.text_secondary};
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {Colors.border};
        border-radius: 3px;
        background: {Colors.bg_input};
    }}
    QCheckBox::indicator:checked {{
        background: {Colors.accent};
        border-color: {Colors.accent};
    }}

    /* === 工具提示 === */
    QToolTip {{
        background-color: {Colors.bg_card};
        border: 1px solid {Colors.border};
        border-radius: 4px;
        padding: 4px 8px;
        color: {Colors.text_primary};
        font-size: 12px;
    }}
    """
