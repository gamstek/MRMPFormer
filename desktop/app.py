"""
app.py — GAMSTEKPEAKing 主窗口
===============================
构建应用主窗口：左侧边栏导航 + 右侧 QStackedWidget 页面容器 + 底部状态栏。
所有功能板块通过 add_page() 注册，侧边栏按钮自动生成。

依赖: PySide6, theme.py
"""

from pathlib import Path
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QFont, QAction, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QPushButton, QLabel, QStatusBar,
    QSizePolicy, QSpacerItem, QFrame, QApplication,
)
from theme import Colors, Fonts, global_stylesheet


class SidebarButton(QPushButton):
    """
    侧边栏导航按钮。
    支持三种状态：激活（蓝色指示条+高亮）、禁用（灰色文字+不可点击）、普通。
    """

    def __init__(self, label: str, index: int, parent=None):
        """
        Args:
            label: 导航项文字（如 "前处理"）
            index: 对应 QStackedWidget 的页面索引
            parent: 父级 widget
        """
        super().__init__(parent)
        self._index = index
        self._active = False
        self._enabled_flag = True

        # 纯文字，不加 emoji 前缀
        self.setText(f"    {label}")
        self.setFont(QFont(Fonts.primary, 13))
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_style()

    def _apply_style(self):
        """根据当前状态刷新 QSS 样式（深蓝侧边栏 + 白字）。"""
        if not self._enabled_flag:
            # 禁用态
            self.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding-left: 20px;
                    border: none;
                    border-left: 3px solid transparent;
                    background-color: transparent;
                    color: rgba(255,255,255,0.3);
                    font-size: 13px;
                }}
            """)
            self.setEnabled(False)
            self.setChecked(False)
        elif self._active:
            # 激活态：红色左边框 + 微亮背景
            self.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding-left: 17px;
                    border: none;
                    border-left: 3px solid {Colors.accent};
                    background-color: rgba(255,255,255,0.1);
                    color: #FFFFFF;
                    font-size: 13px;
                    font-weight: bold;
                }}
            """)
        else:
            # 普通态：白色半透明文字
            self.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding-left: 20px;
                    border: none;
                    border-left: 3px solid transparent;
                    background-color: transparent;
                    color: rgba(255,255,255,0.75);
                    font-size: 13px;
                }}
                QPushButton:hover {{
                    background-color: rgba(255,255,255,0.08);
                }}
            """)

    def set_active(self, active: bool):
        """设置激活态，刷新样式。"""
        self._active = active
        self.setChecked(active)
        self._apply_style()

    def set_enabled_state(self, enabled: bool):
        """设置启用/禁用状态。（避免与 QWidget.setEnabled 冲突，独立命名）"""
        self._enabled_flag = enabled
        self.setEnabled(enabled)
        self._apply_style()

    @property
    def page_index(self) -> int:
        """对应的 QStackedWidget 页面索引。"""
        return self._index


class GAMSTEKPEAKingWindow(QMainWindow):
    """
    GAMSTEKPEAKing 主窗口。

    布局结构:
    ┌──────────┬───────────────────────┐
    │  Logo    │                       │
    │  ──────  │  QStackedWidget       │
    │  导航1   │  (页面内容区)          │
    │  导航2   │                       │
    │  ...     │                       │
    │  ──────  │                       │
    │  关于    │                       │
    ├──────────┴───────────────────────┤
    │  状态栏                          │
    └──────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GAMSTEKPEAKing")
        self.resize(960, 680)
        # 不设 minimumSize，让窗口可自由缩放

        # 设置任务栏图标（从 assets/logo.png 加载）
        logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))

        # 存储侧边栏按钮引用，用于导航切换
        self._sidebar_buttons: list[SidebarButton] = []
        self._current_index = 0  # 当前激活的页面索引

        # ── 中心 Widget ──
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── 侧边栏 ──
        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        # ── 内容区 + 状态栏 ──
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 页面容器（QStackedWidget 实现无闪烁页面切换）
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background-color: {Colors.bg_primary};")
        right_layout.addWidget(self.stack, 1)

        # 底部状态栏
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {Colors.bg_sidebar};
                border-top: 1px solid {Colors.border};
                color: {Colors.text_secondary};
                font-size: 11px;
                padding: 2px 12px;
            }}
        """)
        self.setStatusBar(self.status_bar)
        # 状态栏初始为空，运行时由各页面动态更新

        root_layout.addWidget(right_panel, 1)

    def _build_sidebar(self) -> QWidget:
        """构建左侧导航栏。返回包含 Logo + 导航列表 + 关于按钮的 QWidget。"""
        sidebar = QWidget()
        sidebar.setFixedWidth(180)  # 固定宽度，不可拖拽调整
        sidebar.setStyleSheet(f"background-color: {Colors.bg_sidebar};")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo 区域（与侧边栏同为深蓝色背景）
        logo = QLabel("GAMSTEKPEAKing")
        logo.setFont(QFont(Fonts.primary, 13))
        logo.setStyleSheet(f"""
            color: #FFFFFF;
            padding: 16px 16px 12px 16px;
            font-weight: bold;
        """)
        layout.addWidget(logo)

        # 分隔线
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet(f"color: rgba(255,255,255,0.2); max-height: 1px; margin: 0 12px;")
        layout.addWidget(sep1)
        layout.addSpacing(8)

        # 导航按钮容器（未来按钮动态添加到此布局）
        self._nav_container = QVBoxLayout()
        self._nav_container.setContentsMargins(0, 0, 0, 0)
        self._nav_container.setSpacing(2)
        layout.addLayout(self._nav_container)
        layout.addStretch(1)  # 弹性空间把"关于"推到底部

        # 底部分隔线
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: rgba(255,255,255,0.2); max-height: 1px; margin: 0 12px;")
        layout.addWidget(sep2)
        layout.addSpacing(4)

        # "关于"区域（底部固定，显示版本号）
        about_btn = QLabel("  v0.1.0")
        about_btn.setFont(QFont(Fonts.primary, 11))
        about_btn.setStyleSheet(f"""
            color: rgba(255,255,255,0.5);
            padding: 8px 12px;
        """)
        about_btn.setFixedHeight(36)
        layout.addWidget(about_btn)

        return sidebar

    # ================================================================
    # 公共接口
    # ================================================================

    def add_page(self, name: str, widget: QWidget, enabled: bool = True):
        """
        注册一个功能页面。

        Args:
            name: 侧边栏显示名称（如 "前处理"）
            widget: 页面 QWidget 实例
            enabled: 是否启用（False=灰色禁用态）
        """
        index = self.stack.count()
        btn = SidebarButton(name, index)

        # 点击导航按钮 → 切换对应页面
        btn.clicked.connect(lambda checked, i=index: self.navigate_to(i))

        btn.set_enabled_state(enabled)
        self._nav_container.addWidget(btn)
        self._sidebar_buttons.append(btn)
        self.stack.addWidget(widget)

        # 首个注册的页面自动激活
        if index == 0:
            btn.set_active(True)

    def navigate_to(self, index: int):
        """
        切换到指定页面索引，同步更新侧边栏激活态。

        Args:
            index: 目标页面索引（0-based）
        """
        if index == self._current_index:
            return  # 已在目标页面，无需切换
        if 0 <= index < len(self._sidebar_buttons):
            # 旧按钮去激活
            if 0 <= self._current_index < len(self._sidebar_buttons):
                self._sidebar_buttons[self._current_index].set_active(False)
            # 新按钮激活
            self._sidebar_buttons[index].set_active(True)
            self.stack.setCurrentIndex(index)
            self._current_index = index

    def set_status(self, text: str, color: str = Colors.text_secondary):
        """更新底部状态栏。"""
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {Colors.bg_primary};
                border-top: 1px solid {Colors.border};
                color: {color};
                font-size: 11px;
                padding: 2px 12px;
            }}
        """)
        self.status_bar.showMessage(text)

    def closeEvent(self, event):
        """窗口关闭事件 —— 确保所有后台线程优雅退出。"""
        # QThread 生命周期由各页面自行管理；
        # 此处预留统一清理逻辑的钩子
        event.accept()
