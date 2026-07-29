# GAMSTEKPEAKing 前处理板块 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 GAMSTEKPEAKing 桌面应用骨架 + 前处理板块两个功能（格式转换 msdata→mzML、离子天顶 MS1→CSV）

**Architecture:** PySide6 纯代码布局，左侧边栏导航（180px）+ QStackedWidget 内容区。功能卡片式 UI，后台耗时操作封装为 QThread。`bin/` 内嵌 msdata2mzml.exe 实现格式转换自包含。

**Tech Stack:** Python 3.10/3.11, PySide6, pymzml, numpy, subprocess, pathlib

## Global Constraints

- Python环境使用conda环境`gamstekpeaking`，使用`conda activate gamstekpeaking`来激活，所有包都只能安装到该环境下
- 路径必须用 `os.path.join()` / `pathlib.Path`，禁止硬编码 `/` 或 `\\`
- 所有 `.py` 文件头/函数/关键行/Signal 必须详细注释
- 每个目录必须有 README（根 `README.md`，子 `README_<name>.md`）
- 遵守 QuanFormer 项目 charter：领域无关、科学工作台视觉、dev_log 更新
- 不修改 `model/GUI/`、`model/quanformer/`、`ms2mzml/`
- GAMSTEKPEAKing 完全自包含，无外部 `ms2mzml/` import

---

### Task 1: 项目目录骨架 + 依赖文件

**Files:**
- Create: `gamstekpeaking/requirements.txt`
- Create: `gamstekpeaking/pages/__init__.py`
- Create: `gamstekpeaking/workers/__init__.py`
- Create: `gamstekpeaking/engine/__init__.py`
- Create: `gamstekpeaking/assets/.gitkeep`

**Interfaces:**
- Produces: 完整目录树结构，激活`conda activate gamstekpeaking`环境后,`pip install -r requirements.txt` 可安装所有依赖

- [ ] **Step 1: 创建所有目录**

```powershell
New-Item -ItemType Directory -Force -Path gamstekpeaking/pages, gamstekpeaking/workers, gamstekpeaking/engine, gamstekpeaking/bin, gamstekpeaking/assets
```

- [ ] **Step 2: 创建 requirements.txt**

```txt
PySide6>=6.5.0,<7.0.0
pymzml>=0.9.0,<1.0.0
numpy>=1.24.0,<2.0.0
tqdm>=4.65.0
```

- [ ] **Step 3: 创建 __init__.py 文件**

```python
# pages/__init__.py
"""GAMSTEKPEAKing — 页面模块。每个板块一个文件，由 app.py 的侧边栏路由加载。"""
```

```python
# workers/__init__.py
"""GAMSTEKPEAKing — 后台工作线程模块。封装耗时操作为 QThread，不阻塞 UI。"""
```

```python
# engine/__init__.py
"""GAMSTEKPEAKing — 模型推理引擎（预留）。未来从 model/quanformer/ 逐步迁移模型加载与推理逻辑。"""
```

- [ ] **Step 4: 提交**

```bash
git add gamstekpeaking/
git commit -m "feat(gamstekpeaking): scaffold project skeleton and dependencies"
```

---

### Task 2: 整合 msdata2mzml 运行时

**Files:**
- Create: `gamstekpeaking/bin/`（复制自 `ms2mzml/bin/` 全部内容）
- Create: `gamstekpeaking/bin/README_bin.md`

**Interfaces:**
- Produces: `gamstekpeaking/bin/msdata2mzml.exe` 及所有依赖 DLL 和 `share/OpenMS/` 就绪

- [ ] **Step 1: 复制 bin/ 目录全部内容**

```powershell
Copy-Item -Recurse -Force ms2mzml/bin/* gamstekpeaking/bin/
```

- [ ] **Step 2: 验证关键文件存在**

```powershell
Test-Path gamstekpeaking/bin/msdata2mzml.exe
Test-Path gamstekpeaking/bin/share/OpenMS
```
Expected: 两个路径均返回 `True`

- [ ] **Step 3: 创建 README_bin.md**

````markdown
# bin/ — msdata2mzml 运行时

基于 OpenMS 工具链编译的 `.msdata → .mzML` 格式转换工具，由 `workers/converter.py` 通过 subprocess 调用。

## 文件清单

| 文件 | 作用 |
|------|------|
| `msdata2mzml.exe` | 格式转换主程序（C++/OpenMS） |
| `OpenMS.dll` | OpenMS 核心库 |
| `Qt5Core.dll`, `Qt5Network.dll` | Qt 运行时依赖 |
| `*.dll` | 其他运行时依赖（zlib, xerces, lapack 等） |
| `share/OpenMS/` | OpenMS 数据文件（CHEMISTRY, CV 术语等） |

## 使用方式

```
msdata2mzml.exe <input.msdata>
```
输出自动生成在输入文件的同级目录下的 `<basename>/` 子目录中。

## 注意事项

- 路径不能包含中文字符（OpenMS C++ 层限制）
- 需要设置 `OPENMS_DATA_PATH` 环境变量指向 `share/OpenMS/`
- 来源：`ms2mzml/bin/`（历史工具链，复制至此以实现自包含）
````

- [ ] **Step 4: 提交**

```bash
git add gamstekpeaking/bin/
git commit -m "feat(gamstekpeaking): integrate msdata2mzml runtime"
```

---

### Task 3: 主题系统 theme.py

**Files:**
- Create: `gamstekpeaking/theme.py`

**Interfaces:**
- Produces:
  - `Colors` — 命名颜色常量类
  - `Fonts` — 字体族常量类
  - `global_stylesheet() -> str` — 返回全局 QSS 字符串

- [ ] **Step 1: 创建 theme.py**

```python
"""
theme.py — GAMSTEKPEAKing 全局主题系统
========================================
集中管理配色方案、字体族、QSS 样式表。
所有页面和组件通过导入 Colors/Fonts/global_stylesheet() 保持视觉一致。

设计语言: 深色科技风（Deep Tech），灵感来自现代 IDE 与科学计算平台。
"""

# ============================================================
# 配色方案
# ============================================================

class Colors:
    """命名颜色令牌。直接引用 Colors.xxx 而非硬编码色值。"""
    # 背景
    bg_primary    = "#0F172A"   # 主窗口背景（深邃蓝黑）
    bg_sidebar    = "#1E293B"   # 侧边栏背景
    bg_card       = "#1E293B"   # 卡片背景
    bg_card_hover = "#273449"   # 卡片/导航项悬停
    bg_input      = "#0F172A"   # 输入框背景

    # 文字
    text_primary   = "#E2E8F0"  # 主文字（浅灰白）
    text_secondary = "#94A3B8"  # 次要文字 / 占位符
    text_disabled  = "#64748B"  # 禁用态文字

    # 强调
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


# ============================================================
# 字体族
# ============================================================

class Fonts:
    """字体族常量。优先使用系统自带字体，避免额外安装。"""
    primary = '"Microsoft YaHei", "Segoe UI", sans-serif'
    mono    = '"Cascadia Code", "Consolas", monospace'


# ============================================================
# 全局样式表
# ============================================================

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

    /* === 标签页 === */
    QTabWidget::pane {{
        border: none;
        background: {Colors.bg_primary};
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
```

- [ ] **Step 2: 验证语法**

```powershell
python -c "import sys; sys.path.insert(0, 'gamstekpeaking'); from theme import Colors, Fonts, global_stylesheet; print('OK:', Colors.accent)"
```
Expected: `OK: #38BDF8`

- [ ] **Step 3: 提交**

```bash
git add gamstekpeaking/theme.py
git commit -m "feat(gamstekpeaking): add theme system with Colors, Fonts, and global QSS"
```

---

### Task 4: 主窗口骨架 app.py

**Files:**
- Create: `gamstekpeaking/app.py`

**Interfaces:**
- Produces:
  - `class SidebarButton(QPushButton)` — 侧边栏导航按钮（激活态/禁用态/指示条）
  - `class GAMSTEKPEAKingWindow(QMainWindow)` — 主窗口
    - `__init__()` → 构建侧边栏 + QStackedWidget + 状态栏
    - `add_page(name, icon, widget, enabled=True)` → 注册页面
    - `navigate_to(index)` → 切换页面 + 更新侧边栏激活态
    - `set_status(text, color=...)` → 更新状态栏

- [ ] **Step 1: 创建 app.py**

```python
"""
app.py — GAMSTEKPEAKing 主窗口
===============================
构建应用主窗口：左侧边栏导航 + 右侧 QStackedWidget 页面容器 + 底部状态栏。
所有功能板块通过 add_page() 注册，侧边栏按钮自动生成。

依赖: PySide6, theme.py
"""

from pathlib import Path
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QFont, QAction
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

    def __init__(self, icon_text: str, label: str, index: int, parent=None):
        """
        Args:
            icon_text: emoji 或 Unicode 图标字符
            label: 中文标签文字
            index: 对应 QStackedWidget 的页面索引
            parent: 父级 widget
        """
        super().__init__(parent)
        self._index = index
        self._active = False
        self._enabled_flag = True

        # 按钮文字: "🏔  前处理"
        self.setText(f"  {icon_text}  {label}")
        self.setFont(QFont(Fonts.primary, 13))
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_style()

    def _apply_style(self):
        """根据当前状态刷新 QSS 样式。"""
        if not self._enabled_flag:
            self.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding-left: 20px;
                    border: none;
                    border-left: 3px solid transparent;
                    background-color: transparent;
                    color: {Colors.text_disabled};
                    font-size: 13px;
                }}
            """)
            self.setEnabled(False)
            self.setChecked(False)
        elif self._active:
            self.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding-left: 17px;
                    border: none;
                    border-left: 3px solid {Colors.accent};
                    background-color: {Colors.bg_card_hover};
                    color: {Colors.accent};
                    font-size: 13px;
                    font-weight: bold;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    padding-left: 20px;
                    border: none;
                    border-left: 3px solid transparent;
                    background-color: transparent;
                    color: {Colors.text_primary};
                    font-size: 13px;
                }}
                QPushButton:hover {{
                    background-color: {Colors.bg_card_hover};
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

    布局:
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
        self.setMinimumSize(900, 620)

        # 存储侧边栏按钮引用
        self._sidebar_buttons: list[SidebarButton] = []
        self._current_index = 0

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

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background-color: {Colors.bg_primary};")
        right_layout.addWidget(self.stack, 1)

        # 状态栏
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
        self.status_bar.showMessage("🟢 就绪")

        root_layout.addWidget(right_panel, 1)

    def _build_sidebar(self) -> QWidget:
        """构建左侧导航栏。返回包含 Logo + 导航列表 + 关于按钮的 QWidget。"""
        sidebar = QWidget()
        sidebar.setFixedWidth(180)
        sidebar.setStyleSheet(f"background-color: {Colors.bg_sidebar};")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo 区域
        logo = QLabel("🏔 GAMSTEKPEAKing")
        logo.setFont(QFont(Fonts.primary, 11))
        logo.setStyleSheet(f"""
            color: {Colors.accent};
            padding: 16px 16px 12px 16px;
            font-weight: bold;
        """)
        layout.addWidget(logo)

        # 分隔线
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet(f"color: {Colors.border}; max-height: 1px; margin: 0 12px;")
        layout.addWidget(sep1)
        layout.addSpacing(8)

        # 导航按钮容器（未来按钮动态添加到此）
        self._nav_container = QVBoxLayout()
        self._nav_container.setContentsMargins(0, 0, 0, 0)
        self._nav_container.setSpacing(2)
        layout.addLayout(self._nav_container)
        layout.addStretch(1)  # 把"关于"推到底部

        # 底部分隔线 + 关于
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {Colors.border}; max-height: 1px; margin: 0 12px;")
        layout.addWidget(sep2)
        layout.addSpacing(4)

        about_btn = QLabel("  ⓘ  关于")
        about_btn.setFont(QFont(Fonts.primary, 12))
        about_btn.setStyleSheet(f"""
            color: {Colors.text_disabled};
            padding: 8px 12px;
        """)
        about_btn.setFixedHeight(36)
        layout.addWidget(about_btn)

        return sidebar

    # === 公共接口 ===

    def add_page(self, icon: str, name: str, widget: QWidget, enabled: bool = True):
        """
        注册一个功能页面。

        Args:
            icon: emoji 图标（如 "🏔"）
            name: 侧边栏显示名称（如 "前处理"）
            widget: 页面 QWidget 实例
            enabled: 是否启用（False=灰色禁用）
        """
        index = self.stack.count()
        btn = SidebarButton(icon, name, index)

        # 点击导航按钮 → 切换页面
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
        切换到指定页面索引，更新侧边栏激活态。

        Args:
            index: 目标页面索引（0-based）
        """
        if index == self._current_index:
            return
        if 0 <= index < len(self._sidebar_buttons):
            # 旧按钮去激活
            if 0 <= self._current_index < len(self._sidebar_buttons):
                self._sidebar_buttons[self._current_index].set_active(False)
            # 新按钮激活
            self._sidebar_buttons[index].set_active(True)
            self.stack.setCurrentIndex(index)
            self._current_index = index

    def set_status(self, text: str, color: str = Colors.text_secondary):
        """
        更新底部状态栏文字和颜色。

        Args:
            text: 状态文字（如 "✅ 转换完成"）
            color: CSS 颜色字符串
        """
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {Colors.bg_sidebar};
                border-top: 1px solid {Colors.border};
                color: {color};
                font-size: 11px;
                padding: 2px 12px;
            }}
        """)
        self.status_bar.showMessage(text)

    def closeEvent(self, event):
        """窗口关闭时确保所有后台线程优雅退出。"""
        # 遍历所有子对象，终止活跃的 QThread
        for child in self.findChildren(QWidget):
            pass  # QThread 管理由各页面自行处理
        event.accept()
```

- [ ] **Step 2: 验证导入**

```powershell
python -c "import sys; sys.path.insert(0, 'gamstekpeaking'); from app import GAMSTEKPEAKingWindow; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add gamstekpeaking/app.py
git commit -m "feat(gamstekpeaking): add main window shell with sidebar navigation"
```

---

### Task 5: 格式转换后台线程 workers/converter.py

**Files:**
- Create: `gamstekpeaking/workers/converter.py`

**Interfaces:**
- Consumes: `gamstekpeaking/bin/msdata2mzml.exe`（硬依赖，运行时检查）
- Produces:
  - `class MsdataConverter(QThread)`
    - `progress = Signal(int, int)` — (当前索引, 总数)
    - `file_done = Signal(int, bool, str)` — (索引, 成功标志, 信息如文件大小或错误消息)
    - `error = Signal(str)` — 全局错误
    - `__init__(self, files: list[str], output_dir: str | None = None)`
    - `run(self)` — 逐文件调用 msdata2mzml.exe

- [ ] **Step 1: 创建 workers/converter.py**

```python
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


# 定位 bin/ 目录（相对于本文件向上两级）
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
            # 发出进度
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

        # 最终进度 (total, total)
        self.progress.emit(total, total)
```

- [ ] **Step 2: 验证语法**

```powershell
python -c "import sys; sys.path.insert(0, 'gamstekpeaking'); from workers.converter import MsdataConverter; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add gamstekpeaking/workers/converter.py
git commit -m "feat(gamstekpeaking): add MsdataConverter background thread"
```

---

### Task 6: 离子天顶后台线程 workers/ion_zenith.py

**Files:**
- Create: `gamstekpeaking/workers/ion_zenith.py`

**Interfaces:**
- Produces:
  - `class IonZenithWorker(QThread)`
    - `progress = Signal(int, int)` — (已扫描谱图数, 总数或0)
    - `stats = Signal(int, int)` — (MS1 谱图计数, 扫描峰总数)
    - `finished = Signal(int, float, str)` — (离子数, 耗时秒, 输出路径)
    - `error = Signal(str)`
    - `__init__(self, params: dict, parent=None)` — params 包含 input_mzml, output_csv, mz_min, mz_max 等
    - Parameters dict keys: `input_mzml`, `output_csv`, `mz_min`, `mz_max`, `ppm_tol`, `da_tol`, `intensity_min`, `intensity_max`, `max_spectra`, `build_index`

- [ ] **Step 1: 创建 workers/ion_zenith.py**

```python
"""
ion_zenith.py — 离子天顶算法后台线程
=====================================
封装 IonZenithWorker(QThread)，遍历 mzML 的 MS1 谱图，
按 m/z 容差分箱聚合，每个离子保留最高强度的观测，输出精简 CSV。

算法核心:
  遍历 MS1 → m/z 在容差内合并 → 保留最高强度 → 按 m/z 排序写入 CSV

依赖: pymzml, numpy, csv, time, pathlib, PySide6.QtCore
"""

import csv
import math
import os
import time
from pathlib import Path
import numpy as np
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
    # (ion_count: int, elapsed_sec: float, output_path: str)
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
        self._cancelled = False  # 预留取消标志（后续版本实现）

    def cancel(self):
        """请求取消当前运行。"""
        self._cancelled = True

    def run(self):
        """在线程中执行离子天顶算法。"""
        import pymzml

        p = self._params
        t0 = time.perf_counter()

        # 验证必需参数
        input_path = Path(p.get("input_mzml", ""))
        output_path = Path(p.get("output_csv", ""))
        if not input_path.exists():
            self.error.emit(f"输入文件不存在: {input_path}")
            return
        if not output_path.parent.exists():
            self.error.emit(f"输出目录不存在: {output_path.parent}")
            return

        # 读取参数（带默认值）
        mz_min = float(p.get("mz_min", 50.0))
        mz_max = float(p.get("mz_max", 2000.0))
        ppm_tol = float(p.get("ppm_tol", 10.0))
        da_tol = float(p.get("da_tol", 0.01))
        intensity_min = p.get("intensity_min")  # None or float
        intensity_max = p.get("intensity_max")  # None or float
        max_spectra = int(p.get("max_spectra", 0))
        build_index = bool(p.get("build_index", False))

        if intensity_min is not None:
            intensity_min = float(intensity_min)
        if intensity_max is not None:
            intensity_max = float(intensity_max)

        limit = max_spectra if max_spectra > 0 else None
        best = {}       # mz_key → [mz_center, rt_min, rt_sec, max_intensity, count]
        n_spec = 0      # 总谱图计数
        n_ms1 = 0       # MS1 谱图计数
        n_peaks = 0     # 扫描峰总数

        try:
            run = pymzml.run.Reader(str(input_path), build_index_from_scratch=build_index)
        except Exception as e:
            self.error.emit(f"无法打开 mzML 文件: {e}")
            return

        for spectrum in run:
            if self._cancelled:
                break
            n_spec += 1

            # 只处理 MS1
            ms_level = getattr(spectrum, "ms_level", None)
            if ms_level != 1:
                if limit and n_spec >= limit:
                    break
                continue

            # 获取保留时间
            rt_min, rt_sec = self._parse_rt(spectrum)
            if rt_sec is None:
                if limit and n_spec >= limit:
                    break
                continue

            # 获取峰数组
            arr = self._get_peaks(spectrum)
            if arr is None or arr.size == 0:
                if limit and n_spec >= limit:
                    break
                continue

            n_ms1 += 1
            mz_vals = arr[:, 0]
            int_vals = arr[:, 1]

            # 强度 + m/z 过滤
            mask = np.ones(len(mz_vals), dtype=bool)
            mask &= (mz_vals >= mz_min) & (mz_vals <= mz_max)
            mask &= (int_vals > 0)
            if intensity_min is not None:
                mask &= (int_vals >= intensity_min)
            if intensity_max is not None:
                mask &= (int_vals <= intensity_max)

            idx = np.where(mask)[0]
            if len(idx) == 0:
                if limit and n_spec >= limit:
                    break
                continue

            mz_filt = mz_vals[idx]
            int_filt = int_vals[idx]

            # 按 m/z 容差聚合：同一 key 只保留最高强度
            for j in range(len(mz_filt)):
                mz = float(mz_filt[j])
                intensity = float(int_filt[j])
                n_peaks += 1

                tol = max(mz * ppm_tol * 1e-6, da_tol)
                key = round(mz / tol) * tol

                if key not in best or intensity > best[key][3]:
                    best[key] = [mz, rt_min, rt_sec, intensity, 1]
                else:
                    best[key][3] = max(best[key][3], intensity)  # 确保强度最大
                    best[key][4] += 1

            # 定期发送进度（每 10 张谱图）
            if n_ms1 % 10 == 0:
                self.progress.emit(n_spec, limit or 0)
                self.stats.emit(n_ms1, n_peaks)

            if limit and n_spec >= limit:
                break

        # 写入 CSV
        os.makedirs(output_path.parent, exist_ok=True)
        rows = sorted(best.values(), key=lambda x: x[0])

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["mz", "rt_min", "rt_sec", "max_intensity", "n_observations"])
            for row in rows:
                w.writerow(row)

        elapsed = time.perf_counter() - t0
        self.progress.emit(n_spec, n_spec)
        self.stats.emit(n_ms1, n_peaks)
        self.finished.emit(len(rows), elapsed, str(output_path))

    # === 辅助方法 ===

    @staticmethod
    def _parse_rt(spectrum) -> tuple:
        """
        从 pymzML spectrum 对象提取保留时间。

        Returns:
            (rt_min: float | None, rt_sec: float | None)
        """
        st = getattr(spectrum, "scan_time", None)
        if st is None:
            return None, None
        if isinstance(st, (tuple, list)) and len(st) >= 2:
            val, unit = float(st[0]), str(st[1]).lower()
        else:
            val, unit = float(st), ""
        if math.isnan(val) or math.isinf(val):
            return None, None
        if "min" in unit:
            return val, val * 60.0
        return val / 60.0, val

    @staticmethod
    def _get_peaks(spectrum):
        """
        从 pymzML spectrum 提取 (m/z, intensity) 二维 numpy 数组。

        Returns:
            np.ndarray 或 None
        """
        p = getattr(spectrum, "peaks", None)
        if p is not None and len(p) > 0:
            return np.asarray(p, dtype=np.float64)
        mz_arr = getattr(spectrum, "mz", None)
        int_arr = getattr(spectrum, "i", None)
        if mz_arr is not None and int_arr is not None:
            return np.column_stack((
                np.asarray(mz_arr, dtype=np.float64),
                np.asarray(int_arr, dtype=np.float64),
            ))
        return None
```

- [ ] **Step 2: 验证语法**

```powershell
python -c "import sys; sys.path.insert(0, 'gamstekpeaking'); from workers.ion_zenith import IonZenithWorker; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add gamstekpeaking/workers/ion_zenith.py
git commit -m "feat(gamstekpeaking): add IonZenithWorker background thread"
```

---

### Task 7: 前处理页面 — 格式转换卡片

**Files:**
- Create: `gamstekpeaking/pages/preprocessing.py`（第一部分：格式转换卡片 + 页面框架）

**Interfaces:**
- Consumes: `MsdataConverter` from `workers.converter`, `Colors` from `theme`
- Produces:
  - `class PreprocessingPage(QWidget)` — 前处理板块主页面（包含两个功能卡片）
    - 内部管理 MsdataConverter 和 IonZenithWorker 生命周期

- [ ] **Step 1: 创建 preprocessing.py（格式转换卡片部分）**

```python
"""
preprocessing.py — 前处理板块
==============================
GAMSTEKPEAKing 首批上线板块，包含两个功能卡片：
  1. 格式转换 — msdata → mzML（拖拽 + 批量转换）
  2. 离子天顶 — 遍历 MS1 → 聚合 → CSV

每个卡片封装为独立的 QFrame 子类，通过 Signal 与后台线程通信。

依赖: PySide6, workers.converter, workers.ion_zenith, theme
"""

import os
import subprocess
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QComboBox, QSizePolicy, QScrollArea,
)
from theme import Colors, Fonts


class _DropZone(QFrame):
    """
    拖拽区域组件。支持点击选择文件和拖拽 .msdata 文件。
    拖入时边框变蓝 + 背景微亮。
    """

    files_selected = Signal(list)  # (file_paths: list[str])

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setStyleSheet(f"""
            _DropZone {{
                border: 2px dashed #475569;
                border-radius: 8px;
                background-color: transparent;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        icon = QLabel("📂")
        icon.setFont(QFont(Fonts.primary, 24))
        icon.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon)

        hint = QLabel("拖拽 .msdata 文件到此处\n或 点击选择文件")
        hint.setFont(QFont(Fonts.primary, 13))
        hint.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none;")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

    def mousePressEvent(self, event):
        """点击时弹出文件选择对话框。"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 msdata 文件", "", "MSData Files (*.msdata)"
        )
        if files:
            self.files_selected.emit(files)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖入时高亮边框。"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().endswith(".msdata") for url in urls):
                event.acceptProposedAction()
                self.setStyleSheet(f"""
                    _DropZone {{
                        border: 2px dashed {Colors.accent};
                        border-radius: 8px;
                        background-color: rgba(56, 189, 248, 0.05);
                    }}
                """)

    def dragLeaveEvent(self, event):
        """拖出时恢复默认样式。"""
        self.setStyleSheet(f"""
            _DropZone {{
                border: 2px dashed #475569;
                border-radius: 8px;
                background-color: transparent;
            }}
        """)

    def dropEvent(self, event: QDropEvent):
        """释放时收集 .msdata 文件路径。"""
        self.setStyleSheet(f"""
            _DropZone {{
                border: 2px dashed #475569;
                border-radius: 8px;
                background-color: transparent;
            }}
        """)
        urls = event.mimeData().urls()
        files = [url.toLocalFile() for url in urls if url.toLocalFile().endswith(".msdata")]
        if files:
            self.files_selected.emit(files)


class _FileListItem(QWidget):
    """
    文件列表项：文件名 + 状态图标 + 信息 + 移除按钮。
    状态图标：⏳ 等待 / 🔄 转换中 / ✅ 成功 / ❌ 失败
    """

    removed = Signal(str)  # (file_path: str) — 请求从列表移除

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # 状态图标
        self.status_icon = QLabel("⏳")
        self.status_icon.setFixedWidth(24)
        self.status_icon.setFont(QFont(Fonts.primary, 13))
        layout.addWidget(self.status_icon)

        # 文件名
        fname = os.path.basename(file_path)
        name_label = QLabel(fname)
        name_label.setFont(QFont(Fonts.primary, 12))
        name_label.setStyleSheet(f"color: {Colors.text_primary}; border: none; background: none;")
        layout.addWidget(name_label, 1)

        # 信息（文件大小/错误）
        self.info_label = QLabel("")
        self.info_label.setFont(QFont(Fonts.primary, 10))
        self.info_label.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none;")
        layout.addWidget(self.info_label)

        # 移除按钮
        remove_btn = QPushButton("×")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setFont(QFont(Fonts.primary, 14))
        remove_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {Colors.text_secondary};
                font-size: 16px;
            }}
            QPushButton:hover {{
                color: {Colors.error};
            }}
        """)
        remove_btn.clicked.connect(lambda: self.removed.emit(self.file_path))
        layout.addWidget(remove_btn)

        # 悬停背景
        self.setStyleSheet(f"""
            _FileListItem {{
                background-color: transparent;
                border-radius: 4px;
            }}
            _FileListItem:hover {{
                background-color: {Colors.bg_card_hover};
            }}
        """)

    def set_status(self, status: str, info: str = ""):
        """
        更新状态图标和信息文字。

        Args:
            status: "waiting" | "running" | "success" | "failed"
            info: 附加信息（如 "3 .mzML, 12 MB" 或错误消息）
        """
        icons = {
            "waiting": ("⏳", Colors.text_secondary),
            "running": ("🔄", Colors.warning),
            "success": ("✅", Colors.success),
            "failed":  ("❌", Colors.error),
        }
        icon_text, color = icons.get(status, ("⏳", Colors.text_secondary))
        self.status_icon.setText(icon_text)
        self.status_icon.setStyleSheet(f"color: {color}; border: none; background: none;")

        self.info_label.setText(info)
        if status == "failed":
            self.info_label.setStyleSheet(f"color: {Colors.error}; border: none; background: none; font-size: 10px;")
            self.setToolTip(info)  # 悬停显示完整错误
        elif status == "success":
            self.info_label.setStyleSheet(f"color: {Colors.success}; border: none; background: none; font-size: 10px;")
        else:
            self.info_label.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none; font-size: 10px;")


class ConversionCard(QFrame):
    """
    格式转换功能卡片。

    UI 组成:
      - 拖拽区域（点击/拖拽添加 msdata 文件）
      - 文件列表（状态图标 + 文件名 + 信息 + 移除按钮）
      - 输出目录选择
      - 进度条 + 运行按钮
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files: dict[str, _FileListItem] = {}  # file_path → list item widget
        self._converter = None
        self._is_running = False

        self._build_ui()

    def _build_ui(self):
        """构建卡片 UI 布局。"""
        self.setStyleSheet(f"""
            ConversionCard {{
                background-color: {Colors.bg_card};
                border: 1px solid {Colors.border};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # ── 卡片标题 ──
        title_layout = QHBoxLayout()
        title = QLabel("📦 格式转换")
        title.setFont(QFont(Fonts.primary, 14))
        title.setStyleSheet(f"color: {Colors.text_primary}; font-weight: bold; border: none; background: none;")
        title_layout.addWidget(title)
        subtitle = QLabel("msdata → mzML")
        subtitle.setFont(QFont(Fonts.primary, 12))
        subtitle.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none;")
        title_layout.addWidget(subtitle)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # ── 分隔线 ──
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {Colors.border}; max-height: 1px;")
        layout.addWidget(sep)

        # ── 拖拽区域 ──
        self.drop_zone = _DropZone()
        self.drop_zone.files_selected.connect(self._on_files_added)
        layout.addWidget(self.drop_zone)

        # ── 输出目录选择 ──
        out_layout = QHBoxLayout()
        out_label = QLabel("输出:")
        out_label.setFont(QFont(Fonts.primary, 12))
        out_label.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none;")
        out_layout.addWidget(out_label)

        self.output_combo = QComboBox()
        self.output_combo.addItem("默认（同输入目录）", "default")
        self.output_combo.addItem("自定义...", "custom")
        self.output_combo.currentIndexChanged.connect(self._on_output_changed)
        out_layout.addWidget(self.output_combo, 1)
        out_layout.addStretch(2)
        layout.addLayout(out_layout)

        # ── 文件列表 ──
        self.file_list = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list)
        self.file_list_layout.setContentsMargins(0, 0, 0, 0)
        self.file_list_layout.setSpacing(2)
        self.file_list_layout.addStretch()  # 空列表占位

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.file_list)
        scroll.setMaximumHeight(160)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Colors.border};
                border-radius: 6px;
                background-color: {Colors.bg_input};
            }}
        """)
        layout.addWidget(scroll)

        # ── 底部：进度条 + 运行按钮 ──
        bottom_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        bottom_layout.addWidget(self.progress_bar, 1)

        self.run_btn = QPushButton("▶ 开始转换")
        self.run_btn.setProperty("cssClass", "primary")
        self.run_btn.setFixedWidth(120)
        self.run_btn.clicked.connect(self._on_run)
        bottom_layout.addWidget(self.run_btn)

        layout.addLayout(bottom_layout)

    # === 事件处理 ===

    def _on_files_added(self, files: list[str]):
        """拖拽或选择文件后，追加到列表。"""
        for f in files:
            f = os.path.normpath(f)
            if f in self._files:
                continue  # 去重
            item = _FileListItem(f)
            item.removed.connect(self._on_file_removed)
            # 插入到 stretch 之前
            self.file_list_layout.insertWidget(self.file_list_layout.count() - 1, item)
            self._files[f] = item

    def _on_file_removed(self, file_path: str):
        """从列表中移除文件。"""
        if file_path in self._files:
            item = self._files.pop(file_path)
            self.file_list_layout.removeWidget(item)
            item.deleteLater()

    def _on_output_changed(self, index: int):
        """输出目录切换。'自定义'时弹出目录选择器。"""
        if self.output_combo.currentData() == "custom":
            dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if dir_path:
                # 插入自定义路径到下拉框
                self.output_combo.blockSignals(True)
                self.output_combo.insertItem(0, os.path.basename(dir_path), dir_path)
                self.output_combo.setCurrentIndex(0)
                self.output_combo.blockSignals(False)
            else:
                # 用户取消，切回默认
                self.output_combo.blockSignals(True)
                self.output_combo.setCurrentIndex(0)
                self.output_combo.blockSignals(False)

    def _on_run(self):
        """开始批量转换。"""
        if self._is_running or not self._files:
            return

        # 检查 exe 存在性
        bin_dir = Path(__file__).resolve().parent.parent / "bin"
        exe_path = bin_dir / "msdata2mzml.exe"
        if not exe_path.exists():
            self._show_error(f"未找到转换工具:\n{exe_path}\n请检查 bin/ 目录")
            return

        self._is_running = True
        self.run_btn.setEnabled(False)
        self.drop_zone.setEnabled(False)
        self.progress_bar.setValue(0)

        # 重置所有文件状态为等待
        for item in self._files.values():
            item.set_status("waiting")

        # 获取输出目录
        output_dir = None
        if self.output_combo.currentData() not in ("default", "custom"):
            output_dir = self.output_combo.currentData()

        # 启动后台线程
        file_paths = list(self._files.keys())
        self._converter = MsdataConverter(file_paths, output_dir)
        self._converter.progress.connect(self._on_progress)
        self._converter.file_done.connect(self._on_file_done)
        self._converter.error.connect(self._on_converter_error)
        self._converter.start()

    def _on_progress(self, current: int, total: int):
        """更新进度条。"""
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))

    def _on_file_done(self, index: int, success: bool, info: str):
        """单文件转换完成，更新列表项状态。"""
        file_path = list(self._files.keys())[index]
        item = self._files[file_path]
        if success:
            item.set_status("success", info)
        else:
            item.set_status("failed", info)

        # 检查是否全部完成
        if index == len(self._files) - 1:
            self._on_all_done()

    def _on_converter_error(self, message: str):
        """后台线程致命错误。"""
        self._show_error(message)
        self._reset_ui()

    def _on_all_done(self):
        """所有文件转换完毕，恢复 UI。"""
        self._reset_ui()

    def _reset_ui(self):
        """恢复 UI 到可操作状态。"""
        self._is_running = False
        self.run_btn.setEnabled(True)
        self.drop_zone.setEnabled(True)
        self.progress_bar.setValue(0)

    def _show_error(self, message: str):
        """显示错误提示（状态栏由父级 app.py 处理，此处用 tooltip 兜底）。"""
        # TODO: 后续接入统一的 Toast/通知组件
        print(f"[ConversionCard Error] {message}")


class PreprocessingPage(QWidget):
    """
    前处理板块主页面。
    包含格式转换卡片和离子天顶卡片，纵向排列在可滚动区域中。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # 格式转换卡片
        self.conversion_card = ConversionCard()
        layout.addWidget(self.conversion_card)

        # 离子天顶卡片（预留，Task 8 补充）
        self.ion_zenith_card = None  # type: ignore

        layout.addStretch()
```

- [ ] **Step 2: 验证语法**

```powershell
python -c "import sys; sys.path.insert(0, 'gamstekpeaking'); from pages.preprocessing import PreprocessingPage; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
git add gamstekpeaking/pages/preprocessing.py
git commit -m "feat(gamstekpeaking): add preprocessing page with conversion card"
```

---

### Task 8: 前处理页面 — 离子天顶卡片

**Files:**
- Modify: `gamstekpeaking/pages/preprocessing.py`（追加 IonZenithCard 类 + 集成到 PreprocessingPage）

**Interfaces:**
- Consumes: `IonZenithWorker` from `workers.ion_zenith`, `Colors, Fonts` from `theme`
- Produces: `class IonZenithCard(QFrame)` — 集成到 PreprocessingPage

- [ ] **Step 1: 在 preprocessing.py 中追加 IonZenithCard 类**

在现有文件末尾（`PreprocessingPage` 类之后）追加以下代码：

```python
"""追加: preprocessing.py 补充内容 — IonZenithCard"""

from PySide6.QtCore import QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (
    QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, QApplication,
)
from workers.ion_zenith import IonZenithWorker


class IonZenithCard(QFrame):
    """
    离子天顶功能卡片。

    UI 组成:
      - 输入/输出文件选择行
      - 可折叠高级参数面板（QPropertyAnimation 动画）
      - 参数校验（实时禁用运行按钮）
      - 进度条 + 统计信息 + 运行按钮
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._is_running = False
        self._advanced_expanded = False

        self._build_ui()
        self._connect_validation()

    def _build_ui(self):
        """构建卡片 UI 布局。"""
        self.setStyleSheet(f"""
            IonZenithCard {{
                background-color: {Colors.bg_card};
                border: 1px solid {Colors.border};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # ── 标题 ──
        title_layout = QHBoxLayout()
        title = QLabel("⚡ 离子天顶")
        title.setFont(QFont(Fonts.primary, 14))
        title.setStyleSheet(f"color: {Colors.text_primary}; font-weight: bold; border: none; background: none;")
        title_layout.addWidget(title)
        subtitle = QLabel("遍历 MS1 → 聚合 → CSV")
        subtitle.setFont(QFont(Fonts.primary, 12))
        subtitle.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none;")
        title_layout.addWidget(subtitle)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # ── 分隔线 ──
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {Colors.border}; max-height: 1px;")
        layout.addWidget(sep)

        # ── 输入文件行 ──
        input_layout = QHBoxLayout()
        input_label = QLabel("输入 mzML")
        input_label.setFixedWidth(80)
        input_label.setFont(QFont(Fonts.primary, 12))
        input_label.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none;")
        input_layout.addWidget(input_label)

        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("选择 .mzML 文件...")
        self.input_path_edit.setReadOnly(True)
        input_layout.addWidget(self.input_path_edit, 1)

        input_btn = QPushButton("📂 选择文件")
        input_btn.setProperty("cssClass", "filePick")
        input_btn.clicked.connect(self._on_browse_input)
        input_layout.addWidget(input_btn)
        layout.addLayout(input_layout)

        # ── 输出文件行 ──
        output_layout = QHBoxLayout()
        output_label = QLabel("输出 CSV")
        output_label.setFixedWidth(80)
        output_label.setFont(QFont(Fonts.primary, 12))
        output_label.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none;")
        output_layout.addWidget(output_label)

        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("选择输出路径...")
        self.output_path_edit.setReadOnly(True)
        output_layout.addWidget(self.output_path_edit, 1)

        output_btn = QPushButton("📂 选择路径")
        output_btn.setProperty("cssClass", "filePick")
        output_btn.clicked.connect(self._on_browse_output)
        output_layout.addWidget(output_btn)
        layout.addLayout(output_layout)

        # ── 高级参数（折叠面板） ──
        self.advanced_toggle = QPushButton("▸ 高级参数")
        self.advanced_toggle.setProperty("cssClass", "secondary")
        self.advanced_toggle.setFixedWidth(120)
        self.advanced_toggle.clicked.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle)

        # 高级参数面板（初始折叠）
        self.advanced_panel = QWidget()
        self.advanced_panel.setMaximumHeight(0)
        self.advanced_panel.setVisible(False)
        adv_layout = QVBoxLayout(self.advanced_panel)
        adv_layout.setContentsMargins(0, 8, 0, 0)
        adv_layout.setSpacing(8)

        # m/z 范围
        mz_layout = QHBoxLayout()
        mz_layout.addWidget(QLabel("m/z 范围"))
        self.mz_min_spin = QDoubleSpinBox()
        self.mz_min_spin.setRange(0.0, 100000.0)
        self.mz_min_spin.setValue(50.0)
        self.mz_min_spin.setDecimals(1)
        self.mz_min_spin.setSuffix(" Da")
        mz_layout.addWidget(self.mz_min_spin)

        mz_layout.addWidget(QLabel("—"))
        self.mz_max_spin = QDoubleSpinBox()
        self.mz_max_spin.setRange(0.0, 100000.0)
        self.mz_max_spin.setValue(2000.0)
        self.mz_max_spin.setDecimals(1)
        self.mz_max_spin.setSuffix(" Da")
        mz_layout.addWidget(self.mz_max_spin)
        mz_layout.addStretch()
        adv_layout.addLayout(mz_layout)

        # 容差
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("容差 (ppm)"))
        self.ppm_spin = QDoubleSpinBox()
        self.ppm_spin.setRange(0.0, 1000.0)
        self.ppm_spin.setValue(10.0)
        self.ppm_spin.setDecimals(1)
        self.ppm_spin.setSuffix(" ppm")
        tol_layout.addWidget(self.ppm_spin)

        tol_layout.addWidget(QLabel("容差 (Da)"))
        self.da_spin = QDoubleSpinBox()
        self.da_spin.setRange(0.0, 100.0)
        self.da_spin.setValue(0.01)
        self.da_spin.setDecimals(4)
        self.da_spin.setSuffix(" Da")
        tol_layout.addWidget(self.da_spin)
        tol_layout.addStretch()
        adv_layout.addLayout(tol_layout)

        # 强度过滤
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("强度下限"))
        self.int_min_spin = QDoubleSpinBox()
        self.int_min_spin.setRange(0.0, 1e12)
        self.int_min_spin.setSpecialValueText("(无)")
        self.int_min_spin.setValue(0.0)
        int_layout.addWidget(self.int_min_spin)

        int_layout.addWidget(QLabel("强度上限"))
        self.int_max_spin = QDoubleSpinBox()
        self.int_max_spin.setRange(0.0, 1e12)
        self.int_max_spin.setSpecialValueText("(无)")
        self.int_max_spin.setValue(0.0)
        int_layout.addWidget(self.int_max_spin)
        int_layout.addStretch()
        adv_layout.addLayout(int_layout)

        # 最大谱图数 + 重建索引
        spec_layout = QHBoxLayout()
        spec_layout.addWidget(QLabel("最大谱图数"))
        self.max_spec_spin = QSpinBox()
        self.max_spec_spin.setRange(0, 1000000)
        self.max_spec_spin.setValue(0)
        self.max_spec_spin.setSpecialValueText("0 (全部)")
        spec_layout.addWidget(self.max_spec_spin)

        self.build_index_cb = QCheckBox("重建 mzML 索引")
        spec_layout.addWidget(self.build_index_cb)
        spec_layout.addStretch()
        adv_layout.addLayout(spec_layout)

        layout.addWidget(self.advanced_panel)

        # ── 底部：进度条 + 统计 + 运行按钮 ──
        bottom_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        bottom_layout.addWidget(self.progress_bar, 1)

        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont(Fonts.primary, 11))
        self.stats_label.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none;")
        bottom_layout.addWidget(self.stats_label)

        self.result_label = QLabel("")
        self.result_label.setFont(QFont(Fonts.primary, 11))
        self.result_label.setStyleSheet(f"color: {Colors.success}; border: none; background: none;")
        self.result_label.setOpenExternalLinks(False)
        self.result_label.linkActivated.connect(self._on_open_output_dir)
        bottom_layout.addWidget(self.result_label)

        self.run_btn = QPushButton("▶ 开始运行")
        self.run_btn.setProperty("cssClass", "primary")
        self.run_btn.setFixedWidth(110)
        self.run_btn.clicked.connect(self._on_run)
        self.run_btn.setEnabled(False)  # 初始禁用（未选文件）
        bottom_layout.addWidget(self.run_btn)

        layout.addLayout(bottom_layout)

    # === 参数校验 ===

    def _connect_validation(self):
        """连接所有控件的 change 信号到统一的校验方法。"""
        self.input_path_edit.textChanged.connect(self._validate)
        self.output_path_edit.textChanged.connect(self._validate)
        self.mz_min_spin.valueChanged.connect(self._validate)
        self.mz_max_spin.valueChanged.connect(self._validate)
        self.ppm_spin.valueChanged.connect(self._validate)
        self.da_spin.valueChanged.connect(self._validate)

    def _validate(self):
        """
        校验所有参数。有任何不合法项则禁用运行按钮。

        校验规则:
          - 输入 mzML 文件必须存在
          - 输出 CSV 目录必须存在
          - mz_min < mz_max，且均 >= 0
          - ppm_tol >= 0, da_tol >= 0
        """
        valid = True

        # 输入文件
        input_path = Path(self.input_path_edit.text())
        if not input_path.exists() or input_path.suffix.lower() != ".mzml":
            self.input_path_edit.setStyleSheet(
                f"background-color: {Colors.bg_input}; border: 1px solid {Colors.error}; "
                f"border-radius: 6px; padding: 6px 10px; color: {Colors.text_primary};"
            )
            valid = False
        else:
            self.input_path_edit.setStyleSheet("")  # 恢复默认 QSS

        # 输出目录
        output_path = Path(self.output_path_edit.text())
        if not output_path.parent.exists() or output_path.suffix.lower() != ".csv":
            valid = False

        # m/z 范围
        if self.mz_min_spin.value() >= self.mz_max_spin.value():
            valid = False
        if self.mz_min_spin.value() < 0:
            valid = False

        # 容差
        if self.ppm_spin.value() < 0 or self.da_spin.value() < 0:
            valid = False

        self.run_btn.setEnabled(valid and not self._is_running)

    # === 交互处理 ===

    def _on_browse_input(self):
        """选择输入 mzML 文件。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 mzML 文件", "", "mzML Files (*.mzML)"
        )
        if path:
            self.input_path_edit.setText(path)
            # 自动建议输出路径
            if not self.output_path_edit.text():
                default_out = str(Path(path).with_suffix("")) + "_ion_zenith.csv"
                self.output_path_edit.setText(default_out)

    def _on_browse_output(self):
        """选择输出 CSV 路径。"""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 CSV", "", "CSV Files (*.csv)"
        )
        if path:
            self.output_path_edit.setText(path)

    def _toggle_advanced(self):
        """展开/收起高级参数面板（带动画）。"""
        self._advanced_expanded = not self._advanced_expanded

        if self._advanced_expanded:
            self.advanced_panel.setVisible(True)
            target_height = self.advanced_panel.sizeHint().height()
            self.advanced_toggle.setText("▾ 高级参数")
        else:
            target_height = 0
            self.advanced_toggle.setText("▸ 高级参数")

        # 动画
        anim = QPropertyAnimation(self.advanced_panel, b"maximumHeight")
        anim.setDuration(300)
        anim.setStartValue(self.advanced_panel.height())
        anim.setEndValue(target_height)
        anim.setEasingCurve(QEasingCurve.InOutQuad)
        anim.start()

        if not self._advanced_expanded:
            # 动画结束后隐藏
            anim.finished.connect(lambda: self.advanced_panel.setVisible(False))

    def _on_run(self):
        """启动离子天顶分析。"""
        if self._is_running:
            return

        self._is_running = True
        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.stats_label.setText("")
        self.result_label.setText("")

        # 搜集参数
        params = {
            "input_mzml": self.input_path_edit.text(),
            "output_csv": self.output_path_edit.text(),
            "mz_min": self.mz_min_spin.value(),
            "mz_max": self.mz_max_spin.value(),
            "ppm_tol": self.ppm_spin.value(),
            "da_tol": self.da_spin.value(),
            "intensity_min": self.int_min_spin.value() if self.int_min_spin.value() > 0 else None,
            "intensity_max": self.int_max_spin.value() if self.int_max_spin.value() > 0 else None,
            "max_spectra": self.max_spec_spin.value(),
            "build_index": self.build_index_cb.isChecked(),
            "show_progress": True,
        }

        # 禁用所有高级参数控件
        self._set_advanced_enabled(False)

        self._worker = IonZenithWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.stats.connect(self._on_stats)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, scanned: int, total: int):
        """更新进度条。total=0 时使用 indeterminate 模式。"""
        if total > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(scanned / total * 100))
        else:
            self.progress_bar.setRange(0, 0)  # 不确定模式
        self.stats_label.setText(f"扫描 {scanned} 张谱图")

    def _on_stats(self, ms1_count: int, peaks: int):
        """更新实时统计。"""
        self.stats_label.setText(f"MS1: {ms1_count} | 峰: {peaks}")

    def _on_finished(self, ion_count: int, elapsed: float, output_path: str):
        """分析完成。"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.result_label.setText(
            f'✅ {ion_count} 个离子 | {elapsed:.1f}s | <a href="{output_path}">📂 打开</a>'
        )
        self._reset_ui()

    def _on_error(self, message: str):
        """分析出错。"""
        self.result_label.setText(f"❌ {message}")
        self.result_label.setStyleSheet(f"color: {Colors.error}; border: none; background: none;")
        self._reset_ui()

    def _on_open_output_dir(self, path: str):
        """在文件管理器中打开输出目录。"""
        dir_path = str(Path(path).parent)
        if os.path.exists(dir_path):
            os.startfile(dir_path)

    def _set_advanced_enabled(self, enabled: bool):
        """启用/禁用高级参数面板中的所有控件。"""
        self.advanced_toggle.setEnabled(enabled)
        for spin in [self.mz_min_spin, self.mz_max_spin, self.ppm_spin,
                      self.da_spin, self.int_min_spin, self.int_max_spin,
                      self.max_spec_spin]:
            spin.setEnabled(enabled)
        self.build_index_cb.setEnabled(enabled)

    def _reset_ui(self):
        """恢复 UI 到可操作状态。"""
        self._is_running = False
        self.run_btn.setEnabled(True)
        self._set_advanced_enabled(True)
        self._validate()


# 将 IonZenithCard 集成到 PreprocessingPage.__init__ 中：
# 在 PreprocessingPage.__init__ 的 layout.addStretch() 之前插入:
#   self.ion_zenith_card = IonZenithCard()
#   layout.addWidget(self.ion_zenith_card)
```

- [ ] **Step 2: 修改 PreprocessingPage.__init__ 集成 IonZenithCard**

修改 `PreprocessingPage.__init__` 中的 `layout.addStretch()` 前一行（原为 `self.ion_zenith_card = None  # type: ignore`）：

确认替换 `PreprocessingPage.__init__` 中的占位行为：

文件 `gamstekpeaking/pages/preprocessing.py` 中，找到以下代码段：

```python
        # 离子天顶卡片（预留，Task 8 补充）
        self.ion_zenith_card = None  # type: ignore

        layout.addStretch()
```

替换为：

```python
        # 离子天顶卡片
        self.ion_zenith_card = IonZenithCard()
        layout.addWidget(self.ion_zenith_card)

        layout.addStretch()
```

- [ ] **Step 3: 验证语法**

```powershell
python -c "import sys; sys.path.insert(0, 'gamstekpeaking'); from pages.preprocessing import PreprocessingPage, IonZenithCard; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: 提交**

```bash
git add gamstekpeaking/pages/preprocessing.py
git commit -m "feat(gamstekpeaking): add IonZenithCard with collapsible advanced params"
```

---

### Task 9: 入口文件 main.py + 占位页面

**Files:**
- Create: `gamstekpeaking/main.py`
- Create: `gamstekpeaking/pages/peak_finding.py`（占位）
- Create: `gamstekpeaking/pages/settings.py`（占位）

**Interfaces:**
- Consumes: `GAMSTEKPEAKingWindow` from `app`, `PreprocessingPage` from `pages.preprocessing`
- Produces: 可启动的应用程序入口

- [ ] **Step 1: 创建 main.py**

```python
"""
main.py — GAMSTEKPEAKing 应用入口
==================================
启动 QApplication，加载全局 QSS 样式，创建主窗口并注册所有功能页面。

用法:
    python main.py          # 正常启动
    python main.py --debug  # 调试模式（额外日志输出）

依赖: PySide6, app, theme, pages.preprocessing, pages.peak_finding, pages.settings
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import QApplication, QMessageBox, QLabel
from PySide6.QtCore import Qt

# 确保 gamstekpeaking 包在 sys.path 中
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
    设置全局异常钩子，将未捕获异常写入 error.log 并弹出提示。
    避免程序静默崩溃。
    """
    def _handler(exc_type, exc_value, exc_tb):
        # 键盘中断正常退出
        if exc_type is KeyboardInterrupt:
            sys.exit(0)

        # 记录到 error.log
        log_path = _HERE / "error.log"
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
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
    """应用主入口。"""
    setup_exception_handler()

    debug = "--debug" in sys.argv

    # 创建 QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("GAMSTEKPEAKing")
    app.setOrganizationName("LinShuhaiLAB")

    # 加载全局样式
    app.setStyleSheet(global_stylesheet())

    # 创建主窗口
    window = GAMSTEKPEAKingWindow()

    # 注册页面（按侧边栏顺序）
    window.add_page("🏔", "前处理", PreprocessingPage(), enabled=True)
    window.add_page("📊", "寻峰", PeakFindingPage(), enabled=False)
    window.add_page("📈", "定量", QLabel("定量分析 — 敬请期待"), enabled=False)
    window.add_page("🔬", "模型", QLabel("模型管理 — 敬请期待"), enabled=False)
    window.add_page("⚙", "设置", SettingsPage(), enabled=False)

    window.show()

    if debug:
        print("[DEBUG] GAMSTEKPEAKing 已启动")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 创建占位页面 peak_finding.py**

```python
"""
peak_finding.py — 寻峰板块（占位）
==================================
未来整合 DETR 模型预测、EIC 可视化、峰面积定量等寻峰功能。
当前为占位页面，显示"敬请期待"。
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from theme import Colors, Fonts


class PeakFindingPage(QWidget):
    """寻峰板块占位页面。"""
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
```

- [ ] **Step 3: 创建占位页面 settings.py**

```python
"""
settings.py — 设置板块（占位）
==============================
未来提供全局偏好设置：主题切换、语言选择、默认路径、GPU 选择等。
当前为占位页面，显示"敬请期待"。
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from theme import Colors, Fonts


class SettingsPage(QWidget):
    """设置板块占位页面。"""
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

        hint = QLabel("将提供主题/语言/默认路径/GPU 等全局偏好设置")
        hint.setFont(QFont(Fonts.primary, 12))
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {Colors.text_disabled}; border: none; background: none;")
        layout.addWidget(hint)
```

- [ ] **Step 4: 验证 main.py 启动（无 GUI 模式检查导入）**

```powershell
python -c "import sys; sys.path.insert(0, 'gamstekpeaking'); from main import main; print('Import OK')"
```
Expected: `Import OK`

- [ ] **Step 5: 提交**

```bash
git add gamstekpeaking/main.py gamstekpeaking/pages/peak_finding.py gamstekpeaking/pages/settings.py
git commit -m "feat(gamstekpeaking): add entry point and placeholder pages"
```

---

### Task 10: README 文件

**Files:**
- Create: `gamstekpeaking/README.md`
- Create: `gamstekpeaking/pages/README_pages.md`
- Create: `gamstekpeaking/workers/README_workers.md`
- Create: `gamstekpeaking/engine/README_engine.md`
- Create: `gamstekpeaking/assets/README_assets.md`

`bin/README_bin.md` 已在 Task 2 创建，此处跳过。

- [ ] **Step 1: 创建根目录 README.md**

````markdown
# 🏔 GAMSTEKPEAKing

QuanFormer 项目的下一代统一桌面应用 — 质谱代谢组学全流程工作台。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
python main.py
```

## 目录导航

| 目录 | 说明 | 详情 |
|------|------|------|
| `pages/` | 功能页面模块 | [README_pages.md](pages/README_pages.md) |
| `workers/` | 后台工作线程 | [README_workers.md](workers/README_workers.md) |
| `engine/` | 模型推理引擎（预留） | [README_engine.md](engine/README_engine.md) |
| `bin/` | msdata2mzml 运行时 | [README_bin.md](bin/README_bin.md) |
| `assets/` | 静态资源 | [README_assets.md](assets/README_assets.md) |

## 依赖

- Python 3.10 ~ 3.11
- PySide6 ≥ 6.5.0
- pymzml ≥ 0.9.0
- numpy ≥ 1.24.0
- tqdm ≥ 4.65.0

## 架构

```
┌──────────┬────────────────────────────────┐
│  侧边栏   │  QStackedWidget（页面路由）     │
│  导航     │  ┌──────────────────────────┐ │
│  🏔 前处理 │  │ 格式转换 / 离子天顶 / ... │ │
│  📊 寻峰   │  └──────────────────────────┘ │
│  ...      │                                │
└──────────┴────────────────────────────────┘
```
````

- [ ] **Step 2: 创建 pages/README_pages.md**

````markdown
# pages/ — 功能页面模块

GAMSTEKPEAKing 的各功能板块，每个板块一个 `.py` 文件，由 `app.py` 的侧边栏通过 `add_page()` 注册并路由。

## 文件清单

| 文件 | 作用 | 状态 |
|------|------|------|
| `preprocessing.py` | 前处理板块：格式转换（msdata→mzML）+ 离子天顶（MS1→CSV） | ✅ 已上线 |
| `peak_finding.py` | 寻峰板块：DETR 模型预测、EIC 可视化、峰面积定量 | 🔒 占位 |
| `settings.py` | 设置板块：主题/语言/路径/GPU 偏好 | 🔒 占位 |

## 页面开发指南

每个页面是一个 `QWidget` 子类，在 `main.py` 中通过以下方式注册：

```python
window.add_page("🏔", "前处理", PreprocessingPage(), enabled=True)
```

参数：
- `icon` — emoji 图标字符
- `name` — 侧边栏显示名称
- `widget` — QWidget 实例
- `enabled` — 是否启用（False = 灰色禁用）
````

- [ ] **Step 3: 创建 workers/README_workers.md**

````markdown
# workers/ — 后台工作线程模块

封装耗时操作为 `QThread` 子类，通过 Qt Signal 与 UI 通信，确保界面不阻塞。

## 文件清单

| 文件 | 作用 | 核心 Signal |
|------|------|------------|
| `converter.py` | msdata→mzML 格式转换 | `progress(current, total)`, `file_done(index, ok, info)`, `error(msg)` |
| `ion_zenith.py` | 离子天顶算法（MS1→CSV） | `progress(scanned, total)`, `stats(ms1, peaks)`, `finished(ions, elapsed, path)`, `error(msg)` |

## 线程开发指南

1. 继承 `QThread`，重写 `run()` 方法
2. 定义 Signal 类属性用于对外通信
3. 在 `run()` 中捕获异常，通过 `error.emit()` 传递
4. UI 层通过 `worker.start()` 启动，`worker.quit() + wait()` 终止
````

- [ ] **Step 4: 创建 engine/README_engine.md**

````markdown
# engine/ — 模型推理引擎（预留）

未来从 `model/quanformer/` 逐步迁移模型加载、推理、预测逻辑至此。

## 规划

| 子模块 | 预期功能 |
|--------|---------|
| `loader.py` | 模型权重安全加载（safe_torch_load 封装） |
| `predictor.py` | DETR 推理管线（EIC 提取 + 预测 + 后处理） |
| `device.py` | 设备自动选择（CUDA > MPS > CPU） |

## 设计原则

- 与 `pages/` 和 `workers/` 解耦，engine 不 import UI 代码
- 所有模型操作通过 `engine/` 统一接口，未来 GUI 只依赖 engine
- 保持与现有 `model/quanformer/` 的兼容性，逐步替换而非一次性重写
````

- [ ] **Step 5: 创建 assets/README_assets.md**

````markdown
# assets/ — 静态资源

存放 GAMSTEKPEAKing 的图标、图片等静态资源文件。

## 文件清单

| 文件 | 作用 |
|------|------|
| `logo.png` | 应用 Logo（待设计） |
````

- [ ] **Step 6: 提交**

```bash
git add gamstekpeaking/README.md gamstekpeaking/pages/README_pages.md gamstekpeaking/workers/README_workers.md gamstekpeaking/engine/README_engine.md gamstekpeaking/assets/README_assets.md
git commit -m "docs(gamstekpeaking): add all README files"
```

---

### Task 11: 端到端烟雾测试

**Files:**
- Create: `gamstekpeaking/test_smoke.py`（临时测试脚本，确认后可删除）

**Interfaces:**
- Consumes: 所有已创建的模块
- Produces: 通过/失败报告

- [ ] **Step 1: 运行导入完整性测试**

```powershell
python -c "
import sys; sys.path.insert(0, 'gamstekpeaking')
from theme import Colors, Fonts, global_stylesheet
from app import GAMSTEKPEAKingWindow, SidebarButton
from workers.converter import MsdataConverter
from workers.ion_zenith import IonZenithWorker
from pages.preprocessing import PreprocessingPage, ConversionCard, IonZenithCard
from pages.peak_finding import PeakFindingPage
from pages.settings import SettingsPage
print('ALL IMPORTS OK')
"
```
Expected: `ALL IMPORTS OK`

- [ ] **Step 2: 验证 bin/ 目录完整性**

```powershell
$binOk = (Test-Path gamstekpeaking/bin/msdata2mzml.exe) -and (Test-Path gamstekpeaking/bin/share/OpenMS)
if ($binOk) { Write-Host "BIN CHECK OK" } else { Write-Host "BIN CHECK FAILED" }
```
Expected: `BIN CHECK OK`

- [ ] **Step 3: 验证 QApplication 可创建（无头模式检查）**

```powershell
python -c "
import sys; sys.path.insert(0, 'gamstekpeaking')
from PySide6.QtWidgets import QApplication
from app import GAMSTEKPEAKingWindow
from pages.preprocessing import PreprocessingPage
from pages.peak_finding import PeakFindingPage
from pages.settings import SettingsPage
app = QApplication(sys.argv)
win = GAMSTEKPEAKingWindow()
win.add_page('x', 'test', PreprocessingPage())
print('WINDOW CREATED OK')
app.quit()
"
```
Expected: `WINDOW CREATED OK`

- [ ] **Step 4: 检查 dev_log 更新**

确认 `dev_log.md` 已记录本次实现。

- [ ] **Step 5: 提交**

```bash
git commit --allow-empty -m "test(gamstekpeaking): pass end-to-end smoke test"
```

---

## 验收标准总览

| # | 任务 | 验收标准 |
|---|------|---------|
| 1 | 项目骨架 | `gamstekpeaking/` 目录树完整，`pip install -r requirements.txt` 无报错 |
| 2 | bin 整合 | `gamstekpeaking/bin/msdata2mzml.exe` 存在，README_bin.md 内容完整 |
| 3 | theme.py | `Colors.accent` 输出 `#38BDF8`，`global_stylesheet()` 返回非空 QSS |
| 4 | app.py | 可创建窗口，`add_page()` 后侧边栏按钮正常渲染 |
| 5 | converter.py | `MsdataConverter` 可实例化，Signal 签名正确 |
| 6 | ion_zenith.py | `IonZenithWorker` 可实例化，Signal 签名正确 |
| 7 | 格式转换卡片 | 拖拽区 + 文件列表 + 输出选择 + 运行按钮布局正确，可添加/移除文件 |
| 8 | 离子天顶卡片 | 文件选择 + 折叠面板 + 参数校验 + 运行按钮布局正确 |
| 9 | main.py | `python main.py` 可启动窗口，5 个页面已注册，前处理页为默认激活 |
| 10 | README | 6 份 README 齐全，内容覆盖文件清单与接口说明 |
| 11 | 烟雾测试 | 全部 import 成功，bin 目录完整，窗口可创建 |
