"""
preprocessing.py — 前处理板块
==============================
GAMSTEKPEAKing 首批上线板块，包含两个功能卡片：
  1. 格式转换 — msdata → mzML（拖拽 + 批量转换）
  2. 离子天顶 — 遍历 MS1 → 聚合 → CSV

每个卡片封装为独立的 QFrame 子类，通过 Signal 与后台线程通信。
PreprocessingPage 作为这两个卡片的容器，被 app.py 的侧边栏路由加载。

依赖: PySide6, workers.converter, workers.ion_zenith, theme
"""

import os
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QFileDialog, QProgressBar, QComboBox,
    QSizePolicy, QScrollArea, QLineEdit, QDoubleSpinBox,
    QSpinBox, QCheckBox,
)
from theme import Colors, Fonts
from workers.converter import MsdataConverter
from workers.ion_zenith import IonZenithWorker


# ============================================================
# 内部组件: 拖拽区域
# ============================================================

class _DropZone(QFrame):
    """
    拖拽区域组件。支持点击选择文件和拖拽文件。
    拖入时边框高亮 + 背景微亮，拖出/释放后恢复默认样式。
    可通过 allowed_extensions 参数限制可接受的文件后缀。
    """

    files_selected = Signal(list)  # (file_paths: list[str]) — 用户选择的文件路径列表

    def __init__(self, allowed_extensions: list[str] | None = None, parent=None):
        """
        Args:
            allowed_extensions: 允许的文件后缀列表（如 ['.msdata']），None=不限制
            parent: 父级 widget
        """
        super().__init__(parent)
        self._allowed_exts = allowed_extensions or []
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setStyleSheet(f"""
            _DropZone {{
                border: 2px dashed {Colors.border};
                border-radius: 8px;
                background-color: transparent;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        icon = QLabel("📂")
        icon.setFont(QFont(Fonts.primary, 24))
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("background: transparent; border: none;")  # 修复：去掉深色背景
        layout.addWidget(icon)

        hint = QLabel("拖拽文件到此处\n或 点击选择文件")
        hint.setFont(QFont(Fonts.primary, 13))
        hint.setStyleSheet(f"color: {Colors.text_secondary}; border: none; background: none;")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)

    def mousePressEvent(self, event):
        """点击时弹出文件选择对话框，根据 allowed_extensions 动态生成过滤器。"""
        if self._allowed_exts:
            exts = " ".join(f"*.{e.lstrip('.')}" for e in self._allowed_exts)
            filter_str = f"Allowed Files ({exts})"
        else:
            filter_str = "All Files (*)"
        files, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", filter_str)
        if files:
            self.files_selected.emit(files)

    def _accept_ext(self, filepath: str) -> bool:
        """检查文件后缀是否在允许列表中。无限制时一律接受。"""
        if not self._allowed_exts:
            return True
        return any(filepath.lower().endswith(ext.lower()) for ext in self._allowed_exts)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖入允许后缀的文件时高亮边框。"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(self._accept_ext(url.toLocalFile()) for url in urls):
                event.acceptProposedAction()
                self.setStyleSheet(f"""
                    _DropZone {{
                        border: 2px dashed {Colors.accent};
                        border-radius: 8px;
                        background-color: rgba(220, 38, 38, 0.05);
                    }}
                """)

    def dragLeaveEvent(self, event):
        """拖出时恢复默认虚线边框。"""
        self.setStyleSheet(f"""
            _DropZone {{
                border: 2px dashed {Colors.border};
                border-radius: 8px;
                background-color: transparent;
            }}
        """)

    def dropEvent(self, event: QDropEvent):
        """释放时收集符合后缀条件的文件路径并发出信号。"""
        self.setStyleSheet(f"""
            _DropZone {{
                border: 2px dashed {Colors.border};
                border-radius: 8px;
                background-color: transparent;
            }}
        """)
        urls = event.mimeData().urls()
        files = [url.toLocalFile() for url in urls if self._accept_ext(url.toLocalFile())]
        if files:
            self.files_selected.emit(files)


# ============================================================
# 内部组件: 文件列表项
# ============================================================

class _FileListItem(QWidget):
    """
    文件列表项组件。
    显示: 状态图标 + 文件名 + 附加信息 + 移除按钮(×)
    状态: ⏳ 等待 / 🔄 转换中 / ✅ 成功 / ❌ 失败
    """

    removed = Signal(str)  # (file_path: str) — 请求从列表移除此文件

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setFixedHeight(32)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # 状态图标（初始 ⏳ 等待）
        self.status_icon = QLabel("⏳")
        self.status_icon.setFixedWidth(24)
        self.status_icon.setFont(QFont(Fonts.primary, 13))
        layout.addWidget(self.status_icon)

        # 文件名（取 basename）
        fname = os.path.basename(file_path)
        name_label = QLabel(fname)
        name_label.setFont(QFont(Fonts.primary, 12))
        name_label.setStyleSheet(f"color: {Colors.text_primary}; border: none; background: none;")
        layout.addWidget(name_label, 1)

        # 附加信息（文件大小/错误消息）
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

        # 整行悬停效果
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
        更新状态图标和附加信息。

        Args:
            status: "waiting" | "running" | "success" | "failed"
            info: 附加描述文字（如文件大小或错误消息）
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
            self.info_label.setStyleSheet(
                f"color: {Colors.error}; border: none; background: none; font-size: 10px;"
            )
            self.setToolTip(info)  # 悬停显示完整错误消息
        elif status == "success":
            self.info_label.setStyleSheet(
                f"color: {Colors.success}; border: none; background: none; font-size: 10px;"
            )
        else:
            self.info_label.setStyleSheet(
                f"color: {Colors.text_secondary}; border: none; background: none; font-size: 10px;"
            )


# ============================================================
# 功能卡片 1: 格式转换
# ============================================================

class ConversionCard(QFrame):
    """
    格式转换功能卡片。

    提供通用格式批量转换界面（当前支持 msdata → mzML，未来可扩展更多格式对）：
      - 格式选择器: 源格式 → 目标格式
      - _DropZone: 拖拽/点击添加文件（根据源格式过滤后缀）
      - 文件列表: 每行显示状态图标、文件名、转换结果或错误
      - 输出目录选择: 默认（同输入目录）或自定义路径
      - 进度条 + 运行按钮: 异步执行，实时更新
    """

    # 支持的格式对: (源后缀, 目标后缀, 显示名)
    FORMAT_PAIRS = [
        ("msdata", "mzML", "msdata → mzML"),
        # 未来扩展: (".raw", "mzML", "Thermo RAW → mzML"),
        # 未来扩展: (".d", "mzML", "Bruker .d → mzML"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files: dict[str, _FileListItem] = {}  # {file_path: list_item_widget}
        self._converter: MsdataConverter | None = None
        self._is_running = False

        self._build_ui()
        # 通过 objectName 匹配全局 QSS 中的卡片样式
        self.setObjectName("conversionCard")

    def _build_ui(self):
        """构建卡片 UI 布局。样式由全局 QSS 中 QFrame#conversionCard 提供。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # ── 卡片标题行 ──
        title_layout = QHBoxLayout()
        title = QLabel("📦 格式转换")
        title.setFont(QFont(Fonts.primary, 14))
        title.setStyleSheet(
            f"color: {Colors.text_primary}; font-weight: bold; border: none; background: none;"
        )
        title_layout.addWidget(title)

        # 格式选择器: [源格式 ▼] → [目标格式]
        self.format_combo = QComboBox()
        for src, dst, label in self.FORMAT_PAIRS:
            self.format_combo.addItem(label, (src, dst))
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        title_layout.addWidget(self.format_combo)
        title_layout.addStretch()
        layout.addLayout(title_layout)

        # ── 分隔线 ──
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {Colors.border}; max-height: 1px;")
        layout.addWidget(sep)

        # ── 拖拽区域 ──
        self.drop_zone = _DropZone(allowed_extensions=[".msdata"])  # 初始默认 msdata
        self.drop_zone.files_selected.connect(self._on_files_added)
        layout.addWidget(self.drop_zone)

        # ── 输出目录选择行 ──
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

        # ── 文件列表（可滚动） ──
        self.file_list_container = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_container)
        self.file_list_layout.setContentsMargins(0, 0, 0, 0)
        self.file_list_layout.setSpacing(2)
        self.file_list_layout.addStretch()  # 底部弹簧，列表为空时撑住

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.file_list_container)
        scroll.setMaximumHeight(160)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {Colors.border};
                border-radius: 6px;
                background-color: {Colors.bg_input};
            }}
        """)
        layout.addWidget(scroll)

        # ── 底部: 进度条 + 运行按钮 ──
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
        self.run_btn.setEnabled(False)  # 初始无文件，禁用
        bottom_layout.addWidget(self.run_btn)

        layout.addLayout(bottom_layout)

    # ── 事件处理 ──

    def _on_format_changed(self, index: int):
        """格式对切换时更新拖拽区允许的后缀。"""
        src_ext, dst_ext = self.format_combo.currentData()
        self.drop_zone._allowed_exts = [f".{src_ext}"]

    def _on_files_added(self, files: list[str]):
        """拖拽或选择文件后，追加到文件列表（自动去重）。"""
        for f in files:
            f = os.path.normpath(f)
            if f in self._files:
                continue  # 去重
            item = _FileListItem(f)
            item.removed.connect(self._on_file_removed)
            # 插入到 stretch 之前（倒数第一个位置之前）
            self.file_list_layout.insertWidget(self.file_list_layout.count() - 1, item)
            self._files[f] = item
        # 有文件时启用运行按钮
        self.run_btn.setEnabled(bool(self._files))

    def _on_file_removed(self, file_path: str):
        """从列表中移除指定文件。列表为空时禁用运行按钮。"""
        if file_path in self._files:
            item = self._files.pop(file_path)
            self.file_list_layout.removeWidget(item)
            item.deleteLater()
        self.run_btn.setEnabled(bool(self._files))

    def _on_output_changed(self, index: int):
        """输出目录切换为「自定义」时弹出目录选择器。"""
        if self.output_combo.currentData() == "custom":
            dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
            if dir_path:
                self.output_combo.blockSignals(True)
                self.output_combo.insertItem(0, os.path.basename(dir_path), dir_path)
                self.output_combo.setCurrentIndex(0)
                self.output_combo.blockSignals(False)
            else:
                # 用户取消 → 切回默认
                self.output_combo.blockSignals(True)
                self.output_combo.setCurrentIndex(0)
                self.output_combo.blockSignals(False)

    def _on_run(self):
        """开始批量转换。校验后启动 MsdataConverter 后台线程。"""
        if self._is_running or not self._files:
            return

        # 检查 exe 是否存在
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
        """更新进度条百分比。"""
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

        # 检查是否全部完成（最后一个文件的回调）
        if index == len(self._files) - 1:
            self._on_all_done()

    def _on_converter_error(self, message: str):
        """后台线程致命错误（如 exe 不存在）。"""
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
        """显示错误信息（输出到 stderr，后续可接入统一通知组件）。"""
        import sys
        print(f"[ConversionCard Error] {message}", file=sys.stderr)


# ============================================================
# 功能卡片 2: 离子天顶
# ============================================================

class IonZenithCard(QFrame):
    """
    离子天顶功能卡片。

    提供 mzML MS1 谱图遍历 → CSV 输出的界面：
      - 输入/输出文件选择行
      - 可折叠高级参数面板（QPropertyAnimation 平滑展开/收起）
      - 实时参数校验（不合法时禁用运行按钮）
      - 进度条 + 实时统计 + 运行按钮
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: IonZenithWorker | None = None
        self._is_running = False
        self._advanced_expanded = False
        self._has_interacted = False  # Fix1: 用户交互后才启用校验

        self._build_ui()
        self._connect_validation()
        # 通过 objectName 匹配全局 QSS 中的卡片样式
        self.setObjectName("ionZenithCard")

    def _build_ui(self):
        """构建卡片 UI 布局。样式由全局 QSS 中 QFrame#ionZenithCard 提供。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # ── 标题行 ──
        title_layout = QHBoxLayout()
        title = QLabel("⚡ 离子天顶")
        title.setFont(QFont(Fonts.primary, 14))
        title.setStyleSheet(
            f"color: {Colors.text_primary}; font-weight: bold; border: none; background: none;"
        )
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
        self.input_path_edit.setStyleSheet("")  # 由全局 QSS 控制，校验失败时动态覆盖
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

        # ── 高级参数折叠按钮 + 帮助 ──
        toggle_row = QHBoxLayout()
        self.advanced_toggle = QPushButton("▸ 高级参数")
        self.advanced_toggle.setProperty("cssClass", "secondary")
        self.advanced_toggle.setFixedWidth(120)
        self.advanced_toggle.clicked.connect(self._toggle_advanced)
        toggle_row.addWidget(self.advanced_toggle)

        help_btn = QPushButton("?")
        help_btn.setFlat(True)
        help_btn.setFixedSize(22, 22)
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.setToolTip(
            "m/z 范围 — 只处理该质量范围内的离子\n"
            "容差 (ppm) — 质荷比相对容差，用于离子聚合\n"
            "容差 (Da) — 质荷比绝对容差，与 ppm 同时生效\n"
            "强度下限/上限 — 过滤低/高于该强度的信号，0=不过滤\n"
            "最大谱图数 — 限制处理的谱图数，0=全部\n"
            "重建 mzML 索引 — 若 mzML 缺少索引则自动重建"
        )
        help_btn.setStyleSheet(f"""
            QPushButton {{
                border: 1px solid {Colors.border};
                border-radius: 11px;
                background: {Colors.bg_card_hover};
                color: {Colors.text_secondary};
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                border-color: {Colors.accent};
                color: {Colors.accent};
                background: {Colors.bg_card};
            }}
        """)
        toggle_row.addWidget(help_btn)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        # ── 高级参数面板（初始折叠，maximumHeight=0） ──
        self.advanced_panel = QWidget()
        self.advanced_panel.setMaximumHeight(0)
        self.advanced_panel.setVisible(False)
        adv_layout = QVBoxLayout(self.advanced_panel)
        adv_layout.setContentsMargins(0, 8, 0, 0)
        adv_layout.setSpacing(8)

        # m/z 范围行
        mz_layout = QHBoxLayout()
        mz_layout.addWidget(QLabel("  m/z 范围"))
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

        # 容差行
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("  容差 (ppm)"))
        self.ppm_spin = QDoubleSpinBox()
        self.ppm_spin.setRange(0.0, 1000.0)
        self.ppm_spin.setValue(10.0)
        self.ppm_spin.setDecimals(1)
        self.ppm_spin.setSuffix(" ppm")
        tol_layout.addWidget(self.ppm_spin)

        tol_layout.addWidget(QLabel("  容差 (Da)"))
        self.da_spin = QDoubleSpinBox()
        self.da_spin.setRange(0.0, 100.0)
        self.da_spin.setValue(0.01)
        self.da_spin.setDecimals(4)
        self.da_spin.setSuffix(" Da")
        tol_layout.addWidget(self.da_spin)
        tol_layout.addStretch()
        adv_layout.addLayout(tol_layout)

        # 强度过滤行
        int_layout = QHBoxLayout()
        int_layout.addWidget(QLabel("  强度下限"))
        self.int_min_spin = QDoubleSpinBox()
        self.int_min_spin.setRange(0.0, 1e12)
        self.int_min_spin.setSpecialValueText("(无)")
        self.int_min_spin.setValue(0.0)
        int_layout.addWidget(self.int_min_spin)

        int_layout.addWidget(QLabel("  强度上限"))
        self.int_max_spin = QDoubleSpinBox()
        self.int_max_spin.setRange(0.0, 1e12)
        self.int_max_spin.setSpecialValueText("(无)")
        self.int_max_spin.setValue(0.0)
        int_layout.addWidget(self.int_max_spin)
        int_layout.addStretch()
        adv_layout.addLayout(int_layout)

        # 最大谱图数 + 重建索引
        spec_layout = QHBoxLayout()
        spec_layout.addWidget(QLabel("  最大谱图数"))
        self.max_spec_spin = QSpinBox()
        self.max_spec_spin.setRange(0, 1000000)
        self.max_spec_spin.setValue(0)
        self.max_spec_spin.setSpecialValueText("0 (全部)")
        self.max_spec_spin.setFixedWidth(130)
        spec_layout.addWidget(self.max_spec_spin)

        self.build_index_cb = QCheckBox("重建 mzML 索引")
        spec_layout.addWidget(self.build_index_cb)
        spec_layout.addStretch()
        adv_layout.addLayout(spec_layout)

        layout.addWidget(self.advanced_panel)

        # ── 底部: 进度条 + 统计 + 结果 + 运行按钮 ──
        bottom_layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        bottom_layout.addWidget(self.progress_bar, 1)

        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont(Fonts.primary, 11))
        self.stats_label.setStyleSheet(
            f"color: {Colors.text_secondary}; border: none; background: none;"
        )
        bottom_layout.addWidget(self.stats_label)

        self.result_label = QLabel("")
        self.result_label.setFont(QFont(Fonts.primary, 11))
        self.result_label.setStyleSheet(
            f"color: {Colors.success}; border: none; background: none;"
        )
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

    # ── 参数校验 ──

    def _connect_validation(self):
        """连接所有控件的值变更信号到统一校验方法。"""
        self.input_path_edit.textChanged.connect(self._validate)
        self.output_path_edit.textChanged.connect(self._validate)
        self.mz_min_spin.valueChanged.connect(self._validate)
        self.mz_max_spin.valueChanged.connect(self._validate)
        self.ppm_spin.valueChanged.connect(self._validate)
        self.da_spin.valueChanged.connect(self._validate)

    def _validate(self):
        """
        校验所有参数。任何一项不合法则禁用运行按钮。

        校验规则:
          - 输入 mzML 文件必须存在且后缀为 .mzML
          - 输出 CSV 父目录必须存在且后缀为 .csv
          - mz_min < mz_max，且均 >= 0
          - ppm_tol >= 0, da_tol >= 0
        """
        if not self._has_interacted:  # Fix1: 用户未交互时不显示红色错误
            return
        valid = True

        # 输入文件校验
        input_path = Path(self.input_path_edit.text())
        if not input_path.exists() or input_path.suffix.lower() != ".mzml":
            self.input_path_edit.setStyleSheet(
                f"background-color: {Colors.bg_input}; border: 1px solid {Colors.error}; "
                f"border-radius: 6px; padding: 6px 10px; color: {Colors.text_primary};"
            )
            valid = False
        else:
            self.input_path_edit.setStyleSheet("")  # 恢复全局 QSS

        # 输出路径校验
        output_path = Path(self.output_path_edit.text())
        if not output_path.parent.exists() or output_path.suffix.lower() != ".csv":
            valid = False

        # m/z 范围校验
        if self.mz_min_spin.value() >= self.mz_max_spin.value():
            valid = False
        if self.mz_min_spin.value() < 0:
            valid = False

        # 容差校验
        if self.ppm_spin.value() < 0 or self.da_spin.value() < 0:
            valid = False

        self.run_btn.setEnabled(valid and not self._is_running)

    # ── 交互处理 ──

    def _on_browse_input(self):
        """弹出文件选择器选择输入 .mzML 文件。同时自动建议输出 CSV 路径。"""
        self._has_interacted = True  # Fix1: 用户首次交互后启用校验
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
        """弹出保存对话框选择输出 CSV 路径。"""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 CSV", "", "CSV Files (*.csv)"
        )
        if path:
            self.output_path_edit.setText(path)

    def _toggle_advanced(self):
        """展开/收起高级参数面板。直接切换可见性，父级 QScrollArea 自适应高度。"""
        self._advanced_expanded = not self._advanced_expanded

        if self._advanced_expanded:
            self.advanced_panel.setMaximumHeight(16777215)
            self.advanced_panel.setVisible(True)
            self.advanced_toggle.setText("▾ 高级参数")
        else:
            self.advanced_panel.setVisible(False)
            self.advanced_toggle.setText("▸ 高级参数")

    def _on_run(self):
        """启动离子天顶分析。搜集所有参数传入 IonZenithWorker 后台线程。"""
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

        # 禁用高级参数控件（防止运行中修改）
        self._set_advanced_enabled(False)

        # 启动后台线程
        self._worker = IonZenithWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.stats.connect(self._on_stats)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, scanned: int, total: int):
        """更新进度条。total=0 时使用 indeterminate（不确定）模式。"""
        if total > 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(scanned / total * 100) if total else 0)
        else:
            self.progress_bar.setRange(0, 0)  # 不确定模式（来回滚动）
        self.stats_label.setText(f"扫描 {scanned} 张谱图")

    def _on_stats(self, ms1_count: int, peaks: int):
        """更新实时统计信息。"""
        self.stats_label.setText(f"MS1: {ms1_count} | 峰: {peaks}")

    def _on_finished(self, ion_count: int, elapsed: float, output_path: str):
        """分析完成，显示结果统计。"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.result_label.setText(
            f'✅ {ion_count} 个离子 | {elapsed:.1f}s | '
            f'<a href="{output_path}">📂 打开所在目录</a>'
        )
        self._reset_ui()

    def _on_error(self, message: str):
        """分析出错，显示错误信息。"""
        self.result_label.setText(f"❌ {message}")
        self.result_label.setStyleSheet(
            f"color: {Colors.error}; border: none; background: none;"
        )
        self._reset_ui()

    def _on_open_output_dir(self, path: str):
        """在 Windows 资源管理器中打开输出文件所在目录。"""
        dir_path = str(Path(path).parent)
        if os.path.exists(dir_path):
            os.startfile(dir_path)

    def _set_advanced_enabled(self, enabled: bool):
        """批量设置高级参数面板中所有控件的启用/禁用状态。"""
        self.advanced_toggle.setEnabled(enabled)
        for spin in [
            self.mz_min_spin, self.mz_max_spin, self.ppm_spin,
            self.da_spin, self.int_min_spin, self.int_max_spin,
            self.max_spec_spin,
        ]:
            spin.setEnabled(enabled)
        self.build_index_cb.setEnabled(enabled)

    def _reset_ui(self):
        """恢复 UI 到可操作状态。"""
        self._is_running = False
        self.run_btn.setEnabled(True)
        self._set_advanced_enabled(True)
        self._validate()  # 重新校验参数


# ============================================================
# 前处理板块主页面
# ============================================================

class PreprocessingPage(QWidget):
    """
    前处理板块主页面。

    包含两个功能卡片，包裹在 QScrollArea 中，内容超出时可滚动。
    由 app.py 通过 add_page("前处理", PreprocessingPage()) 注册。
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # 外层滚动区域
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {Colors.bg_primary};
            }}
        """)

        # 内容容器
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # 格式转换卡片
        self.conversion_card = ConversionCard()
        layout.addWidget(self.conversion_card)

        # 离子天顶卡片
        self.ion_zenith_card = IonZenithCard()
        layout.addWidget(self.ion_zenith_card)

        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)
