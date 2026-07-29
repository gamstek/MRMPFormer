# 代码修改记录

> 本文件记录每次代码改动的详细信息，包括时间、类型、改动人、说明、内容与理由。
> 由 `code-change-logger` skill 自动维护。

---

## 26-07-29

### 修改 #1 — 新建 reprocess.py（合并批处理脚本）
- **改动人**: AI Copilot
- **类型**: 重构
- **说明**: 将 batch_post_newtest_under_snr_filtered.py 与 rerun_snr_under_snr_filtered.py 合并为 reprocess.py，统一 `--stage snr/post/snr-post` 三模式
- **内容**: model/tools/batch/reprocess.py — 新建；统一参数 CLI（SNR + post_newtest 全部参数），post 阶段统一用 subprocess 调用以保证参数传递一致性
- **理由**: 两脚本 post_newtest 参数集各自演化已不一致；合并消除参数分叉，降低维护成本

### 修改 #2 — 弃用旧批处理脚本
- **改动人**: AI Copilot
- **类型**: 文档生成
- **说明**: batch_post_newtest_under_snr_filtered.py、rerun_snr_under_snr_filtered.py 头部加弃用注释
- **内容**: 两文件 docstring 加 `.. deprecated:: 2026-07-29` 及指向 reprocess.py 的迁移指引
- **理由**: 向后兼容保留旧文件，引导用户迁移到统一入口

### 修改 #3 — 更新开发日志
- **改动人**: AI Copilot
- **类型**: 文档生成
- **说明**: dev_log.md 新增 2026-07-29 条目
- **内容**: 记录合并重构、新文件创建、弃用标记三项
- **理由**: 按 CLAUDE.md 强制规则同步更新

### 修改 #4 — 新建 chromatogram.py（合并 mzML 色谱工具）
- **改动人**: AI Copilot
- **类型**: 重构
- **说明**: 将 mzml_export_one_chrom.py 与 read_mzml_one_group.py 合并为 chromatogram.py，统一 `list/show/export` 三个子命令
- **内容**: model/tools/mzml/chromatogram.py — 新建；共享 inspect.py 依赖，修复旧脚本 import 路径 bug（mzml_inspect_to_csv→inspect），统一色谱定位、编码处理、Q1/Q3 提取
- **理由**: 两脚本 ~70% 代码重复；共享同一依赖和概念模型；子命令模式更符合 CLI 工具惯例

### 修改 #5 — 弃用旧 mzML 色谱脚本
- **改动人**: AI Copilot
- **类型**: 文档生成
- **说明**: mzml_export_one_chrom.py、read_mzml_one_group.py 头部加弃用注释
- **内容**: 两文件 docstring 加 `.. deprecated:: 2026-07-29` 及指向 chromatogram.py 各子命令的迁移指引
- **理由**: 向后兼容保留旧文件，引导用户迁移到统一入口

### 修改 #6 — 更新开发日志（mzML 色谱工具）
- **改动人**: AI Copilot
- **类型**: 文档生成
- **说明**: dev_log.md 补充 2026-07-29 mzML 色谱工具合并条目
- **内容**: 新增三项记录：chromatogram.py 创建、旧脚本弃用、import 路径 bug 修复
- **理由**: 按 CLAUDE.md 强制规则同步更新

### 修改 #7 — 全面更新 README.md
- **改动人**: AI Copilot
- **类型**: 文档生成
- **说明**: 修正目录名、重写项目结构树、新增辅助工具章节、版本号升级
- **内容**: README.md — 5 处替换：(1) 版本号 v0.3.0→v2.0.0；(2) 项目结构树全面重写（新增 gamstekpeaking/ms2mzml/processed/tools 及 10+ 新管线脚本）；(3) 所有 `main_model`→`model`、`Quanformer/main_model`→`model`；(4) 新增「辅助工具」章节（GamSTekPeaking/ms2mzml/工具集/管线脚本）；(5) 命令路径修正
- **理由**: 原有 README 与实际目录结构严重脱节（核心目录名错误、缺失大量新组件），按 CLAUDE.md 强制规则同步更新

---

## 26-07-28

### 修改 #1 — QComboBox 下拉箭头
- **改动人**: AI Copilot
- **类型**: 调试
- **说明**: QComboBox::down-arrow 子控件缺失导致下拉按钮全白
- **内容**: theme.py — 新增 QComboBox::down-arrow 引用 combo_down_arrow.png；QComboBox::drop-down 加背景色 #EEF2F7 + hover 态
- **理由**: QSS 中 ::drop-down 有样式但 ::down-arrow 未定义，Qt 不渲染箭头；白底+无箭头=全白不可辨

### 修改 #2 — QSpinBox 上下箭头
- **改动人**: AI Copilot
- **类型**: 调试
- **说明**: QSpinBox/QDoubleSpinBox ::up-arrow/::down-arrow 缺失，高级参数面板按钮不可见
- **内容**: theme.py — 新增 ::up-button/::down-button（透明背景+圆角对齐）+ ::up-arrow/::down-arrow；_ensure_assets() 用 QTransform.rotate(180) 从 combo_down_arrow.png 生成 spin_up_arrow.png
- **理由**: 与 QComboBox 同根因——子控件未定义则不渲染；上箭头旋转复用避免手动绘制色差

### 修改 #3 — 红框误触发
- **改动人**: AI Copilot
- **类型**: 调试
- **说明**: IonZenithCard 输入框在用户未交互时即显示红色错误边框
- **内容**: preprocessing.py — IonZenithCard.__init__ 新增 _has_interacted=False；_validate() 开头 guard `if not self._has_interacted: return`；_on_browse_input() 中置 True
- **理由**: textChanged 信号在 widget 构造时就以空串 "" 触发，Path("").exists()=False 导致红框；需等用户首次交互后才启用校验

### 修改 #4 — SpinBox 按钮透明化
- **改动人**: AI Copilot
- **类型**: 重构
- **说明**: SpinBox 按钮默认背景色覆盖外层圆角，视觉上"盖在圆角上"
- **内容**: theme.py — ::up-button/::down-button background-color 由 #EEF2F7 改为 transparent；hover 由 #D1D5DB 改为 #EEF2F7；border-radius 由 5px 对齐 6px
- **理由**: 透明按钮不干扰外层 border-radius，hover 时微浮现提示可点击

### 修改 #5 — 复选框打勾
- **改动人**: AI Copilot
- **类型**: 重构
- **说明**: QCheckBox 选中状态全红填充，视觉效果像错误态
- **内容**: theme.py — QCheckBox::indicator:checked 改为白底+红框+image:url(check_icon.png)；_ensure_assets() 用 QPainter.drawPolyline 绘制 ✓ 勾号
- **理由**: 白底红勾为标准复选框范式；_ensure_assets() 统一管理所有自动生成图标

### 修改 #6 — 高级参数帮助按钮
- **改动人**: AI Copilot
- **类型**: 代码生成
- **说明**: 高级参数区域缺少参数说明入口
- **内容**: preprocessing.py — IonZenithCard._build_ui() 中新增 QPushButton("?") flat 圆形按钮，tooltip 列出 m/z 范围/容差/强度/谱图数/重建索引 6 项参数说明
- **理由**: 首次用户不了解各参数含义，悬停 tooltip 降低学习成本

### 修改 #7 — 最大谱图数宽度
- **改动人**: AI Copilot
- **类型**: 调试
- **说明**: max_spec_spin 默认宽度不足以完整显示 specialValueText "0 (全部)"
- **内容**: preprocessing.py — self.max_spec_spin.setFixedWidth(130)
- **理由**: 中文 specialValueText "0 (全部)" 需 5 字符宽，默认 ~70px 不够

---

## 26-07-15

### 修改 #1
- **改动人**: (待确认)
- **类型**: 新增
- **说明**: 创建 MRMPFormer SimCLR 对比学习完整框架；涉及 `MRMPFormer/` 目录下 8 个文件
- **内容**: models/simclr.py（ResNet50 + ProjectionHead 2048→512→128）、utils/losses.py（NT-Xent/InfoNCE 对比损失）、augmentations.py（5 种 SimCLR 增强：RandomResizedCrop/HorizontalFlip/ColorJitter/Grayscale/GaussianBlur）、dataset.py（无标签图像数据集，返回增强对）、train.py（AdamW + CosineAnnealing + 梯度累积 + TensorBoard）、extract_features.py（推理提取 2048-d backbone 特征）、__init__.py × 2、data/images/ 目录
- **理由**: 用户需要基于 ResNet50 构建特征提取器，选用对比学习方案以获取高质量通用特征

### 修改 #2
- **改动人**: (待确认)
- **类型**: 文档生成
- **说明**: 创建 MRMPFormer 设计文档；涉及 `docs/superpowers/specs/2026-07-15-mrmpformer-simclr-design.md`
- **内容**: 包含目标、文件结构、模型架构、数据增强、损失函数、训练超参数、技术约束、自审清单
- **理由**: brainstorming 流程要求设计文档

### 修改 #3
- **改动人**: (待确认)
- **类型**: 文档生成
- **说明**: 更新 dev_log.md 开发时间线；涉及 `dev_log.md`
- **内容**: 新增 2026-07-15 条目（需求分析/代码生成/文档生成 3 条）
- **理由**: CLAUDE.md 要求每次任务后更新开发日志

---

## 26-07-09

### 修改 #1
- **改动人**: 罗钊
- **类型**: 新增
- **说明**: 创建环境依赖检测与修复系统；涉及 `.github/skills/check-dependencies/check_env.py`、`check_gui.py`、`SKILL.md`；`.github/skills/fix-dependencies/fix_env.py`、`SKILL.md`
- **内容**: check_env.py 全量检测（Python版本/pip包/PyTorch-CUDA匹配/GPU架构/R运行时/模型权重/磁盘空间），输出 Markdown/JSON/quiet 三种格式，支持 --target-env 指定 conda 环境；check_gui.py tkinter 弹窗报告（深色主题/颜色标记/一键修复/Conda环境管理对话框）；fix_env.py 支持 find-env/check/fix/verify 四种子命令
- **理由**: 用户需要一套自动化环境检测工具作为项目前置条件，减少手动排查依赖的时间

### 修改 #2
- **改动人**: 罗钊
- **类型**: 修复
- **说明**: 修复 Windows GBK/UTF-8 编码错配导致 GUI 中文乱码；涉及 `check_env.py` L91-L105、`check_gui.py` L42-L65
- **内容**: check_env.py 的 run_cmd() 统一加 encoding="utf-8", errors="replace" 避免子命令输出被 GBK 误读；check_gui.py 的 run_check() 改用 --outfile 临时文件方案绕开 stdout 管道编码，新增 import tempfile
- **理由**: 中文 Windows 下 sys.stdout.encoding 在非 TTY 时自动切为 GBK，与 json.dumps(ensure_ascii=False) 的 UTF-8 输出不匹配，导致 tkinter 显示乱码

### 修改 #3
- **改动人**: 罗钊
- **类型**: 重构
- **说明**: 精简 check-dependencies 和 fix-dependencies 两个 SKILL.md；涉及 `.github/skills/check-dependencies/SKILL.md`、`.github/skills/fix-dependencies/SKILL.md`
- **内容**: 各从 ~150 行砍至 ~30 行，去除冗余的 AI 互动规则、手动回退步骤、报告模板、质量检查清单，仅保留 frontmatter 触发索引和脚本路径用法
- **理由**: GUI 弹窗已接管交互，skill 退化为 AI 知识索引层，不需要教 AI 怎么做

### 修改 #4
- **改动人**: 罗钊
- **类型**: 调整
- **说明**: README.md 新增「环境检测」小节；涉及 `README.md` L113-L136
- **内容**: 在「环境要求」与「安装」之间插入环境检测说明，包含一行检测命令和检测内容表格；同步更新 dev_log.md 开发时间线
- **理由**: 让新用户安装前先跑检测，降低环境配置失败率

### 修改 #5
- **改动人**: 罗钊
- **类型**: 重构
- **说明**: 整合 dev_log.md 冗余条目；涉及 `dev_log.md` L30-L60
- **内容**: 2026-07-07 从 17 条合并为 7 条（同功能域 utils/GUI/文档合并），2026-07-09 从 9 条合并为 4 条（check/fix 系统合并为一条）
- **理由**: 多次重复说明同类改动，日志臃肿不利于回顾

### 修改 #6
- **改动人**: 罗钊
- **类型**: 重构
- **说明**: code-change-logger SKILL.md 触发模式从每次编辑弹窗改为回答末尾询问；涉及 `.github/skills/code-change-logger/SKILL.md`；联动 `CLAUDE.md`
- **内容**: 删除「每次触发时必须先提问」规则，改为 AI 在含代码编辑的回答末尾加一句「是否登记改动人？」，用户回复后批量写入
- **理由**: 高频编辑场景下每次弹窗严重打断工作流，简化为末尾询问兼顾追溯性和用户体验
