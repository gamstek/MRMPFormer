# 代码修改记录

> 本文件记录每次代码改动的详细信息，包括时间、类型、改动人、说明、内容与理由。
> 由 `code-change-logger` skill 自动维护。

---

## 26-07-20

### 修改 #1
- **改动人**: (待确认)
- **类型**: 新增
- **说明**: 创建 MRMPFormer requirements.txt；涉及 `MRMPFormer/requirements.txt`
- **内容**: 依据源码依赖追踪结果编写独立依赖配置，包含 numpy==1.26.4 / Pillow==10.4.0 / tqdm==4.68.4 / tensorboard>=2.14.0；PyTorch 按 GPU 架构分段 (RTX 50 cu128/RTX 40 cu124/RTX 30 cu124/RTX 20 cu124/CPU)，50 系默认启用
- **理由**: MRMPFormer 作为独立子项目需要专属依赖文件，与 QuanFormer 的 model/requirements.txt 解耦（不引入 pymzml/pyside6/pycocotools 等无关包）

### 修改 #2
- **改动人**: (待确认)
- **类型**: 文档生成
- **说明**: 更新 dev_log.md 开发时间线；涉及 `dev_log.md`
- **内容**: 新增 2026-07-20 条目（需求分析/代码生成 2 条）
- **理由**: CLAUDE.md 要求每次任务后更新开发日志

### 修改 #3
- **改动人**: (待确认)
- **类型**: 新增
- **说明**: 实现早停策略 + tqdm 进度条美化；涉及 `MRMPFormer/utils/config.py`、`MRMPFormer/train.py`
- **内容**: config.py 新增 early_stopping_enabled/patience/min_delta/min_epochs 四个字段；train.py 引入 tqdm 替换原 batch 级 print，进度条实时显示 running loss；训练循环内嵌早停逻辑 (相对改善率 < min_delta 累计 patience 次后触发)，保护期 100 epoch
- **理由**: 分析阶段确定方案 A（相对改善率监控）为最优；用户要求美化进度条（原"每 20 batch 打印"体验差）

### 修改 #4
- **改动人**: (待确认)
- **类型**: 新增
- **说明**: 实现 backbone 分阶段冻结机制；涉及 `MRMPFormer/models/simclr.py`、`MRMPFormer/utils/config.py`、`MRMPFormer/train.py`
- **内容**: simclr.py 拆分 backbone 为 stem/layer1~4/avgpool 独立属性（同时保留 Sequential 兼容），新增 freeze_stages 参数 (0~5) 和 _freeze_stages()/trainable_param_counts() 方法；config.py 新增 freeze_stages 配置（默认 0）；train.py 传入 freeze_stages 并输出"可训练 / 冻结"参数量统计
- **理由**: 用户希望冻结前 3 个 stage（stem+layer1~3）仅训练 layer4+投影头——对色谱图小数据集的半迁移学习策略，减少过拟合、加速训练

### 修改 #5
- **改动人**: (待确认)
- **类型**: 新增
- **说明**: 实现评估体系（Alignment + Uniformity + Retrieval P@K）；涉及 `MRMPFormer/utils/losses.py`、`MRMPFormer/utils/__init__.py`、`MRMPFormer/utils/config.py`、`MRMPFormer/train.py`；新建 `MRMPFormer/evaluate.py`
- **内容**: losses.py 新增 alignment_loss() (正样本对 MSE) 和 uniformity_loss() (高斯势对数)；__init__.py 导出；evaluate.py 支持单文件/双文件对比模式，含 infer_labels_from_paths() 从中文文件名自动推断化合物标签，compute_retrieval_metrics() (P@K) 和 compute_uniformity()，print_table() 格式化对比输出；train.py 新增 compute_alignment_uniformity() 每 N epoch 无梯度计算并写入 TensorBoard；config.py 新增 eval_metrics_every
- **理由**: 上轮分析确定仅 loss 评价太局限，需要补充 probe-free 指标 (A+U) 和标签辅助指标 (P@K)；评估脚本用于基线 vs 训练后对比，为 SOTA 追踪提供量化依据

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
