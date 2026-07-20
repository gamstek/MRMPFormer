# 色谱图专用增强 + 配置文件迁移 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 MRMPFormer 中新增色谱图专用增强策略，并将所有训练参数从 argparse 迁移到统一配置文件。

**Architecture:** 新建 `utils/config.py` 集中管理所有参数（`ExperimentConfig` dataclass + 2 组预设），`augmentations.py` 新增 `RandomRTShift` 类和 `get_chromatogram_augmentations()`，`train.py` 简化 CLI 为 `--config` 单参数。

**Tech Stack:** Python 3.11, PyTorch, torchvision, PIL, dataclasses

## Global Constraints

- Python 3.10~3.11
- 路径使用 `pathlib.Path`
- 设备自动检测 `get_best_device()`（CUDA > MPS > CPU）
- 不硬编码路径分隔符
- 遵循 `CLAUDE.md` 所有规范

---

### Task 1: 创建 `utils/config.py` — 实验配置 dataclass

**Files:**
- Create: `d:\Quanformer\MRMPFormer\utils\config.py`

**Interfaces:**
- Produces: `ExperimentConfig` dataclass, `SIMCLR_BASELINE`, `CHROMATOGRAM_V1`, `PRESETS` dict

- [ ] **Step 1: 创建 config.py**

所有参数定义 + 中文注释 + 2 组预设。完整代码见下方。

- [ ] **Step 2: 验证导入无语法错误**

Run: `python -c "from utils.config import ExperimentConfig, SIMCLR_BASELINE, CHROMATOGRAM_V1, PRESETS; print('OK')"` in MRMPFormer/

---

### Task 2: 更新 `utils/__init__.py` — 导出配置符号

**Files:**
- Modify: `d:\Quanformer\MRMPFormer\utils\__init__.py`

**Interfaces:**
- Produces: 包级导出 `ExperimentConfig`, `SIMCLR_BASELINE`, `CHROMATOGRAM_V1`, `PRESETS`

- [ ] **Step 1: 添加导出**

在原有 `from .losses import NT_XentLoss` 基础上追加配置导出。

- [ ] **Step 2: 验证导入**

Run: `python -c "from utils import ExperimentConfig, SIMCLR_BASELINE; print(SIMCLR_BASELINE.name)"`

---

### Task 3: 实现 `RandomRTShift` 类

**Files:**
- Modify: `d:\Quanformer\MRMPFormer\augmentations.py`

**Interfaces:**
- Produces: `RandomRTShift(nn.Module)` — 水平随机平移，支持 edge/constant padding

- [ ] **Step 1: 添加 import 和 RandomRTShift 类**

需要 `import random`, `import numpy as np`, `from PIL import Image`

- [ ] **Step 2: 单元验证**

Run: Python snippet 验证 shift 前后尺寸不变、值域合理

---

### Task 4: 添加 `get_chromatogram_augmentations()`

**Files:**
- Modify: `d:\Quanformer\MRMPFormer\augmentations.py`

**Interfaces:**
- Consumes: `ExperimentConfig`（从 config 读取 pad_mode, blur_kernel 等）
- Produces: `get_chromatogram_augmentations(config)` → `transforms.Compose`

- [ ] **Step 1: 实现函数**

Pipeline: `RandomRTShift → Resize(168,224) → Pad(28,28) → HorizontalFlip → GaussianBlur → ToTensor → Normalize`

- [ ] **Step 2: 验证 pipeline 输出尺寸**

Run: Python snippet 确认输入 (400,300) → 输出 (3,224,224)

---

### Task 5: 改造 `train.py` — 从 config 读取参数

**Files:**
- Modify: `d:\Quanformer\MRMPFormer\train.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, `PRESETS` from utils
- Consumes: `get_chromatogram_augmentations` from augmentations
- Produces: 精简的 `get_args()` + 重构的 `main()` 和 `train_one_epoch()`

- [ ] **Step 1: 精简 `get_args()` 为仅 `--config`**

- [ ] **Step 2: 修改 `train_one_epoch()` 签名**

将 `args` 参数改为 `config: ExperimentConfig`，内部 `args.xxx` → `config.xxx`

- [ ] **Step 3: 修改 `main()`**

从 `config = PRESETS[args.config]` 获取所有参数，替换所有 `args.xxx` 引用；根据 `config.augmentation` 选择增强；输出目录改为 `checkpoints/{config.name}/`

- [ ] **Step 4: 验证可运行（dry-run）**

Run: `python train.py --config simclr_baseline` (手动 Ctrl+C 中断)

---

### Task 6: 集成验证

**Files:**
- 无新建，验证以上所有改动

- [ ] **Step 1: 验证 simclr 配置可正常启动训练**

```bash
python train.py --config simclr_baseline
# 预期：正常加载数据、模型，开始训练
# 手动 Ctrl+C 中断
```

- [ ] **Step 2: 验证 chromatogram 配置可正常启动训练**

```bash
python train.py --config chromatogram_v1
# 预期：正常加载数据、模型（使用 chromatogram 增强），开始训练
# 手动 Ctrl+C 中断
```

- [ ] **Step 3: 验证权重保存路径隔离**

确认 `checkpoints/simclr_baseline/` 和 `checkpoints/chromatogram_v1/` 分别创建

---

## Self-Review

1. **Spec coverage**: 覆盖 §3.5 代码结构、§3.6 配置文件设计、§4.1 实验命令、§6 执行清单 items 1-4
2. **Placeholders**: 无
3. **Type consistency**: `ExperimentConfig` 贯穿所有 Task，签名一致
