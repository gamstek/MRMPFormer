# GamstekPeaking

## 简介

GamstekPeaking是引力波智谱科学智能部研发的，用于LC-MS 代谢组学色谱峰检测与定量工具包。包括格式转换前处理、MRMPFormer和后处理三个板块。

---

### MRMFormer简介

MRMPFormer是基于 **DETR（ResNet-50 + Transformer）系列** 的色谱峰检测模型。 
核心思路：在提取离子色谱图（EIC）生成的 ROI 图像上训练目标检测网络，识别真实色谱峰并定位峰边界，实现积分面积定量。

- **输入**：`.mzML` 原始质谱数据
- **输出**：峰面积 CSV + 预测标注图
- **模型**：ResNet-50 骨干 + 1层Encoder + 1层Decoder（hidden_dim=256, nheads=8）
- **查询数**：num_queries=3（最多同时检出 3 个峰）
- **开发版本**：v2.8.13
- **当前开发范围**：仅 **Targeted × Centroided（MRM）** 模式；其余三组合（Targeted × Profile / Untargeted × Centroided / Untargeted × Profile）保留现状、暂不开发

---

## 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| Python | **3.11**（Conda 环境名固定为 `gamstekpeaking` |
| 包管理器 | Conda |
| R（可选） | 4.0+，仅 Untargeted 模式需要（⚠️ Untargeted 模式暂不开发，可不安装） |

**PyTorch 版本**（按 GPU 选择）：

| GPU 系列 | CUDA | torch | torch vision |
|----------|------|-------|-------------|
| RTX 50 (5060–5090) | 12.8 (cu128) | ≥2.7.0 | ≥0.22.0 |
| RTX 40 / 30 / 20 | 12.4 (cu124) | 2.6.0 | 0.21.0 |
| CPU / Apple Silicon (MPS) | — | 2.6.0 | 0.21.0 |

> 根目录 `requirements.txt` 已内置上述所有配置段，按需取消/注释对应行即可。当前默认启用 **RTX 40 系 (CUDA 12.4)**。

### 环境检测

使用以下方法之一进行环境的检测


```bash
# GUI 弹窗检测（含一键修复）
python .github/skills/check-dependencies/check_gui.py




# 纯终端文本报告（推荐）
python .github/skills/check-dependencies/check_env.py
```

### 环境安装/修复

安装分为两步：**① 安装 Python 环境 → ② 安装项目依赖**。

#### 第一步：安装 Python 环境

> 项目要求 **Python3.11** 推荐使用以下方式安装。

```bash
# 创建独立环境并指定 Python 版本（3.11）
conda create -n gamstekpeaking python=3.11
conda activate gamstekpeaking
```

#### 第二步：安装项目依赖

手动输入下面的代码进行环境依赖安装/修复：

```bash
pip install -r requirements.txt
```

> 💡 想一步完成「Python 环境 + 依赖」？也可直接使用 `environment.yml`（内置 Python 3.11 与全部依赖，等价于上面两步）：
>
> ```bash
> cd model
> conda env create -f environment.yml
> conda activate gamstekpeaking
> ```

**验证环境依赖**：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import pymzml; print('pymzml OK')"
```

> ⚠️ 推理前需确认模型权重文件必须存于 `model/` 目录下。

---

## 前处理（原始数据 → mzML）

仪器厂商导出的原始文件（`.msdata` / `.wiff` / `.wiff2`）需先转换为标准 `.mzML` 格式，转换工具位于 `converters/`：

| 脚本 | 输入 | 输出 | 工具链 |
|------|------|------|--------|
| `converters/msdata.py` | `.msdata` | `.mzML` | `msdata2mzml.exe`（OpenMS，运行时内置于 `msdata_bin/`） |
| `converters/wiff.py` | `.wiff` / `.wiff2` | `.mzML` | `msconvert.exe`（ProteoWizard，运行时内置于 `wiff_bin/`） |
| `converters/rename_cn.py` | — | — | 中文文件名 → 英文（`.msdata` 预处理） |

### 使用步骤

```bash
cd converters

# 0. （仅中文文件名需要）预览并重命名
python rename_cn.py                 # 预览映射
python rename_cn.py --no-dry-run    # 确认后执行

# 1. 将原始文件放入 converters/data/ 目录
# 2. 预览待转换文件（不执行转换）
python msdata.py --dry-run          # .msdata
python wiff.py --dry-run            # .wiff / .wiff2

# 3. 批量转换，输出自动生成于 data/<文件名>/ 子目录
python msdata.py                    # .msdata → .mzML
python wiff.py                      # .wiff → .mzML（默认带峰检测）
python wiff.py --no-peak-picking    # 保留 profile 原始轮廓
```

### 注意事项

- 项目路径不得含中文（OpenMS C++ 层限制）
- WIFF 文件需要同名 `.wiff.scan` 配套文件
- `msdata2mzml.exe` 仅接受位置参数，且即使成功也可能返回退码 858，脚本以「是否生成 .mzML 文件」判定成败
- 转换得到的 `.mzML` 可直接作为下方「推理」章节各模式的输入

---

## 推理

> 💡 四种分析模式（Targeted / Untargeted × Centroided / Profile）的原理、适用场景与操作流程，详见 [User_Tutorials.md](User_Tutorials.md)。
> ⚠️ **当前项目仅开发 Targeted × Centroided（MRM）模式**，其余三模式保留现状、暂不开发。

统一入口 `model/inference/cli.py`（在 `model/` 目录下用 `python -m inference.cli` 调用），通过 `--mode` 切换 3 种运行模式（`roi` / `pipeline` 均支持 `--mzml` 单文件或 `--batch_dir` 目录递归扫描，含子目录）：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `pipeline` | 完整管线：ROI 提取 → 预测 → SNR 筛选 → 精修（单文件或目录递归） | ⭐ 生产环境（推荐） |
| `roi` | 仅 EIC/ROI 提取（`--plot` 时附加预测画图，无预测 CSV） | 检查 XIC/ROI 质量 |
| `batch_dir` | 对已有 XIC/ROI 中间结果目录批量预测+积分 | 续跑 / 断点恢复 / ROI 复用 |

---

### 完整管线（推荐）

端到端：ROI 提取 → 模型预测 → SNR 筛选 → 峰区间精修。

```bash
cd model

# 批量 mzML（最常用；目录递归含子目录）
python -m inference.cli --mode pipeline \
  --model checkpoint/quanformer.pth \
  --batch_dir ../data/test1/mzML \
  --output_dir ../output/pipeline_batch \
  --threshold 0.99 --plot \
  --snr_min 3.0 \
  --pipeline_min_max_intensity 1000 \
  --pipeline_min_chrom_points 10

# 单个 mzML
python -m inference.cli --mode pipeline \
  --model checkpoint/quanformer.pth \
  --mzml ../data/test_oulu_23.mzML \
  --output_dir ../output/pipeline_single_test \
  --threshold 0.99 --plot
```

**输出结构**（以 `--output_dir ../output/pipeline_batch` 为例，`<样品>` 为 mzML 文件名去后缀）：

```
../output/pipeline_batch/
├── xic-roi-batch/                       # EIC 提取 + ROI 图像（每样品一个子目录）
├── batch_predictions/                   # 模型预测 CSV
│   └── <样品>/
│       ├── prediction.csv               # 模型预测结果（峰面积 + 置信度）
│       └── predicted_plots/             # 预测框标注图（--plot 时生成）
├── snr_filtered/                        # SNR 筛选 + 峰区间精修
│   └── <样品>/
│       └── SNR_box_3.0/                 # 目录名随 --snr_min 变化（3.0 → SNR_box_3.0）
│           ├── prediction.csv           # SNR 筛选后预测
│           ├── prediction_refined.csv   # ⭐ 最终精修结果（峰面积 + 置信度）
│           └── refined_plots/           # 精修标注图（--plot 时生成）
├── pipeline_timing.log                  # 阶段计时日志
└── pipeline_timing_runs.jsonl           # 计时记录（JSONL）
```

---

### 轻量模式

不需要完整管线（SNR 筛选和区间精修）时使用。
注意：`mzml` 仅做 EIC/ROI 提取（`--plot` 时附加预测画图，不输出预测 CSV）；需要预测结果请使用 `pipeline` 或 `batch_dir`。

```bash
# 单个 mzML（输出到 <output_dir>/<文件名>/）
python -m inference.cli --mode roi \
  --model checkpoint/quanformer.pth \
  --mzml ../data/test1/mzML/B1.mzML --output_dir results/roi

# 批量 mzML（目录递归含子目录）
python -m inference.cli --mode roi \
  --model checkpoint/quanformer.pth \
  --batch_dir ../data/test1/mzML --output_dir results/roi

# 对已有 ROI 目录批量预测+积分
python -m inference.cli --mode batch_dir \
  --model checkpoint/quanformer.pth \
  --batch_dir results/roi --output_dir results/pred
```

---

### 推理参数速查

> 以下为 `model/inference/cli.py` **全部**命令行参数（与 argparse 定义一一对应）。
> 「完整参数模板」可直接复制到终端，按注释填写/删减；除 `--model` 外所有参数均可省略（使用默认值）。

**完整参数模板**（注释即填写说明）：

```bash
python -m inference.cli \
  # ==================== 基础参数 ====================
  # 运行模式（默认 pipeline），可选：roi / batch_dir / pipeline
  --mode pipeline \
  # 【必填】模型权重 .pth 路径（相对 model/ 目录）
  --model checkpoint/quanformer.pth \
  # 置信度阈值（默认 0.99，建议 0.99 起步，过低会引入假峰）
  --threshold 0.99 \
  # 积分方式（默认 linear）：linear / raw / external_baseline
  --integration_method linear \
  # 高斯平滑 sigma（默认 0.0，越大峰越平滑但可能合并近邻峰）
  --smooth_sigma 0.0 \
  # 输出目录（默认自动生成）
  --output_dir ../output/pipeline_batch \
  # [roi / pipeline] 输入 mzML 文件路径，或包含 mzML 的目录（递归扫描）
  --mzml ../data/test1/mzML/B1.mzML \
  # [roi / pipeline] mzML 目录（递归扫描含子目录）；
  # [batch_dir] testXIC 输出目录
  --batch_dir ../data/test1/mzML \
  # 生成预测框标注图（flag，不加则不生成图）
  --plot \
  # ==================== Pipeline QC 参数 ====================
  # [已弃用] 标准品 CSV，传入仅打印提示，不再参与 ROI 定位
  # --standard_refs_csv xxx.csv \
  # [QC] XIC 平滑后最大强度低于此值 → 不生成 ROI（默认 1000；0=关闭）
  --pipeline_min_max_intensity 1000 \
  # [QC] 单条色谱 RT 点数少于此值 → 剔除（默认 10；0=关闭）
  --pipeline_min_chrom_points 10 \
  # ==================== SNR 筛选参数 ====================
  # 框外 SNR 最低阈值（默认 3.0，越高要求信噪比越严）
  --snr_min 3.0 \
  # SNR 计算时强度高斯平滑 sigma（默认 0.8）
  --snr_gaussian_sigma 0.8 \
  # 框外噪声至少点数（默认 5）
  --snr_min_noise_points 5 \
  # ==================== Post 精修参数 ====================
  # 精修输出 CSV 文件名（默认 prediction_refined.csv）
  --post_output_name prediction_refined.csv \
  # 小峰相对主峰的 RT 容差（默认 0.25 min）
  --post_small_peak_rt_tol 0.25 \
  # 次峰相对主峰动态最小比例（默认 0.04，略降有利于弱次峰通过）
  --post_min_secondary_ratio 0.04 \
  # 噪声阻碍系数（默认 0.45，略降有利于弱次峰通过）
  --post_noise_barrier_ratio 0.45 \
  # ROI 次峰全局门槛放宽系数（默认 0.055）
  --post_secondary_roi_global_gate_relax_frac 0.055 \
  # 峰顶单侧估计截停时的最大 RT 跨度 min（默认 0.24）
  --post_edge_max_span_min 0.24 \
  # 单侧低噪声分位数（默认 55；越高→截停阈值越高→边界外推越短）
  --post_edge_noise_percentile 55.0 \
  # 小峰边界外扩 padding（默认 0.08）
  --post_small_boundary_pad 0.08 \
  # 边界外推后验窗口点数（默认 0；0=仅首点阈值，外扩更少）
  --post_boundary_posterior_lookahead 0 \
  # 后验均值相对阈值倍数上限（默认 1.25，lookahead>0 时生效）
  --post_boundary_posterior_mean_scale 1.25 \
  # 关闭谷值回退（默认启用谷值回退；传入此 flag 才关闭）
  --post_disable_valley_fallback \
  # 小峰失败时关闭左右重预测（默认开启；传入此 flag 才关闭）
  --post_disable_lr_repredict_on_small_fail \
  # 精修后最低置信度（默认 0.99）
  --post_min_confidence 0.99 \
  # 精修后最低 SNR（默认 3.0）
  --post_min_snr 3.0 \
  # 小峰噪声窗口半宽（默认 0.30）
  --post_small_noise_window_half 0.30 \
  # 主峰边界噪声分位数（默认 20.0）
  --post_main_boundary_noise_percentile 20.0 \
  # 精修绘图平滑 sigma（默认 0.8）
  --post_plot_sigma 0.8 \
  # 精修绘图子目录名（默认 refined_plots）
  --post_plot_dir_name refined_plots \
  # 边框阈值模式（默认 roi_bottom_decile_mean）：
  #   roi_bottom_decile_mean / stable_tail_mean / low_percentile
  --post_edge_noise_stop_mode roi_bottom_decile_mean \
  # 三连微降早停（相对峰高，默认 0.010；0=关闭）
  --post_edge_flat_triplet_step_frac 0.010 \
  # 修正框宽上限：≤ 原始预测宽 × 倍数（默认 1.08，不强行扩框）
  --post_refine_width_max_expand_vs_pred 1.08 \
  # 修正框宽上限：≤ ROI 窗口 × 比例（默认 0.45）
  --post_refine_width_max_frac_of_roi 0.45 \
  # 启用小峰相对主峰的 RT 门控（默认关闭；传入此 flag 才启用）
  --post_enable_small_peak_rt_gate \
  # ==================== 输出控制 ====================
  # 不写 pipeline_timing.log / pipeline_timing_runs.jsonl（终端仍打印计时汇总）
  --no_timing \
  # SNR 筛选时生成 筛选保留/筛选剔除/ 红框标注 jpeg（默认关闭，省磁盘）
  --save_snr_jpeg
```

**常用参数速查**：

**通用参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | **必填** | 模型 `.pth` 路径 |
| `--mode` | `pipeline` | 运行模式：`roi` / `batch_dir` / `pipeline` |
| `--threshold` | `0.99` | 置信度阈值 |
| `--integration_method` | `linear` | `linear` / `raw` / `external_baseline` |
| `--smooth_sigma` | `0.0` | 高斯平滑 sigma |
| `--plot` | — | 生成预测框标注图 |
| `--output_dir` | 自动 | 输出目录 |

**Pipeline 参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--snr_min` | `3.0` | 框外 SNR 最低阈值 |
| `--pipeline_min_max_intensity` | `1000` | XIC 最大强度低于此值跳过 |
| `--pipeline_min_chrom_points` | `10` | 色谱点数少于此跳过 |
| `--post_min_confidence` | `0.99` | 精修后最低置信度 |
| `--no_timing` | — | 不写计时日志文件（终端仍打印） |
| `--save_snr_jpeg` | — | SNR 筛选生成红框标注图（默认关闭） |

---

### Untargeted 模式

> ⚠️ Untargeted 模式暂不开发。原 `getFeature.py`（R + CentWave 特征提取）已从仓库中删除，如需恢复请从 git history 找回。仅做 MRM（Targeted × Centroided）定量可跳过本节。

---

## 训练

### 生成 COCO 数据集

训练要求数据集为 COCO 格式，由 `preprocessing/coco_annotation.py` 从 mzML + 人工标注 xlsx 生成（EIC 图像与推理管线完全一致：400x300、apex±1min 窗口）：

```bash
cd model

python -m preprocessing.coco_annotation \
  --mzmls ../data/test/20260715_shiyaoyuan_test/20260715_shiyaoyuan_test_1.mzML \
          ../data/test/20260715_shiyaoyuan_test/20260715_shiyaoyuan_test_2.mzML \
  --labels ../data/test/testcase_data.xlsx \
  --output_dir ../data/coco \
  --val_stems 20260715_shiyaoyuan_test_2
```

标注 xlsx 沿用 `testcase_data.xlsx` 布局（`comonent`/`channel`/`peak_start`/`peak_end`/`sample_id` 列），按 `native_id`「化合物名-1/-2」与色谱对齐，`peak_start/peak_end`（分钟）经每张 ROI 的窗口线性映射为像素 bbox；无标注 ROI（TIC 等）作为负样本纳入。

### COCO 数据格式

`framework/datasets/coco.py` 按以下固定路径读取（`--coco_path` 指向数据集根目录）：

```
<coco_path>/
├── train/               # 训练 EIC 图像 (400x300 JPEG)
│   └── train_coco.json  # 训练标注（bbox + category_id=1 峰类）
├── val/                 # 验证图像
│   └── val_coco.json    # 验证标注
└── _xic/                # XIC 中间产物（可再生成，训练不读取）
```

### 训练命令

```bash
cd model

python -m train \
  --coco_path ../data/coco \
  --output_dir output \
  --device auto \
  --epochs 30 --batch_size 4 \
  --lr 1e-4 --lr_backbone 1e-5
```

### 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--coco_path` | `data/coco` | 数据集根目录 |
| `--device` | `auto` | CUDA > MPS > CPU 自动选择 |
| `--epochs` | `30` | 训练轮数 |
| `--batch_size` | `4` | 批大小 |
| `--lr` / `--lr_backbone` | `1e-4` / `1e-5` | 学习率 |
| `--enc_layers` / `--dec_layers` | `1` / `1` | Transformer 编/解码器层数 |
| `--num_queries` | `10`（checkpoint 为 `3`） | 最大检出峰数 |
| `--resume` | — | 从检查点恢复 |
| `--eval` | — | 仅评估，不训练 |

### 微调与评估

```bash
# 从预训练权重继续训练
python -m train \
  --coco_path ../data/coco --output_dir output_finetune \
  --device auto --resume checkpoint/quanformer.pth --epochs 50

# 仅评估
python -m train \
  --coco_path ../data/coco \
  --resume checkpoint/quanformer.pth --device auto --eval
```

> ⚠️ 恢复训练时会自动跳过 `class_embed` 和 `query_embed` 权重（维度可能不匹配）。
> 💡 当前 `quanformer.pth` 的训练参数：`enc_layers=1, dec_layers=1, num_queries=3, hidden_dim=256, nheads=8, dim_feedforward=2048, dropout=0.1`。

---

## 项目结构

```
MRMPFormer/
├── requirements.txt              # pip 依赖（model + desktop 合并，GPU 分段）
├── model/                        # ⭐ 核心代码（须在此目录下用 python -m <pkg> 调用）
│   ├── train.py                  # 训练入口（python -m train ...）
│   ├── environment.yml           # Conda 环境（name: gamstekpeaking）
│   ├── inference/                # 推理：CLI 入口 + 预测器 + 两轮检测
│   │   └── cli.py                #   统一推理入口（python -m inference.cli --mode ...）
│   ├── models/                   # 模型定义（quanformer / mrmpformer / shared）
│   ├── preprocessing/            # 前处理：xic_extraction / ion_zenith / masked_roi_generator
│   ├── postprocessing/           # 后处理：peak_refinement / snr_filter / valley_split / evaluation
│   ├── framework/                # DETR 训练框架（datasets / util / engine）
│   ├── utils/                    # 推理辅助（io / quantify / mzml_load / roi_rt_mapping 等）
│   ├── tools/                    # 批处理 / 诊断 / 可视化 / 基准测试
│   ├── checkpoint/               # 模型权重
│   └── resources/                # 示例数据
├── desktop/                      # PySide6 桌面 GUI（前处理 / 寻峰 / 定量 / 设置）
├── converters/                   # 格式转换（msdata/wiff → mzML）
├── data/                         # 测试数据
└── docs/                         # 项目文档
```

---

## 辅助工具

### converters — 格式转换

支持两种厂商格式 → 标准 `.mzML`：

```bash
cd converters

# msdata → mzML（OpenMS 工具链）
python rename_cn.py      # 中文文件名→英文（首次）
python msdata.py         # 批量转换

# wiff → mzML（ProteoWizard 工具链）
python wiff.py           # 批量转换
```

> 要求对应 `*_bin/` 包含运行时依赖，项目路径不得含中文。

### 工具集（`model/tools/`）

| 子模块 | 用途 | 入口示例 |
|--------|------|----------|
| `batch/` | 批量重处理（SNR / post） | `python -m tools.batch.reprocess --stage snr` |
| `mzml/` | 色谱图查看/导出 | `python -m tools.mzml.chromatogram list <file>` |
| `benchmark/` | 性能基准测试 | `python -m tools.benchmark.runner --help` |
| `visualization/` | 可视化 | — |
| `tests/` | 测试脚本 | — |

---

## 平台注意事项

| 平台 | 注意事项 |
|------|----------|
| **Windows** | 路径避免空格（影响 R 调用）；`pycocotools` 可能需 Visual C++ Build Tools |
| **Linux** | 无桌面环境运行 GUI 需 X11 转发（`ssh -X`）或 `xvfb` |
| **macOS** | Apple Silicon 自动使用 MPS 加速；Intel Mac 仅 CPU |

---

## 常见问题

<details>
<summary><b>CUDA 不可用？</b></summary>

```bash
nvidia-smi                          # 确认驱动正常
python -c "import torch; print(torch.cuda.is_available())"
```
若返回 `False`，重装对应 GPU 的 PyTorch 版本。
</details>

<details>
<summary><b>无 GPU 可以运行吗？</b></summary>

可以。编辑根目录 `requirements.txt`，启用 `CPU Only` 段（`--index-url .../cpu`），注释其他 GPU 段后重装。代码自动回退 CPU。
</details>

<details>
<summary><b>Untargeted 模式报错 FileNotFoundError: xcms_peak_list.csv？</b></summary>

R 或 Bioconductor 包未正确安装，请按照上方「Untargeted 模式」节重新安装 R 环境。
</details>

<details>
<summary><b>Windows 路径相关错误？</b></summary>

避免路径含**空格**和**中文**。推荐 `D:\data\mrmpformer\` 之类简洁路径。
</details>

---

> 更多细节：[项目全景](docs/PROJECT_PANORAMA.md) · [已知问题](docs/Bugs.md) · [跨平台部署](docs/MRMPFormer%20跨平台部署指南.md)
