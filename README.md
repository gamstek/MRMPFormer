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

统一入口 `model/main.py`，通过 `--mode` 切换 7 种运行模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `pipeline_batch_mzml` | 完整管线：批量 mzML | ⭐ 生产环境（推荐） |
| `pipeline_mzml` | 完整管线：单个 mzML | 单样品端到端测试 |
| `single` | 单张图 JSON 输入/输出 | 调试 / API 集成 |
| `mzml` | 单个 mzML：仅 EIC/ROI 提取（`--plot` 时附加预测画图，无预测 CSV） | 检查 XIC/ROI 质量 |
| `batch_mzml` | 批量 mzML：仅 EIC/ROI 提取（`--plot` 时附加预测画图，无预测 CSV） | 批量检查 XIC/ROI 质量 |
| `batch_dir` | 已有 XIC 中间结果的目录 | 续跑 / 断点恢复 |
| `batch_json_dir` | 目录下所有 JSON 逐张处理 | JSON 数据集批处理 |

---

### 完整管线（推荐）

端到端：ROI 提取 → 模型预测 → SNR 筛选 → 峰区间精修。

```bash
cd model

# 批量 mzML（最常用）
python main.py --mode pipeline_batch_mzml \
  --model checkpoint/quanformer.pth \
  --batch_dir ../data/test1/mzML \
  --output_dir ../output/pipeline_batch \
  --threshold 0.99 --plot \
  --snr_min 3.0 \
  --pipeline_min_max_intensity 1000 \
  --pipeline_min_chrom_points 10

# 单个 mzML
python main.py --mode pipeline_mzml \
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
注意：`mzml` / `batch_mzml` 仅做 EIC/ROI 提取（`--plot` 时附加预测画图，不输出预测 CSV）；需要预测结果请使用 `pipeline_*` 或 `batch_dir`。

```bash
# 单张图（JSON，适合调试/API）
python main.py --mode single \
  --model checkpoint/quanformer.pth \
  --input input.json --threshold 0.99 --plot

# stdin 管道
echo '{"rt":[1,2,3,4,5],"intensity":[100,500,800,400,50]}' | \
  python main.py --mode single --model checkpoint/quanformer.pth

# 单个 mzML
python main.py --mode mzml \
  --model checkpoint/quanformer.pth \
  --mzml ../data/test1/mzML/B1.mzML --output_dir results/single

# 批量 mzML
python main.py --mode batch_mzml \
  --model checkpoint/quanformer.pth \
  --batch_dir ../data/test1/mzML --output_dir results/batch
```

**单张图 JSON 格式**：

输入：
```json
{"rt": [1.0, 2.0, 3.0], "intensity": [100, 500, 200], "baseline_x": [], "baseline_y": []}
```

输出：
```json
{"detections": [{"x1":120, "x2":180, "score":0.998, "area":12345.6, "rt_min":2.1, "rt_max":3.4}]}
```

---

### 推理参数速查

> 以下为 `model/main.py` **全部**命令行参数（与 argparse 定义一一对应）。
> 「完整参数模板」可直接复制到终端，按注释填写/删减；除 `--model` 外所有参数均可省略（使用默认值）。

**完整参数模板**（注释即填写说明）：

```bash
python main.py \
  # ==================== 基础参数 ====================
  # 运行模式（默认 single），可选：
  #   single / mzml / batch_mzml / batch_dir / batch_json_dir / pipeline_mzml / pipeline_batch_mzml
  --mode pipeline_batch_mzml \
  # 【必填】模型权重 .pth 路径（相对 model/ 目录）
  --model checkpoint/quanformer.pth \
  # [single] 输入 JSON 文件路径，- 表示 stdin（默认 -）
  --input input.json \
  # [single] 输出 JSON 文件路径，- 表示 stdout（默认 -）
  --output output.json \
  # 置信度阈值（默认 0.99，建议 0.99 起步，过低会引入假峰）
  --threshold 0.99 \
  # 积分方式（默认 linear）：linear / raw / external_baseline
  --integration_method linear \
  # 高斯平滑 sigma（默认 0.0，越大峰越平滑但可能合并近邻峰）
  --smooth_sigma 0.0 \
  # 输出目录（默认自动生成）
  --output_dir ../output/pipeline_batch \
  # [single] 保留临时文件（flag，不加则不保留）
  --keep_temp \
  # [mzml / pipeline_mzml] 输入 mzML 文件路径
  --mzml ../data/test1/mzML/B1.mzML \
  # [batch_mzml / pipeline_batch_mzml] mzML 目录；
  # [batch_dir] testXIC 输出目录；[batch_json_dir] JSON 目录
  --batch_dir ../data/test1/mzML \
  # 生成预测框标注图（flag，不加则不生成图）
  --plot \
  # [batch_json_dir] 所有预测图统一目录（默认 <batch_dir>/predicted_plots_all）
  --batch_plot_dir ../output/plots_all \
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
  --post_enable_small_peak_rt_gate
```

**常用参数速查**：

**通用参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | **必填** | 模型 `.pth` 路径 |
| `--mode` | `single` | 运行模式 |
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

---

### Untargeted 模式（R + CentWave）

> ⚠️ Untargeted 模式暂不开发（保留现状）；仅做 MRM（Targeted × Centroided）定量可跳过本节。

#### 安装 R 环境

```bash
# 检查
R --version

# Ubuntu/Debian
sudo apt install --no-install-recommends r-base libxml2-dev
```

在 R 控制台安装 Bioconductor 包：

```r
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("xcms")
BiocManager::install("MSnbase")
install.packages("dplyr")
```

#### 运行特征提取

```bash
cd model

python getFeature.py \
  --source resources/example/centroided \
  --polarity positive --ppm 10 \
  --minWidth 5 --maxWidth 50 \
  --s2n 5 --noise 100 \
  --mzDiff 0.015 --prefilter 3
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--source` | `resources/example` | mzML 数据目录 |
| `--polarity` | `positive` | 极性 |
| `--ppm` | `10` | MS1 ppm 容差 |
| `--minWidth` / `--maxWidth` | `5` / `50` | 峰宽范围 |
| `--s2n` | `5` | 信噪比阈值 |
| `--mzDiff` | `0.015` | m/z 差异阈值 |

---

## 训练

### 训练命令

```bash
cd model

python mrmpformer/main.py \
  --coco_path data/peak-all \
  --output_dir output \
  --device auto \
  --epochs 30 --batch_size 4 \
  --lr 1e-4 --lr_backbone 1e-5
```

### COCO 数据格式

```
<coco_path>/
├── train2017/                    # EIC 图像 (JPEG)
├── val2017/                      # 验证图像
└── annotations/
    ├── instances_train2017.json  # 标注（bbox + category_id）
    └── instances_val2017.json
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
python mrmpformer/main.py \
  --coco_path data/peak-all --output_dir output_finetune \
  --device auto --resume checkpoint/quanformer.pth --epochs 50

# 仅评估
python mrmpformer/main.py \
  --coco_path data/peak-all \
  --resume checkpoint/quanformer.pth --device auto --eval
```

> ⚠️ 恢复训练时会自动跳过 `class_embed` 和 `query_embed` 权重（维度可能不匹配）。
> 💡 当前 `quanformer.pth` 的训练参数：`enc_layers=1, dec_layers=1, num_queries=3, hidden_dim=256, nheads=8, dim_feedforward=2048, dropout=0.1`。

---

## 项目结构

```
MRMPFormer/
├── requirements.txt              # pip 依赖（model + desktop 合并，GPU 分段）
├── model/                        # ⭐ 核心代码
│   ├── main.py                   # 推理入口（--mode 驱动）
│   ├── getFeature.py             # Untargeted 特征提取
│   ├── environment.yml           # Conda 环境（name: gamstekpeaking）
│   ├── mrmpformer/               # 核心包（模型/训练/推理/前后处理/管线）
│   │   ├── main.py               #   训练入口
│   │   ├── models/               #   模型定义
│   │   └── datasets/             #   COCO 数据加载
│   ├── utils/                    # 推理辅助（EIC/定量/绑图）
│   ├── tools/                    # 批处理/诊断/可视化
│   ├── checkpoint/               # 模型权重
│   └── resources/                # 示例数据
├── converters/                   # 格式转换（msdata/wiff → mzML）
├── gamstekpeaking/               # Streamlit Web 工作台（已废弃，见 desktop/）
├── data/                         # 测试数据
├── docs/                         # 项目文档
└── paper/                        # 论文
```

---

## 辅助工具

### GamSTekPeaking — Web 工作台

基于 Streamlit 的代谢组学全流程界面：

```bash
cd gamstekpeaking
pip install -r requirements.txt
python main.py
```

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
