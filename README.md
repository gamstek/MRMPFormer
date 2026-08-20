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

## 工作流总览

本项目核心工作流为**三段式闭环：训练 → 推理 → 评估**（此前置一环"构建数据集"）。

| 环节 | 做什么 | 入口 | 产物 |
|---|---|---|---|
| ① 构建数据集 | 标注 xlsx + mzML → COCO 格式 bbox 标注（bbox 直接映射人工 `peak_start/peak_end`） | `model/preprocessing/coco_annotation.py` | `data/coco/`（train/val + `train_coco.json`/`val_coco.json`） |
| ② 训练 | 用 COCO 数据集训练/微调 DETR 模型 | `python -m train --config configs/quanformer_baseline.json`（从零）/ `quanformer_v2_finetune.json`（微调） | `checkpoint/quanformer.pth`（基线）、`quanformerv2.pth`（微调） |
| ③ 推理 | 用测试 mzML + 已训练模型产出预测值 | `python -m inference.cli --mode pipeline --model checkpoint/quanformer.pth` | `batch_predictions/<样品>/prediction.csv` → `prediction_snr.csv` → `prediction_refined.csv` |
| ④ 评估 | 用预测值 + 人工标注计算模型效果 | `python -m tools.evaluation.evaluate_baseline --labels ../data/label/20260715_shiyaoyuan_test.xlsx` | `evaluation_report.json`（P/R/F1、面积 R²、RT 偏差、RSD）、`match_details.csv`、`area_pairs.csv` |

**要点**：

- **标注数据集双重身份**：`data/label/20260715_shiyaoyuan_test.xlsx` 既是训练标注来源（→ COCO bbox），也是评估 GT——两环节共用一份标注是预期设计；
- **评估口径**：评估读阶段②**原始 `prediction.csv`**（衡量模型原始检测能力），**不含** SNR 筛选/精修后处理；若需评估整条管线最终产物，目标应改为 `prediction_refined.csv`（另一套口径）；
- **数据泄漏防护**：训练/评估分样品（train=`test_1`、val=`test_2`），不在同一样品上既训又评；
- **快速路径**：训练、推理、评估的具体命令与参数见下文「训练」「推理」两节。

---

## 数据目录规范

`data/` 按「实验隔离 + 数据类型五分法」组织；正式实验数据放 `data/<实验名>/`，测试数据放 `data/test/`，**两者内部结构一致**：

```
data/
├── coco/     # 对应实验构造好的 COCO 训练数据集（train/ + train_coco.json、val/ + val_coco.json；_xic/ 为构建时的 XIC 中间产物）
├── label/    # 各实验的人工标注数据（统一存 data/label/<trial>.xlsx；布局：化合物/通道/rt/peak_label/多峰起止/面积）
├── mzml/     # 原始数据转换得到的 mzML 文件
├── msdata/   # 原始 msdata 文件
└── wiff/     # 原始 wiff 文件
```

**数据流转关系**（五类目录对应工作流的原料与产物）：

```
wiff/ + msdata/ ──(converters 格式转换)──> mzml/ ──(coco_annotation + label/)──> coco/
```

**当前状态**：仅 `data/test/` 存放数据（20260715 试药园两次进样实验）；`data/` 顶层五个目录暂为空，新实验加入时按同结构在其下创建 `data/<实验名>/`。

**注意**：
- 推理/评测等中间产物**不写入** `data/`，统一输出到 `../output/`（见上文输出目录约定）；
- `data/` 整体在 `.gitignore` 中，不入版本库。

---

## 质量控制（QC）

数据质量问题是定量实验失败的常见根因：**错误标注**会污染训练 bbox（模型学到错边界）、误导评估 GT（模型好坏被误判）；**异常通道**（低强度、少点数、低信噪比）会产生假阳性检测。为此管线设置了多层 QC 防线。

### QC 防线总览

| # | 防线 | 位置 | 检查内容 | 参数（默认） | 结果去向 |
|---|---|---|---|---|---|
| 1 | **标注 RT 一致性**（新） | `label_qc` → 训练数据构建 & 推理管线 | ①跨样品：同化合物同通道在各样品间 RT 极差；②双离子：同一样品中定量/定性离子 RT 极差。**极差 >1 min 判疑似实验有误**：警示人工复核 + 涉事行剔除（不生成 ROI / 不进训练 bbox） | `--qc_label_rt_tol`(1.0) | `output/QC/<run>/qc_label_rt.csv` |
| 2 | ROI 通道级 | `preprocessing/xic_extraction.py` | 平滑后整条 XIC 最大强度过低、RT 点数过少 → 不生成 ROI | `--pipeline_min_max_intensity`(1000)、`--pipeline_min_chrom_points`(10) | 各样品 `pipeline_qc_excluded.csv` → 汇总 `qc_roi_channels.csv` |
| 3 | 预测框级 | `inference/predictor.py` | score < 阈值的检测框不输出；feature 无化合物则跳过 | `--threshold`(0.99) | `qc_prediction_threshold.csv` |
| 4 | SNR 框级 | `postprocessing/snr_filter.py` | 框外 SNR、框外噪声点数联合判定 | `--snr_min`(3.0)、`--snr_min_noise_points`(5) | `box_outside_snr_report.csv` → 汇总 `qc_snr_boxes.csv` |
| 5 | 精修框级 | `postprocessing/peak_refinement.py` | 精修置信度、SNR、次峰比例、框宽上限等门控 | `--post_min_confidence`(0.99)、`--post_min_snr`(3.0) 等 | `qc_post_refinement.csv` |

### 标注 RT 一致性检查（防线 1 详述）

两类检查的物理依据：同一色谱方法下同化合物 RT 高度稳定（连续进样漂移通常 <0.1 min）；定量/定性离子必然共流出。**极差 >1 min** 说明某样品标注画错、通道张冠李戴或仪器 RT 异常漂移。

剔除粒度：

| 场景 | 策略 |
|---|---|
| 跨样品、组内样品数 ≥3 | 仅剔偏离组中位数 >1 min 的样品（多数派可信） |
| 跨样品、组内样品数 =2 | 两行都剔（无法仲裁谁错）+ 警示人工复核 |
| 双离子 | 两通道都剔（无法判断定量/定性谁错） |

推理侧通过可选参数 `--labels` 启用（不传则此检查跳过，保持"推理无需标注"的契约）；训练数据构建（`coco_annotation`）默认启用。

### 统一 QC 输出

所有环节的 QC 结果表统一写入 **`../output/QC/<run_name>/`**（run_name 为本次运行输出目录名，训练侧为 `coco_<实验名>_<时间戳>`）：

```
output/QC/<run_name>/
├── qc_label_rt.csv            # 标注 RT 一致性（含保留行，便于复核）
├── qc_roi_channels.csv        # ROI 通道级剔除汇总（含 reason）
├── qc_prediction_threshold.csv# 预测阈值剔除统计
├── qc_snr_boxes.csv           # SNR 逐框明细汇总
├── qc_post_refinement.csv     # 精修门控剔除明细
└── qc_summary.md              # 各环节检查数/剔除数/人工复核清单
```

> **当前状态**：五道防线与统一 QC 输出目录已全部实施运行（`docs/plan_qc.md` P1/P2/P3 完成）。2026-08-20 防线 1 在 20260715 实验标注中查出 20 项需人工复核（跨样品/双离子 RT 极差 12.7~25.9 min）。

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


```powershell
# GUI 弹窗检测（含一键修复）
python .github/skills/check-dependencies/check_gui.py




# 纯终端文本报告（推荐）
python .github/skills/check-dependencies/check_env.py
```

### 环境安装/修复

安装分为两步：**① 安装 Python 环境 → ② 安装项目依赖**。

#### 第一步：安装 Python 环境

> 项目要求 **Python3.11** 推荐使用以下方式安装。

```powershell
# 创建独立环境并指定 Python 版本（3.11）
conda create -n gamstekpeaking python=3.11
conda activate gamstekpeaking
```

#### 第二步：安装项目依赖

手动输入下面的代码进行环境依赖安装/修复：

```powershell
pip install -r requirements.txt
```

> 💡 想一步完成「Python 环境 + 依赖」？也可直接使用根目录的 `environment.yml`（内置 Python 3.11 与全部依赖，等价于上面两步）：
>
> ```powershell
> conda env create -f environment.yml
> conda activate gamstekpeaking
> ```

**验证环境依赖**：

```powershell
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

```powershell
cd converters

# 0. （仅中文文件名需要）预览并重命名
python rename_cn.py                 # 预览映射
python rename_cn.py --no-dry-run    # 确认后执行

# 1. 将原始文件放入项目根目录 data/ 下（读取路径：data/msdata、data/wiff）
# 2. 预览待转换文件（不执行转换）
python msdata.py --dry-run          # .msdata
python wiff.py --dry-run            # .wiff / .wiff2

# 3. 批量转换，输出统一生成于 data/mzml/<文件名>/ 子目录
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

## 桌面端 GUI（GAMSTEKPEAKing）

除命令行外，项目提供基于 **PySide6** 的桌面图形界面（`desktop/`），目前「前处理」板块（格式转换 + 离子天顶）已上线，其余板块（寻峰 / 定量 / 模型 / 设置）为灰色占位状态。

### 启动应用

```powershell
cd desktop
python main.py          # 正常启动
python main.py --debug  # 调试模式（额外日志输出到 stdout）
```

> 环境要求：`gamstekpeaking` conda 环境（依赖见根目录 `requirements.txt`，已含 PySide6）。未捕获异常会写入 `desktop/error.log` 并弹窗提示。

### 界面布局

- **左侧边栏**：前处理（可用）/ 寻峰 / 定量 / 模型 / 设置（灰色为未上线占位）
- **右侧内容区**：当前板块的功能卡片，耗时任务由后台线程（`desktop/workers/`）执行，进度实时更新

### 功能卡片 1：格式转换（msdata → mzML）

1. 点击「📂」选择或直接**拖拽** `.msdata` 文件到虚线框内（可多选，自动去重）
2. 选择输出目录：**默认**（与输入文件同目录）或**自定义**（选择目录后存入）
3. 点击「▶ 开始转换」，底部进度条实时显示进度
4. 每个文件行显示状态：⏳ 等待 / 🔄 转换中 / ✅ 成功（含产物大小）/ ❌ 失败（悬停查看原因）

### 功能卡片 2：离子天顶（MS1 → CSV）

遍历 mzML 的 MS1 谱图，提取每个 m/z 信号的顶点（最高强度），输出 `(m/z, RT, intensity, n_observations)` 的 CSV。

1. 选择**输入 mzML** 文件（自动建议同目录输出 CSV 路径，可修改）
2. （可选）展开「▸ 高级参数」调整参数；参数不合法时「▶ 开始运行」自动禁用：

| 参数 | 默认 | 说明 |
|------|------|------|
| m/z 范围 | 50 – 2000 Da | 仅处理该质量范围内的离子 |
| 容差 (ppm) | 10.0 | 质荷比相对容差，用于离子聚合 |
| 容差 (Da) | 0.01 | 质荷比绝对容差，与 ppm 同时生效 |
| 强度下限 / 上限 | 无 | 过滤低/高于该强度的信号，0=不过滤 |
| 最大谱图数 | 0（全部） | 限制处理的谱图数 |
| 重建 mzML 索引 | 关 | mzML 缺少索引时自动重建 |

3. 点击「▶ 开始运行」，实时显示已扫描谱图数与聚合峰数
4. 完成后显示离子总数与耗时，可点击「📂 打开所在目录」定位产物 CSV

---

## 推理

> 💡 四种分析模式（Targeted / Untargeted × Centroided / Profile）的原理、适用场景与操作流程，详见 [User_Tutorials.md](User_Tutorials.md)。
> ⚠️ **当前项目仅开发 Targeted × Centroided（MRM）模式**，其余三模式保留现状、暂不开发。

统一入口 `model/inference/cli.py`（在 `model/` 目录下用 `python -m inference.cli` 调用），通过 `--mode` 切换 3 种运行模式（`roi` / `pipeline` 均支持 `--mzml` 单文件或 `--batch_dir` 目录递归扫描，含子目录）：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `pipeline` | 完整管线：ROI 提取 → 预测 → SNR 筛选 → 精修（单文件或目录递归） | ⭐ 生产环境（推荐） |
| `roi` | 仅 EIC/ROI 提取（无需 `--model`，无预测 CSV） | 检查 XIC/ROI 质量 |
| `batch_dir` | 对已有 XIC/ROI 中间结果目录批量预测+积分 | 续跑 / 断点恢复 / ROI 复用 |

---

### 完整管线（推荐）

端到端：ROI 提取 → 模型预测 → SNR 筛选 → 峰区间精修。

```powershell
cd model

# 批量 mzML（最常用；目录递归含子目录）
python -m inference.cli --mode pipeline `
  --model checkpoint/quanformer.pth `
  --batch_dir ../data/test/mzml `
  --output_dir ../output/pipeline_batch `
  --threshold 0.99 --plot `
  --snr_min 3.0 `
  --pipeline_min_max_intensity 1000 `
  --pipeline_min_chrom_points 10

# 单个 mzML
python -m inference.cli --mode pipeline `
  --model checkpoint/quanformer.pth `
  --mzml ../data/test/mzml/20260715_shiyaoyuan_test_1.mzML `
  --output_dir ../output/pipeline_single_test `
  --threshold 0.99 --plot
```

> **ROI 生成方式（B 范式）**：默认以各通道谱图**最高强度点**居中；传 `--labels` 时改为**标注驱动**——仅标注命中通道生成 ROI，窗口中心 = 标注 `rt` 字段，并由防线 1（标注 RT 一致性）把关（剔除涉事通道）。不传 `--labels` 时推理无需标注（契约不变）。

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
│           ├── prediction_snr.csv       # SNR 筛选后预测（旧版为 prediction.csv）
│           ├── feature.csv              # SNR 通过行的紧凑子集（全量在 xic-roi-batch/<样品>/）
│           ├── roi_windows.csv          # 同上（筛选后子集；目录自包含，便于单独运行 post）
│           ├── xic_matrix.npy           # 同上
│           ├── snr_kept/                # 保留行红框标注图（--save_snr_jpeg 时生成；旧版目录名为 筛选保留/）
│           ├── snr_dropped/             # 剔除行红框标注图（同上；旧版为 筛选剔除/）
│           ├── prediction_refined.csv   # ⭐ 最终精修结果（峰面积 + 置信度）
│           └── refined_plots/           # 精修标注图（--plot 时生成）
├── pipeline_timing.log                  # 阶段计时日志
└── pipeline_timing_runs.jsonl           # 计时记录（JSONL）
```

---

### 轻量模式

不需要完整管线（SNR 筛选和区间精修）时使用。
注意：`roi` 仅做 EIC/ROI 提取（不输出预测 CSV，也无需 `--model`）；想看预测框标注请两步走：先 `roi` 生成 ROI，再 `batch_dir --plot` 画图；需要完整预测结果请使用 `pipeline`。

```powershell
# 单个 mzML（输出到 <output_dir>/<文件名>/；无需 --model）
python -m inference.cli --mode roi `
  --mzml ../data/test/mzml/20260715_shiyaoyuan_test_1.mzML --output_dir ../output/test/roi_check

# 批量 mzML（目录递归含子目录）
python -m inference.cli --mode roi `
  --batch_dir ../data/test/mzml --output_dir ../output/test/roi_check

# 对已有 ROI 目录批量预测+积分（--plot 时同时生成预测框标注图）
python -m inference.cli --mode batch_dir `
  --model checkpoint/quanformer.pth `
  --batch_dir ../output/test/roi_check --output_dir ../output/test/pred_check --plot
```

> **输出目录约定**：所有推理输出统一写到 `../output/`（相对 `model/` 目录）。各模式 `--output_dir` 默认值：
>
> | 模式 | 默认 `--output_dir` | 说明 |
> |------|---------------------|------|
> | `roi` | `../output/inference/xic-roi-batch` | EIC/ROI 提取产物，可供 `batch_dir` 模式复用 |
> | `batch_dir` | `../output/inference/batch_predictions` | 对已有 ROI 目录的批量预测 |
> | `pipeline` | `../output/inference/full_pipeline` | 完整管线（ROI + 预测 + SNR + 精修） |
>
> **测试/试跑请显式指定 `../output/test/<名称>`** 单独存放，不与正式产物混放。

---

### 推理参数速查

> 以下为 `model/inference/cli.py` **全部**命令行参数（与 argparse 定义一一对应）。
> 「完整参数模板」可直接复制到终端，按注释填写/删减；除 `--model` 外所有参数均可省略（使用默认值）。

**完整参数模板**（注释即填写说明）：

```powershell
python -m inference.cli `
  # ==================== 基础参数 ====================
  # 运行模式（默认 pipeline），可选：roi / batch_dir / pipeline
  --mode pipeline `
  # 【必填】模型权重 .pth 路径（相对 model/ 目录；roi 模式非必填）
  --model checkpoint/quanformer.pth `
  # 置信度阈值（默认 0.99，建议 0.99 起步，过低会引入假峰）
  --threshold 0.99 `
  # 积分方式（默认 linear）：linear / raw / external_baseline
  --integration_method linear `
  # 高斯平滑 sigma（默认 0.0，越大峰越平滑但可能合并近邻峰）
  --smooth_sigma 0.0 `
  # 输出目录（默认自动生成）
  --output_dir ../output/pipeline_batch `
  # [roi / pipeline] 输入 mzML 文件路径，或包含 mzML 的目录（递归扫描）
  --mzml ../data/test/mzml/20260715_shiyaoyuan_test_1.mzML `
  # [roi / pipeline] mzML 目录（递归扫描含子目录）；
  # [batch_dir] testXIC 输出目录
  --batch_dir ../data/test/mzml `
  # [pipeline/batch_dir] 生成预测框标注图（roi 模式不支持）
  --plot `
  # ==================== Pipeline QC 参数 ====================
  # [QC] XIC 平滑后最大强度低于此值 → 不生成 ROI（默认 1000；0=关闭）
  --pipeline_min_max_intensity 1000 `
  # [QC] 单条色谱 RT 点数少于此值 → 剔除（默认 10；0=关闭）
  --pipeline_min_chrom_points 10 `
  # ==================== SNR 筛选参数 ====================
  # 框外 SNR 最低阈值（默认 3.0，越高要求信噪比越严）
  --snr_min 3.0 `
  # SNR 计算时强度高斯平滑 sigma（默认 0.8）
  --snr_gaussian_sigma 0.8 `
  # 框外噪声至少点数（默认 5）
  --snr_min_noise_points 5 `
  # ==================== Post 精修参数 ====================
  # 精修输出 CSV 文件名（默认 prediction_refined.csv）
  --post_output_name prediction_refined.csv `
  # 小峰相对主峰的 RT 容差（默认 0.25 min）
  --post_small_peak_rt_tol 0.25 `
  # 次峰相对主峰动态最小比例（默认 0.04，略降有利于弱次峰通过）
  --post_min_secondary_ratio 0.04 `
  # 噪声阻碍系数（默认 0.45，略降有利于弱次峰通过）
  --post_noise_barrier_ratio 0.45 `
  # ROI 次峰全局门槛放宽系数（默认 0.055）
  --post_secondary_roi_global_gate_relax_frac 0.055 `
  # 峰顶单侧估计截停时的最大 RT 跨度 min（默认 0.24）
  --post_edge_max_span_min 0.24 `
  # 单侧低噪声分位数（默认 55；越高→截停阈值越高→边界外推越短）
  --post_edge_noise_percentile 55.0 `
  # 小峰边界外扩 padding（默认 0.08）
  --post_small_boundary_pad 0.08 `
  # 边界外推后验窗口点数（默认 0；0=仅首点阈值，外扩更少）
  --post_boundary_posterior_lookahead 0 `
  # 后验均值相对阈值倍数上限（默认 1.25，lookahead>0 时生效）
  --post_boundary_posterior_mean_scale 1.25 `
  # 关闭谷值回退（默认启用谷值回退；传入此 flag 才关闭）
  --post_disable_valley_fallback `
  # 小峰失败时关闭左右重预测（默认开启；传入此 flag 才关闭）
  --post_disable_lr_repredict_on_small_fail `
  # 精修后最低置信度（默认 0.99）
  --post_min_confidence 0.99 `
  # 精修后最低 SNR（默认 3.0）
  --post_min_snr 3.0 `
  # 小峰噪声窗口半宽（默认 0.30）
  --post_small_noise_window_half 0.30 `
  # 主峰边界噪声分位数（默认 20.0）
  --post_main_boundary_noise_percentile 20.0 `
  # 精修绘图平滑 sigma（默认 0.8）
  --post_plot_sigma 0.8 `
  # 精修绘图子目录名（默认 refined_plots）
  --post_plot_dir_name refined_plots `
  # 边框阈值模式（默认 roi_bottom_decile_mean）：
  #   roi_bottom_decile_mean / stable_tail_mean / low_percentile
  --post_edge_noise_stop_mode roi_bottom_decile_mean `
  # 三连微降早停（相对峰高，默认 0.010；0=关闭）
  --post_edge_flat_triplet_step_frac 0.010 `
  # 修正框宽上限：≤ 原始预测宽 × 倍数（默认 1.08，不强行扩框）
  --post_refine_width_max_expand_vs_pred 1.08 `
  # 修正框宽上限：≤ ROI 窗口 × 比例（默认 0.45）
  --post_refine_width_max_frac_of_roi 0.45 `
  # 启用小峰相对主峰的 RT 门控（默认关闭；传入此 flag 才启用）
  --post_enable_small_peak_rt_gate `
  # ==================== 输出控制 ====================
  # 不写 pipeline_timing.log / pipeline_timing_runs.jsonl（终端仍打印计时汇总）
  --no_timing `
  # SNR 筛选时生成 snr_kept/snr_dropped/ 红框标注 jpeg（默认关闭，省磁盘）
  --save_snr_jpeg
```

**常用参数速查**：

**通用参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | 必填（roi 除外） | 模型 `.pth` 路径 |
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

训练要求数据集为 COCO 格式，由 `preprocessing/coco_annotation.py` 从 **mzML + 标注 xlsx 联合生成**（EIC 图像与推理管线完全一致：400x300、标注 RT ±1min 窗口）：

```powershell
cd model

python -m preprocessing.coco_annotation `
  --mzmls ../data/test/mzml/20260715_shiyaoyuan_test_1.mzML `
          ../data/test/mzml/20260715_shiyaoyuan_test_2.mzML `
  --labels ../data/label/20260715_shiyaoyuan_test.xlsx `
  --output_dir ../data/test/coco `
  --force                    # B 范式切换后必须 --force 重建（旧缓存为旧格式生成）
```

**ROI 由标注驱动（B 范式）**：每行标注 `(compound, channel)` 经 `label_key` → `native_id`「化合物名-1/-2」匹配 mzML 色谱，**窗口中心 = 标注 `rt` 字段**（非谱图最高强度点）；未标注的 mzML 通道不生成 ROI。标注格式（`data/label/<trial>.xlsx`，多峰）：

- `peak_label`：**0=负样本 / 1=正样本**（其余值不入数据集）
- `peak_start1-3` / `peak_end1-3`：正样本的多峰区间，每个有效区间生成一个 bbox（**最多 3 个**）
- `peak_label=0` 的行同样生成 ROI 图但**无 bbox**，作为训练负样本（训练模型识别"图上无峰"）
- 窗口中心取自 `rt` 列；RT 一致性 QC（防线 1）构建时默认启用，跨样品/双离子极差 >1 min 的可疑行剔除、不入数据集

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

```powershell
cd model

python -m train `
  --coco_path ../data/test/coco `
  --output_dir output `
  --device auto `
  --epochs 30 --batch_size 4 `
  --lr 1e-4 --lr_backbone 1e-5
```

### 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--coco_path` | `data/test/coco` | 数据集根目录 |
| `--device` | `auto` | CUDA > MPS > CPU 自动选择 |
| `--epochs` | `30` | 训练轮数 |
| `--batch_size` | `4` | 批大小 |
| `--lr` / `--lr_backbone` | `1e-4` / `1e-5` | 学习率 |
| `--enc_layers` / `--dec_layers` | `1` / `1` | Transformer 编/解码器层数 |
| `--num_queries` | `10`（checkpoint 为 `3`） | 最大检出峰数 |
| `--resume` | — | 从检查点恢复 |
| `--eval` | — | 仅评估，不训练 |

### 微调与评估

```powershell
# 从预训练权重继续训练
python -m train `
  --coco_path ../data/test/coco --output_dir output_finetune `
  --device auto --resume checkpoint/quanformer.pth --epochs 50

# 仅评估
python -m train `
  --coco_path ../data/test/coco `
  --resume checkpoint/quanformer.pth --device auto --eval
```

> ⚠️ 恢复训练时会自动跳过 `class_embed` 和 `query_embed` 权重（维度可能不匹配）。
> 💡 当前 `quanformer.pth` 的训练参数：`enc_layers=1, dec_layers=1, num_queries=3, hidden_dim=256, nheads=8, dim_feedforward=2048, dropout=0.1`。

---

## 项目结构

```
MRMPFormer/
├── requirements.txt              # pip 依赖（model + desktop 合并，GPU 分段）
├── environment.yml               # Conda 环境（name: gamstekpeaking；在仓库根目录使用）
├── model/                        # ⭐ 核心代码（须在此目录下用 python -m <pkg> 调用）
│   ├── train.py                  # 训练入口（python -m train ...）
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

```powershell
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

```powershell
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
