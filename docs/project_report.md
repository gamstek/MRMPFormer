# MRMPFormer 项目报告

> 报告依据：`README.md`、`CLAUDE.md`、`docs/` 下全部文档（实验报告、四模型联合评估报告、方案/设计文档、Bug 清单）、`dev_log.md`、`imporove.md`、`User_Tutorials.md` 以及项目当前代码与输出实况整理。
> 报告日期：2026-08-24

---

## 目录

1. [项目总览](#1-项目总览)
2. [技术架构](#2-技术架构)
3. [数据体系](#3-数据体系)
4. [核心算法与模型](#4-核心算法与模型)
5. [推理管线](#5-推理管线)
6. [质量控制体系（QC）](#6-质量控制体系qc)
7. [评估体系](#7-评估体系)
8. [实验历程与关键发现](#8-实验历程与关键发现)
9. [四模型联合评估](#9-四模型联合评估)
10. [桌面端与工具链](#10-桌面端与工具链)
11. [工程治理与开发流程](#11-工程治理与开发流程)
12. [当前状态与后续规划](#12-当前状态与后续规划)

---

## 1. 项目总览

### 1.1 项目定位

**GamstekPeaking**（MRMPFormer）是**引力波智谱**科学智能部自主研发的 **LC-MS 代谢组学色谱峰检测与定量工具包**，版权归引力波智谱所有。其核心是 MRMPFormer——一种基于 **DETR（ResNet-50 + Transformer）系列** 的色谱峰检测模型。

核心思路：在提取离子色谱图（EIC）生成的 ROI 图像上训练目标检测网络，识别真实色谱峰并定位峰边界，进而对峰区间积分得到峰面积，实现定量分析。

### 1.2 输入与输出

| 项目 | 说明 |
|---|---|
| **输入** | `.mzML` 原始质谱数据（Targeted 模式另需含 `Compound Name / mz / RT` 的特征表） |
| **输出** | `prediction_refined.csv`（峰面积 + 置信度）、预测框标注图、`box_outside_snr_report.csv`、QC 系列报告表 |
| **模型** | ResNet-50 骨干 + Transformer 编解码器（hidden_dim=256, nheads=8, num_queries=3） |
| **当前开发范围** | 仅 **Targeted × Centroided（MRM 靶向定量）** 模式；其余三组合（Targeted × Profile / Untargeted × Centroided / Untargeted × Profile）保留现状、暂不开发，相关 R/CentWave 代码已从仓库删除 |

### 1.3 三大板块

工具包由三段组成：

1. **格式转换前处理**（`converters/` + `desktop/`）：厂商原始格式（`.msdata` / `.wiff`）→ 标准 `.mzML`，并提供桌面图形界面；
2. **MRMPFormer 模型**（`model/`）：训练、推理、评估三段式闭环；
3. **后处理**（`model/postprocessing/`）：SNR 筛选、峰区间精修、积分定量。

### 1.4 核心工作流：三段式闭环

本项目标准工作流为「**构建数据集 → 训练 → 推理 → 评估**」四环节闭环：

| 环节 | 做什么 | 入口 | 产物 |
|---|---|---|---|
| ① 构建数据集 | 标注 xlsx + mzML → COCO 格式 bbox（bbox 直接映射人工 `peak_start/peak_end`） | `model/preprocessing/coco_annotation.py` | `data/coco/`（train/val + `*_coco.json`） |
| ② 训练 | 用 COCO 数据集训练/微调 DETR 模型 | `python -m train --config configs/*.json` | `checkpoint/*.pth` |
| ③ 推理 | 测试 mzML + 已训练模型产出预测值 | `python -m inference.cli --mode pipeline` | `prediction.csv` → `prediction_snr.csv` → `prediction_refined.csv` |
| ④ 评估 | 预测值 + 人工标注计算模型效果 | `python -m tools.evaluation.evaluate_baseline` | `evaluation_report.json`、`match_details.csv`、`area_pairs.csv` |

**要点**：标注数据集具有双重身份（既是训练 bbox 来源也是评估 GT），两环节共用一份标注是预期设计；评估默认读阶段②原始 `prediction.csv`（衡量模型原始检测能力，不含后处理）；训练/评估分样品隔离，防数据泄漏。

---

## 2. 技术架构

### 2.1 目录结构

```
MRMPFormer/
├── requirements.txt          # pip 依赖（model + desktop 合并，GPU/CPU 分段）
├── environment.yml           # Conda 环境（name: gamstekpeaking）
├── CLAUDE.md                 # AI 辅助指令（项目级全局约束）
├── dev_log.md                # 项目开发日志（强制维护）
├── README.md / User_Tutorials.md / imporove.md
├── model/                    # ⭐ 核心代码（七子包扁平结构）
│   ├── train.py              #   训练入口（python -m train ...）
│   ├── inference/            #   推理：CLI 入口 + 预测器 + 两轮检测
│   ├── models/               #   模型定义（quanformer / mrmpformer / shared）
│   ├── preprocessing/        #   前处理：xic_extraction / ion_zenith / coco_annotation / label_qc
│   ├── postprocessing/       #   后处理：peak_refinement / snr_filter / valley_split / evaluation
│   ├── framework/            #   DETR 训练框架（datasets / util / engine，fork 自 Facebook DETR）
│   ├── utils/                #   推理辅助（io / quantify / mzml_load / roi_rt_mapping 等）
│   ├── tools/                #   批处理 / 诊断 / 可视化 / 基准测试 / 实验脚本
│   ├── checkpoint/           #   模型权重（5 个 .pth）
│   └── configs/              #   训练/推理/评估配置文件（JSON + 内联注释）
├── desktop/                  # PySide6 桌面 GUI（前处理已上线，其余占位）
├── converters/               # 格式转换（msdata/wiff → mzML）
├── data/                     # 测试数据（gitignore，不入库）
└── docs/                     # 项目文档（实验报告、评估报告、设计文档等）
```

### 2.2 分层架构与单向依赖

- `model/models/`：纯 DETR 模型定义（不依赖业务逻辑）；
- `model/framework/`：DETR 训练框架（datasets / util / engine）；
- `model/inference/`：模型加载 + 预测 + 可视化 + 统一 CLI 入口；
- `model/preprocessing/`：mzML 加载 → EIC → ROI 图像；MS1 谱图聚合；
- `model/postprocessing/`：积分定量 + 峰质量 + box↔RT 映射 + SNR 筛选 + 峰区间精修；
- `model/utils/`：公共工具；`model/tools/`：批处理 / 诊断 / 可视化 / 基准测试；
- `desktop/`：PySide6 图形界面（独立于 CLI；workers 为算法模块的 Qt 薄包装）；
- `converters/`：格式转换工具。

各层单向依赖：`inference → preprocessing/postprocessing → models/framework → utils`。

### 2.3 硬性工程约束（CLAUDE.md）

| 类别 | 约束 |
|---|---|
| **开发范围** | 仅 MRM 模式；禁止新增功能/重构非 MRM 代码（`extract_xic_from_arrays` 等保留现状）；公共代码改动须以 MRM 模式回归验证 |
| **环境** | Python 3.11；Conda 环境名固定 `gamstekpeaking`；PyTorch 按 GPU 选版本（RTX 50 系 cu128 ≥2.7.0 / RTX 40·30·20 系 cu124 2.6.0 / CPU·MPS 2.6.0） |
| **设备** | 禁止硬编码 `device='cuda'`，一律 `resolve_torch_device()`（CUDA > MPS > CPU 自动检测） |
| **模型加载** | 禁止直接 `torch.load()`，一律 `safe_torch_load()` 兼容跨 PyTorch 版本 |
| **路径** | 禁止硬编码分隔符，一律 `os.path.join()` / `pathlib.Path` |
| **数据目录** | 「实验隔离 + 数据类型五分法」（coco/label/mzml/msdata/wiff）；中间产物一律写 `../output/`，禁止写入 `data/` |
| **命令执行** | 训练/推理等真实运行命令整理后交用户执行（TRAE 沙箱会拦截 matplotlib 渲染 DLL 延迟加载与 site-packages 写入）；沙箱内仅跑纯数据验证/逻辑单测/编译检查 |

### 2.4 平台与兼容性

- Windows：避免路径空格与中文；`pycocotools` 已有 cp311 wheel；
- Linux：无桌面环境运行 GUI 需 X11 转发或 xvfb；
- macOS：Apple Silicon 自动使用 MPS 加速。
- 已知修复记录（Bugs.md）：GPU 硬编码、路径分隔符硬编码、Python 版本矛盾、空 MS1 谱图崩溃、R 命令引号、plot_results 过度并行、torch.load 兼容等 15 项均已修复。

---

## 3. 数据体系

### 3.1 数据目录规范（五分法）

```
data/
├── coco/     # COCO 训练数据集（train/ + train_coco.json、val/ + val_coco.json；_xic/ 为 XIC 中间产物）
├── label/    # 各实验人工标注（统一 data/label/<trial>.xlsx；布局：化合物/通道/rt/peak_label/多峰起止/面积）
├── mzml/     # 转换得到的 mzML
├── msdata/   # 原始 msdata
└── wiff/     # 原始 wiff
```

数据流转：`wiff/ + msdata/ →(converters)→ mzml/ →(coco_annotation + label/)→ coco/`。

### 3.2 数据集清单

| 数据集 | 内容 | 用途 |
|---|---|---|
| **20260715_shiyaoyuan**（`data/test/`） | 试药园两次进样（`_1`/`_2`），60 化合物 ×2 离子标注（120 行），COCO train/val 各 61 图 | 最初训练/微调与历次评测主数据 |
| **test1 标准试卷** | 11 mzML（8 农残混标 + 3 空白），96 个 GT 峰，`data/label/test1.xlsx` | 四模型横评的统一外域基准（此前未参与任何模型训练） |
| **traindata3** | 109 mzML（90 数字样品 + 9 空白 + 10 QC），16335 训练图 / 12677 框 | mrmpformerv1 从零训练 |
| **merged** | shiyaoyuan + traindata2（B 范式含负样本，7:3 划分） | quanformerv3 微调 |
| **multisrc** | traindata3×98 + shiyaoyuan_1 + traindata2×5 + test1×7 = **17796 图 / 13929 框**；留出 QC/QC5 + shiyaoyuan_2 + 农残混标-7/8 + 空白1-2/3 作 val | mrmpformerv2 多源联合训练 |

### 3.3 COCO 数据集构建（B 范式：标注驱动 ROI）

`preprocessing/coco_annotation.py` 从 mzML + 标注 xlsx 联合生成 COCO 数据集：

- **ROI 由标注驱动**：每行标注 `(compound, channel)` 经 `label_key` → `native_id`「化合物名-1/-2」匹配 mzML 色谱，**窗口中心 = 标注 `rt` 字段**（非谱图最高强度点，此即 B 范式，保证训练/推理图像口径统一）；未标注的通道不生成 ROI；
- **正负样本显式区分**：`peak_label=0` 为负样本（生成 ROI 图但无 bbox，模型学习"图上无峰"）；`peak_label=1` 为正样本（`peak_start1-3/peak_end1-3` 每个有效区间一个 bbox，最多 3 个）；其余值不入数据集；
- **bbox 映射**：`peak_start/peak_end`（分钟）经 `roi_windows.csv` 窗口线性映射为像素坐标（y 全高 [0,300]）；
- 训练侧构建默认启用「标注 RT 一致性 QC」（防线 1），可疑行剔除、不入数据集；
- 图像规格与推理管线完全一致：400×300 JPEG，标注 RT ±1min 窗口；
- ⚠️ 已知问题：B 范式后训练数据不再有「未标注通道负样本」，负样本仅来自 `peak_label=0`（当前仅 2 个），正负不平衡需后续补充。

### 3.4 标注格式

`data/label/<trial>.xlsx` 支持多峰格式：`peak_label`/`peak_count`/`peak_start1-3`/`peak_end1-3`/`area1-3` 等；单数 `peak_start/peak_end` 保留兼容旧文件。

---

## 4. 核心算法与模型

### 4.1 DETR 基础

模型基于 Facebook DETR（`model/framework/` fork 自官方），采用 ResNet-50 骨干 + 1 层 Transformer 编码器 + Transformer 解码器，object queries 通过匈牙利匹配与 GT 配对，一次前向输出「框 + 类别」集合。

### 4.2 QuanFormer（基线架构）

- ResNet-50 骨干 + 1 层 Encoder + 1 层 Decoder（hidden_dim=256, nheads=8, dim_feedforward=2048, dropout=0.1）；
- num_queries=3（单图最多检出 3 个峰）；
- 损失：交叉熵分类（eos_coef=0.1）+ L1 回归（权重 5）+ GIoU/CIoU（权重 2）；
- 约 30M 参数；
- **出身考据（重要）**：`checkpoint/quanformer.pth` 实为**外部下载权重**——DETR COCO 预训练 → 外部 autopeakV3 项目 peak-all 数据集（≈113 样品）微调 30+ epochs，非本项目从零训练（内嵌 `args` 字段为铁证）。其在 test1 上 F1@0.1=0.010 的差表现根因是**外来框约定与本项目评估协议的系统性错位**（窄框 ~0.23 min vs 人工宽积分边界 ~0.47 min，差 ~0.11 min），且训练集大概率不含空白负样本。

### 4.3 MRMPFormer v1：三层 Decoder + FDR 边界逐层精化（论文架构）

v1 是论文架构的原始实现（`model/models/mrmpformer/`），核心是 **FDR（Fine-grained Distribution Refinement，细粒度分布精化）**：

**网络结构**：

```
ResNet-50 → 1-layer Encoder → memory
Object Queries
Decoder Layer 1 → 完整二维初始框 b0=(cx,cy,w,h) + FDR Head1 → z1 → P1 → (l1,r1)
Decoder Layer 2 → FDR Head2 → Δz2; z2=z1+Δz2 → P2 → (l2,r2)
Decoder Layer 3 → FDR Head3 → Δz3; z3=z2+Δz3 → P3 → (l3,r3)

最终框：b*=(l3, t0, r3, b0_bottom)   ← 左右边界取 L3，上下边界取 L1 初始框
最终分类：仅采用 Decoder Layer 3 的峰概率
```

关键机制：
- **Logits 残差累加**：第 2、3 层预测上一层累计 logits 的残差（`z2=z1+Δz2`、`z3=z2+Δz3`），残差加在 Softmax 前的 Logits 上；
- **逐层边界位置反馈**：每层精化得到的左右边界 `(xL,xR)` 经 BoundaryPositionMLP 编码后加入下一层 Query 位置特征，形成反馈通路（默认可导，`detach_boundary_feedback=false` 可消融）；
- **非均匀 Bin**：33 个 bin，`W(n)=sign(u)|u|^p`（p=2.0），严格单调、对称覆盖 0；概率期望解码得到归一化偏移，尺度因子默认取初始框宽；
- 上下边界始终由第一层二维框头负责；禁止坐标残差双重累计。

**损失函数**（全部配置化）：
- 分类：**Softmax Focal Loss**（α=0.25, γ=2.0），处理背景远多于峰的类别不平衡；三层均输出分类但主损失只取 L3，中间层辅助分类由配置控制；
- 定位：**动态加权 L1**（λ_c=1/(w_gt+ε)，窄峰中心定位加强）+ **PW-CIoU**（中心距离项按 `bar_w/(w_gt+ε)` 加权，窄峰增强、宽峰放宽；`bar_w` 必须来自训练集统计而非 mini-batch）；
- **FDR 分布监督**：三层累计分布均接受左右边界软标签 CE（层权重 0.5/0.7/1.0，系数 v1 为 2.0），软标签由真实边界偏移在非均匀 Bin 上两点线性插值生成（工程回退方案，非未经确认的论文 FGL 原式）；
- 参数量 30.45M。

**v1 的已知缺陷（评估时点）**：matcher `iou_type` 未透传（恒用 GIoU，与 PW-CIoU 口径错位）；FDR 损失有效权重偏高（≈8.8）。

### 4.4 mrmpformerv2：架构修复 + 多源联合训练（当前最优）

v2 = v1 热启动 + 三项关键优化：

| 优化项 | 内容 | 效果 |
|---|---|---|
| **P0-1 matcher 修复** | `build_matcher` 透传 `iou_type=ciou`，匈牙利匹配与 PW-CIoU 损失口径一致（此前恒用 GIoU） | 边界精度提升主因 |
| **P0-2 损失再平衡** | `fdr_loss_coef` 2.0→1.0（有效权重 8.8→4.4），释放分类/召回梯度 | 分类学习恢复 |
| **P2-8 多源联合训练** | multisrc 数据集（17796 图/13929 框），从 v1 热启动，AMP 混合精度 30 epochs（61 分钟） | 假阳性抑制 + 跨域泛化 |

训练加速三项同步加入：`--amp` / `--tf32` / `--cudnn_benchmark`。`pw_ciou_mean_width` 按 multisrc 合并集实测 0.1809。

### 4.5 训练入口与配置体系

- `train.py` 支持 `--config`（JSON 作为默认参数，CLI 可覆盖），环境无 pyyaml 故用标准库 json；
- 配置含内联注释（`_comment_*` 键），加载逻辑统一过滤 `_` 前缀；
- 恢复训练自动跳过 `class_embed`/`query_embed` 权重（维度可能不匹配）；`--reset_optimizer` 支持纯权重微调；
- `build_predictor` 按 checkpoint 的 `args.model` 路由模型变体（quanformer / mrmpformer），v1 自动触发旧 checkpoint 迁移（L2/L3 由 L1 复制初始化）。

### 4.6 训练期 COCO 评估指标

每次 epoch 末输出 12 行标准 COCO 检测指标（AP 6 行 + AR 6 行）：

- AP@0.50:0.95 为横向可比主指标；AP@0.50 宽松口径；AP@0.75 对边界贴合极度敏感；
- 本项目特殊性：bbox 即人工积分边界，**AP50 与 AP75 之差 ≈ 边界贴合度 ≈ 定量口径的 RT 起止偏差来源**；
- AR@1 贴近部署口径（每通道最终只保留面积最大框）；`num_queries=3` 使 AR@10=AR@100；
- small 档恒为 -1 属常态（val 标注 1410 框中无 small 档），-1 表示未定义而非 0；
- 解读经验：若定量优秀而 AP75 低，说明框宽约定与人工边界存在系统性偏差（实验日志 001 教训）。

---

## 5. 推理管线

### 5.1 统一入口与三种模式

统一入口 `model/inference/cli.py`（`python -m inference.cli`），`--mode` 切换三种模式：

| 模式 | 说明 | 适用场景 |
|---|---|---|
| `pipeline` | 完整管线：ROI 提取 → 预测 → SNR 筛选 → 精修（单文件或目录递归） | ⭐ 生产环境（推荐） |
| `roi` | 仅 EIC/ROI 提取（无需 `--model`，无预测 CSV） | 检查 XIC/ROI 质量 |
| `roi2inference` | 对已有 XIC/ROI 中间结果目录批量预测+积分 | 续跑 / 断点恢复 / ROI 复用 |

`roi`/`pipeline` 均支持 `--mzml` 单文件或 `--batch_dir` 目录递归扫描（含子目录）；不同子目录同名 stem 自动路径展平防覆盖。

### 5.2 四阶段产物

| 阶段 | 输出目录 | 产物 |
|---|---|---|
| ① ROI 生成 | `xic-roi-batch/<key>/` | ROI jpeg（400×300）、`feature.csv`、`roi_windows.csv`、`xic_matrix.npy`、`pipeline_qc_excluded.csv` |
| ② 模型预测 | `batch_predictions/<key>/` | `prediction.csv`（峰面积+置信度）、`predicted_plots/`（--plot） |
| ③ SNR 筛选 | `snr_filtered/<key>/SNR_box_<thr>/` | `prediction_snr.csv`、`box_outside_snr_report.csv`、`snr_kept/`/`snr_dropped/`（--save_snr_jpeg） |
| ④ 峰区间精修 | 同上 | `prediction_refined.csv` ⭐、`refined_plots/`（--plot） |
| 计时 | `base_out/` | `pipeline_timing.log`、`pipeline_timing_runs.jsonl` |

### 5.3 ROI 生成（B 范式）

- 不传 `--labels`：各通道谱图**最高强度点**居中（推理无需标注的契约）；
- 传 `--labels`：**标注驱动**——仅标注命中通道生成 ROI，窗口中心 = 标注 `rt` 字段，并由防线 1（标注 RT 一致性）把关剔除涉事通道。

### 5.4 关键参数速查

- **通用**：`--model`（必填，roi 除外）、`--threshold`（默认 0.99）、`--integration_method`（linear/raw/external_baseline）、`--smooth_sigma`（0.0）、`--plot`、`--output_dir`；
- **Pipeline QC**：`--pipeline_min_max_intensity`（1000）、`--pipeline_min_chrom_points`（10）；
- **SNR**：`--snr_min`（3.0）、`--snr_gaussian_sigma`（0.8）、`--snr_min_noise_points`（5）；
- **Post 精修**：20+ 参数（次峰比例、噪声窗口、边界外推、谷值回退、置信度/SNR 门控等），详见 README 完整参数模板；
- **输出控制**：`--no_timing`、`--save_snr_jpeg`。

> 输出目录约定：正式产物统一写到 `../output/`；测试/试跑显式指定 `../output/test/<名称>` 单独存放。

### 5.5 常见工程问题

- `batch_size=1` 逐张推理导致 GPU 利用率仅 ~15%（待修复）；
- 模型每次预测重新 `torch.load`（GUI 场景，待缓存化）；
- `plot_results` 用 `joblib(n_jobs=-1)` 会导致 I/O 争抢（已改默认 n_jobs=2）。

---

## 6. 质量控制体系（QC）

数据质量问题是定量实验失败的常见根因：**错误标注**污染训练 bbox、误导评估 GT；**异常通道**（低强度/少点数/低信噪比）产生假阳性检测。为此管线设置了**五道防线**：

| # | 防线 | 位置 | 检查内容 | 参数（默认） |
|---|---|---|---|---|
| 1 | **标注 RT 一致性** | `label_qc` → 数据构建 & 推理管线 | ①跨样品：同化合物同通道 RT 极差；②双离子：定量/定性离子 RT 极差。极差 >1 min 判疑似实验有误，警示人工复核 + 涉事行剔除 | `--qc_label_rt_tol`(1.0) |
| 2 | ROI 通道级 | `preprocessing/xic_extraction.py` | 平滑后整条 XIC 最大强度过低、RT 点数过少 → 不生成 ROI | `--pipeline_min_max_intensity`(1000)、`--pipeline_min_chrom_points`(10) |
| 3 | 预测框级 | `inference/predictor.py` | score < 阈值的检测框不输出 | `--threshold`(0.99) |
| 4 | SNR 框级 | `postprocessing/snr_filter.py` | 框外 SNR、框外噪声点数联合判定 | `--snr_min`(3.0)、`--snr_min_noise_points`(5) |
| 5 | 精修框级 | `postprocessing/peak_refinement.py` | 精修置信度、SNR、次峰比例、框宽上限等门控 | `--post_min_confidence`(0.99)、`--post_min_snr`(3.0) 等 |

### 6.1 防线 1 详述（标注 RT 一致性）

物理依据：同一色谱方法下同化合物 RT 高度稳定（连续进样漂移通常 <0.1 min）；定量/定性离子必然共流出。极差 >1 min 说明标注画错、通道张冠李戴或仪器 RT 异常漂移。

剔除粒度（宁缺毋滥策略）：

| 场景 | 策略 |
|---|---|
| 跨样品、组内样品数 ≥3 | 仅剔偏离组中位数 >1 min 的样品（多数派可信） |
| 跨样品、组内样品数 =2 | 两行都剔（无法仲裁谁错）+ 警示人工复核 |
| 双离子 | 两通道都剔（无法判断定量/定性谁错） |

实现为两遍向量化 groupby（labels 单次解析三方复用）；推理侧通过可选 `--labels` 启用（不传则跳过，保持"推理无需标注"契约）；训练数据构建默认启用。

### 6.2 防线 2 详述（ROI 通道级）

**位置**：`preprocessing/xic_extraction.py`（`extract_xic_with_pyopenms`，参数经 cli `_pipeline_qc_kwargs` 下发）。

**检查内容**：ROI 生成前的**通道级预过滤**，两道判定——

1. **最大强度门限**：平滑后整条 XIC 的最大强度低于 `--pipeline_min_max_intensity`（默认 1000）→ 通道强度过低，判为无有效信号，不生成 ROI；
2. **点数门限**：单条色谱的 RT 点数少于 `--pipeline_min_chrom_points`（默认 10）→ 通道点太稀，无法支撑可靠峰检测，不生成 ROI。

两参数均支持置 0 关闭。

**B 范式下的补充剔除**（标注驱动模式新增 reason）：标注了 `rt` 但缺失/非法 → `label_rt_missing`；标注了但 mzML 无对应通道 → `label_no_channel`；被防线 1 判「疑似实验有误」的涉事通道 → 剔除（reason 记入同一 `pipeline_qc_excluded.csv` 结构）。

**语义与结果去向**：这是**通道级**预过滤——低强度/少点数通道**不进模型**（区别于防线 4 的框级复核）；被剔除通道记录于各样品 `pipeline_qc_excluded.csv`（含 reason 列），跨样品汇总为 `qc_roi_channels.csv`。

### 6.3 防线 3 详述（预测框级）

**位置**：`inference/predictor.py`。

**检查内容**：模型输出阶段的**框级阈值门控**——

1. **置信度门限**：`score < --threshold`（默认 0.99）的检测框不输出（建议 0.99 起步，过低会引入假峰；四模型联合评估后对 mrmpformerv2 建议 0.5，因其置信度平台宽阔）；
2. **feature 校验**：feature 中无对应化合物的通道跳过；
3. **通道去重**：每 `(mz, q3)` 通道只保留面积最大的检测行。

**结果去向**：剔除统计写入 `qc_prediction_threshold.csv`。⚠️ 注意：DETR 每图输出 `num_queries=3` 个候选框，推理端按「峰概率 argmax」选框（实验日志 001 的取类 bug 已修复为 `[1:]` 峰概率列），本道防线的阈值是在此之后对最终输出框的门控。

### 6.4 防线 4 详述（SNR 框级）

**位置**：`postprocessing/snr_filter.py`。

**检查内容**：对预测框做**信噪比复核**（阶段③），联合判定——

- **框外 SNR**：预测框边界之外的信号与噪声之比，`--snr_min`（默认 3.0）为最低阈值；
- **框外噪声点数**：`--snr_min_noise_points`（默认 5），要求框外噪声样本足够，保证 SNR 统计可靠；
- SNR 计算时强度先做高斯平滑（`--snr_gaussian_sigma` 默认 0.8）。

**语义**：与防线 2 互补——防线 2 在阶段①从**通道**层面排除低质量输入，防线 4 在阶段③对**已检出的框**做信噪比质量门控。管线模式下 QC 参数仅在阶段①生效、阶段③只做 SNR（修复了早期双阶段重复下发同一对参数的冗余问题；`snr_filter` 独立 CLI 入口仍保留参数供单独调用）。

**结果去向**：逐框明细写 `box_outside_snr_report.csv`（含 `passed_*` 列），跨样品汇总为 `qc_snr_boxes.csv`；通过行输出 `prediction_snr.csv`；`--save_snr_jpeg` 时生成 `snr_kept/`（保留）与 `snr_dropped/`（剔除）红框标注图。

### 6.5 防线 5 详述（精修框级）

**位置**：`postprocessing/peak_refinement.py`。

**检查内容**：峰区间精修后的**最终质量门控**（阶段④，20+ 项参数），核心门控包括——

| 门控 | 参数（默认） | 说明 |
|---|---|---|
| 精修后最低置信度 | `--post_min_confidence`(0.99) | 低于则整框剔除 |
| 精修后最低 SNR | `--post_min_snr`(3.0) | 低于则整框剔除 |
| 次峰最小比例 | `--post_min_secondary_ratio`(0.04) | 次峰相对主峰动态比例下限 |
| 小峰 RT 容差 | `--post_small_peak_rt_tol`(0.25) | 小峰相对主峰 RT 偏差上限 |
| 框宽上限 | `--post_refine_width_max_expand_vs_pred`(1.08)、`--post_refine_width_max_frac_of_roi`(0.45) | 修正框宽不得超过原始预测/ROI 窗口的倍数与比例 |
| 边界外推约束 | `--post_edge_max_span_min`(0.24)、`--post_edge_noise_percentile`(55) 等 | 峰顶单侧估计截停的 RT 跨度与噪声分位数 |

**语义**：本道防线是精修流程（主峰/次峰识别、谷值拆分、边界外推）完成后的**最终把关**——未通过的行不会出现在 `prediction_refined.csv` 中，即"不出现在 refined 输出即被剔"。

**结果去向**：剔除明细汇总为 `qc_post_refinement.csv`，供逐框追溯剔除原因。

### 6.6 统一 QC 输出

所有环节 QC 结果统一写入 `../output/QC/<run_name>/`：

```
├── qc_label_rt.csv            # 标注 RT 一致性（含保留行）
├── qc_roi_channels.csv        # ROI 通道级剔除汇总
├── qc_prediction_threshold.csv# 预测阈值剔除统计
├── qc_snr_boxes.csv           # SNR 逐框明细
├── qc_post_refinement.csv     # 精修门控剔除明细
└── qc_summary.md              # 各环节检查数/剔除数/人工复核清单
```

### 6.7 真实数据发现（QC 价值实证）

- **2026-08-20**：防线 1 在 20260715 实验标注（120 行）首跑即发现 **8 组双离子 RT 异常、16 行判「疑似实验有误」**——乙酰甲胺磷（极差 25.8 min）、灭螨醌（15.9）、甲羧除草醚（12.8）、羟基-灭螨醌（15.9）等定量/定性离子不共流出，极可能是定性离子通道标注到了干扰峰。此前 v1/v2 评估 GT 与 COCO 训练 bbox 均含这些可疑行，相关通道需在人工复核后重新审视。
- **2026-08-24**：test1 评估链路启用 QC 后剔除 **22 项**（223-对硫磷定量/定性离子 RT 极差 ~4.2 min，跨全部 11 个样品），涉及通道不生成 ROI、不进训练、不参与评估指标。

---

## 7. 评估体系

### 7.1 评估协议（双口径）

`tools/evaluation/evaluate_baseline.py` 定义：

- **检测口径**：预测起止 vs 人工起止偏差均 ≤ 容差判 TP，算 P/R/F1；容差取 0.1 / 0.3 / 0.5 min（早期版本为 tIoU>0.95，因模型框比人工积分边界宽 0.06~0.09 min 而过严，已废弃）；
- **定量口径**：与检测同容差配对后计算面积 R²、RT 起止偏差中位、RSD；
- **评估读阶段②原始 `prediction.csv`**（不含 SNR 筛选/精修等后处理）；若要评估整条管线最终产物应改读 `prediction_refined.csv`——两套口径不可混用；
- 支持 `--run_inference 1` 自动跑推理 / `0` 复用已有 prediction.csv。

### 7.2 指标定义

| 指标 | 定义 | 读法 |
|---|---|---|
| TP / FP / FN | 命中 / 误检（含空白样臆造峰）/ 漏检 | ±0.1 min 严格口径最考验边界贴合 |
| P / R / F1 | 精度 / 召回 / 调和均值 | 综合检测能力 |
| RT 起/止偏差 | 配对成功预测与人工起止偏差中位数 | 边界贴合精度，直接影响定量 |
| 面积 R² | 预测面积 vs 人工面积线性拟合 | 定量准确性 |
| RSD | 同化合物跨进样预测面积相对标准差（中位） | 定量重复性 |
| FP_blank | 空白样（无峰通道）上的假阳性数 | 假阳性抑制能力 |

### 7.3 评估链路修复记录

2026-08-22 test1 导入时修复三处（不修则结果全错）：

1. `evaluate_baseline._parse_gt_peaks`：跳过 `peak_label=0`（空白负样本），避免 `"0"` 被解析成 RT=0 的伪 GT 峰；
2. GT 匹配：兼容 compound 列已带通道后缀的标注，否则 `label_key` 二次拼接导致 GT 全空；
3. `visualize_compare`：多峰格式（peak_start1-3）解析修复，修复前 GT 全 None（TP/FN/FP 全 0 假象）。

---

## 8. 实验历程与关键发现

### 8.1 实验日志 001：v1「先成功后失败」之谜 —— 取类 bug 与 shadow query 假象（2026-08-17）

**现象**：同一份 v1 权重、同一批数据，两次评测结果天壤之别（2026-08-16：面积 R²=0.99998、定量优秀；2026-08-17 口径修正后：检测 F1=0.008「全灭」）。

**根因**：`model/utils/predict_utils.py` 的**取类 bug**——修复前 `probas = pred_logits.softmax(-1)[0, :, :-1]` 取到的是**背景概率列**，留下的是「模型自认为背景」的 query 的框（shadow query）；修复后 `[1:]` 取峰概率列，选中「自信为峰」的 query 的框。

**逐 query 铁证**（`tools/evaluation/dump_queries.py` 直接前向模型打印 3 个 query 的 P(峰)/P(背) 与框偏差）：

```
阿维菌素-1  GT[16.428, 16.969]
  q0  P(峰) 0.990  框RT[16.484, 16.714]  起偏 +0.06 / 止偏 -0.26
  q1  P(峰) 0.000  框RT[16.435, 17.113]  起偏 +0.01 / 止偏 +0.14   ← 旧bug选中(背>0.9)
  q2  P(峰) 0.995  框RT[17.295, 17.577]  起偏 +0.87 / 止偏 +0.61   ← 新代码选中(峰>0.9)
```

统计：14 张有 GT 图中「背景概率最高 query 的框更贴 GT」占 13/14。v1 的三条 query 呈现系统性分工：q0/q2 为旧训练「紧框包 apex」约定（P(峰)≈0.99），q1 为唯一符合人工积分约定的宽框（P(背)≈0.00）——**旧 bug 阴差阳错捡到了唯一的好框**，之前 v1 所有好看的定量数字（R²=0.99998）全部来自 shadow 框，且 prediction.csv 里的 score≈0.99 实为背景概率。

**统一口径复测（±0.1 min，score≥0.90）**：v1 F1=0.008（真实水平）vs v2（微调）F1=0.455、面积 R²=0.99999、RT 起止偏差中位 0.059/0.080 min、RSD 中位 1.91%。

**方法论教训**：
- DETR 类多 query 模型评测必须核验「选框依据」与「置信度语义」是否一致；
- 单一指标好（面积 R²）不等于流程正确，需多口径交叉验证；
- 修复取类 bug 是**评测口径修正**，v1 不是「变差」而是「显形」。

### 8.2 实验日志 003：quanformer「基线」真实出身考据（2026-08-22，仅分析）

读取 `checkpoint/quanformer.pth` 内嵌 `args` 字段（外部训练遗留铁证）：

```
resume='D:\workspace\autopeakV3\detr-r50-e632da11.pth'   ← DETR 官方 COCO 预训练起点
coco_path='D:\workspace\train-dataset\peak-all'          ← 外部色谱峰数据集
output_dir='D:\workspace\autopeakV3\output\peakdetr\...' ← 外部项目 autopeakV3
```

**真实训练链**：DETR COCO 预训练 → 外部 peak-all 数据集微调 → 下载为本项目"基线"。据此**推翻**此前「数据饥饿从零训练」的旧分析，baseline 差的真实根因：

1. **框约定错位（主因）**：peak-all 标注为贴 apex 窄框（~0.23 min），与本项目人工宽积分边界（~0.47 min）系统性差 ~0.11 min——恰好卡在 ±0.1 之外、±0.3 之内（解释 F1@0.1=0.010 → F1@0.3=0.914 跳变：不是检不到，是口径不对）；
2. **空白盲区**：peak-all 大概率不含空白负样本，模型带着"每图必有峰"先验，在 3 张空白图上照常 0.99 触发（36 通道臆造 17 峰）；
3. **v2 零检出根因**：v2 = 外部权重 + shiyaoyuan **单样品 61 图**微调（lr 1e-5 × 10 epochs），双重病理——**灾难性遗忘**（窄域微调磨掉宽域泛化）+ **窄域概念重定义**（正样本从 apex 斑块换成人工宽积分边界）。v3 对照（同基线同 epochs，仅数据多样性提升）正常触发 105 框，证明「窄数据微调宽模型 = 遗忘快于学习」。

**启示**：quanformer.pth 不宜再作为"本项目基线"参与对比；下载权重的第一步是考据 `checkpoint['args']`；窄域微调成熟模型的正确姿势是小学习率 + 少 epoch + 保留多样本（或多源回放）。

### 8.3 实验日志 002：test1 标准试卷、四模型横评与 mrmpformerv2 的诞生（2026-08-22）

**背景**：反复出现"同一模型换数据集排名反转"，各模型能力缺乏统一、公平的度量。引入 **test1 标准试卷**（从未参与任何模型训练的外域基准），并把多源联合训练作为根治跨域不稳定的主要手段。

**关键事件**：
- test1 数据准备（mzML 按内部 sample name 重命名，81~91 保留）+ 评估链路三处修复；
- 四模型首轮横评：mrmpformerv1 在 test1 上 F1@0.1=0.925 碾压 baseline（0.010）/v3（0.039）；放宽容差后 baseline/v3 跳升至 0.914/0.927——证明它们"检得到峰"但边界系统性偏差 ~0.11 min；
- **对照实验**：用 test1 7 个样品微调基线（留出 4 个 24 峰），微调基线 F1@0.1=0.836（边界偏差 0.11→0.009，边界约定可被数据学到），但空白臆造 7 FP 未被解决；v1 仍领先 0.06 F1——**架构优势（空白抑制）真实存在，非纯约定巧合**；
- **架构审查**（含训练日志诊断）发现：FDR 三层精化 IoU 增益仅 +0.005、FDR 损失有效权重 ≈8.8 过度主导、右边界 MAE 系统性偏大 23%、以及真实 bug——`build_matcher` 未透传 `iou_type`，匈牙利匹配恒用 GIoU 与 PW-CIoU 口径错位；
- 执行 P0-1（matcher 修复，已提交 git a14c438）、P0-2（fdr_loss_coef 2.0→1.0）、P0-4（阈值校准扫描 0.4~0.7 平台 F1≥0.92，推荐 0.5）、P2-8（multisrc 多源数据集）→ **mrmpformerv2** = v1 热启动 + 全部改动，AMP 训练 30 epochs（61 分钟）。

---

## 9. 四模型联合评估

### 9.1 评估设置

- 推理阈值 score≥0.5（统一）；各数据集内四模型共享同一套 ROI 图（400×300，标注驱动 B 范式）；
- 检测口径：起止偏差均 ≤ 容差判 TP（0.1 / 0.3 / 0.5 min）；定量与检测同容差配对。

**数据集与数据泄漏矩阵**：

| 评估集 | baseline | quanformerv3 | mrmpformerv1 | mrmpformerv2 |
|---|---|---|---|---|
| test1 全量（11 样品，96 峰） | ✗ | ✗ | ✗ | ✓（7/11 在训练集） |
| test1 **holdout4**（24 峰，公平留出） | ✗ | ✗ | ✗ | ✗（均为 val） |
| shiyaoyuan（100 峰） | ✓ 训练域 | ✓ 训练域 | ✗ 外域 | ✓ 半外域 |

### 9.2 公平留出集（holdout4，24 峰，四模型均未训练）

| 模型 | F1@0.1 | F1@0.3 | RT 起/止@0.1 | R²@0.3 | RSD | 空白 FP |
|---|---|---|---|---|---|---|
| baseline | 0.000 | 0.800 | — | 0.98004 | 1.27% | 12 |
| quanformerv3 | 0.035 | 0.842 | 0.019/0.085 | 0.98108 | 1.09% | 9 |
| mrmpformerv1 | 0.894 | 0.936 | 0.016/0.032 | 0.98968 | 1.31% | 1 |
| **mrmpformerv2** | **0.913** | **0.957** | **0.007/0.012** | **0.99052** | 1.33% | **0** |

### 9.3 test1 全量（96 峰；v2 有 7/11 训练泄漏，仅供参考）

| 模型 | F1@0.1 | F1@0.3 | TP/FP/FN@0.1 | RT 起/止@0.1 | R²@0.3 | 空白 FP |
|---|---|---|---|---|---|---|
| baseline | 0.010 | 0.914 | 1/111/95 | 0.0996/0.0858 | 0.98201 | 17 |
| quanformerv3 | 0.039 | 0.927 | 4/105/92 | 0.0199/0.0901 | 0.98332 | 14 |
| mrmpformerv1 | 0.925 | 0.946 | 86/4/10 | 0.0125/0.0334 | 0.99167 | 2 |
| **mrmpformerv2** | **0.931** | **0.963** | **87/4/9** | **0.0051/0.0040** | **0.99221** | **1** |

### 9.4 shiyaoyuan（100 峰；域内/域外对照）

| 模型 | F1@0.1 | F1@0.3 | F1@0.5 | RT 起/止@0.1 | R²@0.5 |
|---|---|---|---|---|---|
| baseline（域内） | 0.590 | 0.950 | 0.980 | 0.0542/0.0494 | 0.99997 |
| quanformerv3（域内） | 0.570 | 0.980 | **1.000** | 0.0191/0.0472 | 0.99997 |
| mrmpformerv1（外域） | 0.380 | 0.870 | 1.000 | 0.0295/0.0443 | 0.99998 |
| mrmpformerv2（半外域） | 0.442 | 0.874 | 0.995 | **0.0160/0.0346** | 0.99998 |

### 9.5 阈值敏感性（v2，test1 全量，tol=0.1）

```
thr    TP   FP   FN      P       R      F1
0.30   90   16    6   0.849  0.938  0.891
0.40   90    9    6   0.909  0.938  0.923
0.50   87    4    9   0.956  0.906  0.930  ← 最优
0.60   86    3   10   0.966  0.896  0.930
0.70   84    2   12   0.977  0.875  0.923
0.80   76    2   20   0.974  0.792  0.874
0.90   46    1   50   0.979  0.479  0.643  ← 高阈值召回崩塌
```

0.4~0.7 平台 F1≥0.92（最大回撤 <1%），**建议生产配置固定 threshold=0.5**；偏重精度可上探 0.7，偏重召回可降至 0.4。

### 9.6 核心结论

1. **边界贴合度是四模型的真实分水岭**：baseline/v3 并非"检不到峰"（±0.3 下召回 0.99+），而是框边界与 test1 人工积分约定存在 ~0.1 min 系统性错位——这是**训练数据约定与考卷约定的差异**，非架构能力问题；
2. **v1→v2 增量归因**：matcher 修复 + 损失再平衡 → 边界精度（止边界偏差 -63%、起边界 -56%）；多源空白负样本 → 假阳性抑制（空白 FP 1→0）；**召回未变**（剩余 3 个 FN 为同一批难例，需 P1 级改动）；
3. **±0.1 严格 F1 ≈ 边界约定匹配度**：跨数据集排名反转（test1 上 v 系碾压 / shiyaoyuan 上 baseline 主场）是约定差异所致；根治靠多源联合训练，架构优化不能替代数据多样性；
4. **空白负样本的数量与多样性是假阳性抑制的第一要素**（baseline 每张空白图臆造 6 峰，会造成假阳性质控违规）；mrmpformerv2 在公平留出的空白样上**零臆造**，是唯一通过"空白质控"的模型；
5. **定量能力全员优秀**（R² 0.980~1.000、RSD ≤3.6%），一旦配对成功差异很小——**定量瓶颈在检测配对，不在积分算法**；
6. **v2 跨域边界一致性最佳**：在 shiyaoyuan（半外域）F1@0.1=0.442 > v1（纯外域）0.380，且 RT 偏差反超两个域内模型——多源训练在不损失主域精度的前提下显著改善跨域泛化。

### 9.7 选型建议

| 排名 | 模型 | 一句话评价 |
|---|---|---|
| 🥇 | **mrmpformerv2** | 边界最准（RT 偏差减半）、空白零臆造、跨域稳定性最佳、阈值平台宽；**推荐生产部署（threshold=0.5）** |
| 🥈 | mrmpformerv1 | 严格口径仍可用的第二选择，边界精度与空白抑制均逊于 v2 |
| 🥉 | quanformerv3 | 域内（shiyaoyuan）宽容差表现极佳，但边界约定绑定训练域，外域严格口径失效 |
| 4 | baseline | 仅域内宽容差可用；空白臆造严重，不建议生产 |

**局限说明**：test1 全量下 v2 有训练泄漏，公平结论以 holdout4 为准；shiyaoyuan 仅 2 样品、holdout4 仅 24 峰，统计效力有限；容差为对称双侧口径。

---

## 10. 桌面端与工具链

### 10.1 桌面端 GUI（GAMSTEKPEAKing，`desktop/`）

基于 **PySide6** 的桌面图形界面，左侧边栏（180px）+ 卡片式内容区 + 深色科技风主题（QSS 全局样式表）：

- **已上线**：「前处理」板块——功能卡片 1「格式转换」（拖拽/多选 `.msdata`，后台 QThread 逐文件调用 `msdata2mzml.exe`，进度条 + 文件级状态 ⏳/🔄/✅/❌）；功能卡片 2「离子天顶」（遍历 mzML 的 MS1 谱图，提取每个 m/z 信号顶点，输出 `(m/z, RT, intensity, n_observations)` CSV，含 m/z 范围/容差/强度过滤/最大谱图数/重建索引等参数）；
- **灰色占位**：寻峰 / 定量 / 模型 / 设置（未来整合 DETR 预测、EIC 可视化、峰面积定量、模型管理）；
- 工程细节：未捕获异常写入 `desktop/error.log` 并弹窗；耗时任务由 `desktop/workers/`（QThread + Signal）执行，进度实时更新；`workers/ion_zenith.py` 为 `preprocessing/ion_zenith.py` 纯算法模块的 Qt 薄包装（算法已抽离、去 Qt 依赖，可用 `python -m preprocessing.ion_zenith` 直接调用）。

### 10.2 格式转换工具链（`converters/`）

| 脚本 | 输入 | 输出 | 工具链 |
|---|---|---|---|
| `converters/msdata.py` | `.msdata` | `.mzML` | `msdata2mzml.exe`（OpenMS，运行时内置于 `msdata_bin/`） |
| `converters/wiff.py` | `.wiff` / `.wiff2` | `.mzML` | `msconvert.exe`（ProteoWizard，`wiff_bin/`） |
| `converters/rename_cn.py` | — | — | 中文文件名 → 英文 |

工程教训：项目路径不得含中文（OpenMS C++ 层限制）；WIFF 需同名 `.wiff.scan` 配套文件；`msdata2mzml.exe` 仅接受位置参数且成功也可能返回退码 858，以「是否生成 .mzML 文件」判定成败；`wiff.py` 默认带峰检测（peakPicking → Centroided），`--no-peak-picking` 保留 Profile 轮廓。

### 10.3 辅助工具集（`model/tools/`）

| 子模块 | 用途 |
|---|---|
| `batch/` | 批量重处理（`--stage snr/post/snr-post` 三模式统一） |
| `mzml/` | 色谱图查看/导出（`list/show/export` 三子命令） |
| `benchmark/` | 性能基准测试（sampler / aggregate / report / runner 四模块拆分，GPU 显存采样 + 统计聚合 + 报告输出） |
| `evaluation/` | 一键评测、对比可视化、逐 query 诊断 |
| `diagnostics/` | box↔RT 映射核查、chrom-SNR 对齐、案例证据导出等 |
| `experiments/` | 江南/欧陆实验专项脚本（面积对比、化合物存在性等） |
| `visualization/` | GT vs 预测绘图、精修 XIC 绘图 |
| `maintenance/` | 强制空白负样本、结果整理、XIC 再生成 |

---

## 11. 工程治理与开发流程

### 11.1 项目级 AI 指令（CLAUDE.md）

作为项目级全局指令自动注入每次模型调用上下文，强制：开发范围收敛于 MRM、环境与路径硬约束、数据/输出目录规范、外部命令执行策略（沙箱限制说明）、架构边界、已知陷阱清单。

### 11.2 开发日志机制（dev_log.md）

强制规则：每次模型完成生成/修改/测试后须同步更新 `dev_log.md`——项目概述（目标/输入/输出/方法介绍，仅在定位变化时更新）+ 按日期分组的开发时间线（`- <类型>(<作用域>): <描述>`，类型含需求分析/数据建模/代码生成/调试/文档生成/重构/测试/其他）。配套 `.github/skills/dev-log-writer/` 技能模板。

### 11.3 测试体系

- `model/tests/test_mrmpformer_v1.py`：提示词 §15 全部 8 类测试 **21 项全过**（CPU + DummyBackbone 免下载）——Shape / 残差零初始化恒等与逐元素累加 / 分布解码（one-hot 期望=W(n)）/ 最终框组装（左右=L3 上下=L1）/ 分类唯一来源 L3 / 边界反馈梯度 / 损失（空目标/单峰/多峰/极窄峰/退化框/Focal 无 NaN/动态 L1 公式/PW-CIoU 权重/软标签）/ legacy 迁移 / 4 样本 tiny-set 过拟合；
- 22 项单测回归贯穿历次重构（训练终端输出改造、label_map 崩溃修复等均以单测回归验证）；
- `label_qc`、`coco_annotation` 等数据侧模块均有针对性单测。

### 11.4 代码质量改进记录

- **prediction.csv「一名三义」修复**（plan_products §1）：`prediction.csv` = 阶段②原始预测（唯一含义）、`prediction_snr.csv` = 阶段③SNR 后、`prediction_refined.csv` = 阶段④精修；SNR 阶段改名 + 读方 5 处带旧名回退兼容；
- **中文目录名 ASCII 化**：`筛选保留/筛选剔除/` → `snr_kept/snr_dropped/`；
- **QC 参数双阶段重复下发修复**：QC 仅在阶段①生效，阶段③只做 SNR；
- **roi2inference 模式 integration_method 透传**：修复手搓 Namespace 硬编码；
- **模式重构（7→3）**：消除文件收集重复、每图重载模型、双代码路径分叉等 6 项 bug；
- **训练终端输出优化**：中英对照 → 紧凑英文短码 + 中文汇总，逐步行/汇总块/图例分层，22 项单测回归；
- **终端输出方案文档**（plan_terminal）：术语去 legacy、设备横幅去重、日志分级语义修正、风格统一、噪音清理等（部分实施）。

### 11.5 环境依赖检测

`.github/skills/check-dependencies/`：`check_env.py`（纯终端文本报告）、`check_gui.py`（GUI 弹窗 + 一键修复）、`fix_env.py`（find-env/check/fix/verify），配套 `check-dependencies` / `fix-dependencies` 两个技能。

---

## 12. 当前状态与后续规划

### 12.1 当前模型家族（`model/checkpoint/`）

| 权重 | 定位 | 生产建议 |
|---|---|---|
| `mrmpformerv2.pth` | **当前最优**（多源联合训练 + 架构修复） | ✅ 推荐部署（threshold=0.5） |
| `mrmpformerv1.pth` | 论文架构原始实现（traindata3 训练） | 备选 |
| `quanformerv3.pth` | 基线架构 + merged 多数据微调 | 域内宽容差可用 |
| `quanformer.pth` | 外部下载权重（非本项目自训） | ❌ 不建议作为基线/生产 |
| `quanformerv2.pth` | 单样品微调（灾难性遗忘） | ❌ 已证伪 |

### 12.2 输出产物实况（`output/`）

- `output/QC/`：五道防线统一 QC 表（含 coco_multisrc、coco_traindata3 等各训练 run + 推理 run）；
- `output/evaluation/`：四模型 × 多数据集 × 多容差的完整评估矩阵（test1 / holdout4 / shiyaoyuan × tol 0.1/0.3/0.5 × baseline/v2/v3/v1fdr/multisrc/test1ft），含 `joint4_results.json`、可视化对比（`vis_v1fdr_vs_v3/`）；
- `output/test/`：历次推理/评测试跑产物；`output/train/`：各训练 run 日志。

### 12.3 后续规划（基于实验结论）

1. **P1 级架构改动**突破剩余 FN 难例：中间层 box 监督、FDR bin 值域放宽（v2 漏检的 3 峰与 v1 相同，属难例而非容量问题）；
2. **数据面**：补充空白负样本标注（当前仅 2 个，正负不平衡）；峰形数据增强提升小样本稳定性；新数据接入优先走 multisrc 联合训练增量微调（保持留出样品作 val）；
3. **工程面**：置信度阈值统一（第 4 项）、`build_predictor` 路由收尾（第 5 项）、`train.py` resume 逻辑修复（第 6 项）、解析时不读 TIC 图（第 8 项）、batch 推理性能（batch_size=1 → GPU 利用率 ~15%）、GUI 模型缓存加载；
4. **实验面**：标注 QC 复核后的数据重建（20260715 16 行 + test1 22 行涉事通道）、评估容差口径按业务中心偏差口径补充。

### 12.4 项目总结

MRMPFormer 项目已经走完「推理可用 → 可重训可评测 → 架构升级 → 多源联合训练」的完整迭代路径：从最初依赖外部下载权重的"黑盒基线"，到建立 COCO 数据闭环、五道 QC 防线、双口径评估协议，再到实现三层 Decoder + FDR 边界精化的论文架构（v1），最终通过 matcher 修复、损失再平衡与多源数据联合训练收敛到综合最优的 **mrmpformerv2**（holdout4 公平口径 F1@0.1=0.913、空白零臆造、跨域稳定）。过程中沉淀了三条核心方法论：**DETR 多 query 评测必须核验置信度语义**（shadow query 假象）、**严格容差 F1 度量的是边界约定匹配度而非模型能力**（跨域排名反转的根源）、**空白负样本多样性与数量决定假阳性抑制**（质控场景的生命线）。当前项目已具备面向生产的完整能力：格式转换 → 数据构建 → 训练 → 推理 → 后处理 → 评估 → QC 全链路闭环，并有桌面端 GUI、环境检测、开发日志等工程化配套，可支撑后续真实样品规模化定量实验。
