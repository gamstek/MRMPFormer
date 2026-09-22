# GamstekPeaking

## 简介

GamstekPeaking是引力波智谱科学智能部研发的，用于 LC-MS 代谢组学色谱峰检测与定量工具包。核心工作流为**前处理 → 训练 → 推理（含后处理）→ 评估**四大板块，覆盖从原始质谱数据到定量结果的完整闭环。

***

### MRMFormer简介

MRMPFormer是基于 **DETR（ResNet-50 + Transformer）系列** 的色谱峰检测模型。
核心思路：在提取离子色谱图（EIC）生成的 ROI 图像上训练目标检测网络，识别真实色谱峰并定位峰边界，实现积分面积定量。

- **输入**：`.mzML` 原始质谱数据
- **输出**：峰面积 CSV + 预测标注图
- **模型**：ResNet-50 骨干 + Transformer（hidden\_dim=256, nheads=8）。两个变体由 `--model` 选择：`quanformer`（1 Encoder + 1 Decoder 基线）；`mrmpformer_v1`（1 Encoder + 3 Decoder + FDR 边界逐层精化）
- **查询数**：num\_queries=3（最多同时检出 3 个峰）
- **开发版本**：v2.8.13
- **当前开发范围**：仅 **Targeted × Centroided（MRM）** 模式；其余三组合（Targeted × Profile / Untargeted × Centroided / Untargeted × Profile）保留现状、暂不开发

***

## 工作流总览

本项目核心工作流为**四大板块闭环：前处理 → 训练 → 推理（含后处理）→ 评估**。后处理（SNR 筛选、峰区间精修、面积积分）作为推理管线的后半段，与推理共用统一入口。

| 板块      | 做什么                                                                 | 主要入口                                                                                                                               | 核心产物                                                                                        |
| ------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| ① 前处理  | 原始数据格式转换 → mzML；标注 QC（防线1）；EIC/ROI 提取（标注驱动，防线2） | `converters/`（格式转换）、`model/preprocessing/`（标注 QC + XIC/ROI 提取）                                                              | `data/mzml/`、`xic_roi/<样品>/`（ROI jpeg + feature.csv + roi_windows.csv）、QC 表          |
| ② 训练    | 标注 xlsx + mzML → COCO bbox 数据集（bbox 直接映射人工 `peak_start/peak_end`）→ 训练/微调 DETR 模型 | `model/preprocessing/coco_annotation.py`（数据集）、`python -m train --config ...`（训练）                                               | `data/coco/`（train/val + `train_coco.json`/`val_coco.json`）、`checkpoint/*.pth`             |
| ③ 推理（含后处理） | ROI → 模型预测 → SNR 筛选 → 峰区间精修 → 面积积分 → 推理报告              | `python -m inference.cli --mode pipeline ...`（统一入口）                                                                                 | `predictions_model/<样品>/`、`prediction_refined/<样品>/`、根级 `all.csv`、`inference_report_<实验名>.md` |
| ④ 评估    | 用预测值 + 人工标注计算检测与定量指标                                             | `python -m tools.evaluation.evaluate_baseline --labels ../data/label/<实验>.xlsx`                                                | `evaluation_report.json`（P/R/F1、面积 R²、RT 偏差、RSD）、`match_details.csv`、`area_pairs.csv`     |

**要点**：

- **标注数据集双重身份**：`data/label/<实验>.xlsx` 既是训练标注来源（→ COCO bbox），也是评估 GT——两环节共用一份标注是预期设计；
- **评估口径**：评估读推理板块**原始** **`model_prediction_<样品>.csv`**（衡量模型原始检测能力），**不含** SNR 筛选/精修后处理；若需评估整条管线最终产物，目标应改为 `prediction_refined.csv`（另一套口径）；
- **数据泄漏防护**：训练/评估分样品（train=`test_1`、val=`test_2`），不在同一样品上既训又评；
- **快速路径**：四大板块的具体命令与参数见下文「前处理」「推理」「训练」「评估」四节。

***

## 数据目录规范

`data/` 按数据类型组织为五类目录（前三个为原始输入，后两个为派生产物，**与各入口代码默认路径一致**）：

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

**当前状态**：仓库内实际存在 `data/label/`（`test1.xlsx`、`traindata1~3.xlsx` 等标注）、`data/msdata/`（`test1`、`traindata1/2`）、`data/wiff/`（`traindata3.wiff` + 同名 `.scan`）；`data/mzml/`（转换产物）与 `data/coco/`（数据集构建产物）当前不在仓库——前者需用 `converters/` 转换生成或自行放置，后者由 `coco_annotation.py` 构建生成。新实验加入时按同结构在其下创建 `data/<实验名>/`。

**注意**：

- 推理/评测等中间产物**不写入** `data/`，统一输出到 `../output/`（见上文输出目录约定）；
- `data/` 整体在 `.gitignore` 中，不入版本库。

***

## 质量控制（QC）

数据质量问题是定量实验失败的常见根因：**错误标注**会污染训练 bbox（模型学到错边界）、误导评估 GT（模型好坏被误判）；**异常通道**（低强度、少点数、低信噪比）会产生假阳性检测。为此管线设置了多层 QC 防线。

### QC 防线总览

| # | 防线               | 位置                                  | 检查内容                                                                                                      | 参数（默认）                                                                 | 结果去向                                                   |
| - | ---------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------ |
| 1 | **标注 RT 一致性** | `label_qc` → 训练数据构建 & 推理管线          | ①跨样品：同化合物同通道在各样品间 RT 极差；②双离子：同一样品中定量/定性离子 RT 极差。**极差 >1 min 判疑似实验有误**：警示人工复核 + 涉事行剔除（不生成 ROI / 不进训练 bbox） | `--qc_label_rt_tol`(1.0)                                               | `output/QC/<run>/qc1_label_rt.csv`                     |
| 2 | ROI 通道级          | `preprocessing/xic_extraction.py`   | 平滑后整条 XIC 最大强度过低、RT 点数过少 → 不生成 ROI                                                                        | `--pipeline_min_max_intensity`(1000)、`--pipeline_min_chrom_points`(10) | 各样品 `pipeline_qc_excluded.csv` → 汇总 `qc2_roi.csv`      |
| 3 | 预测框级             | `inference/predictor.py`            | score ≤ 阈值的检测框不输出；feature 无化合物则跳过                                                                         | `--threshold`(0.5)                                                     | 各样品 `qc3_threshold_<样本名>.csv` → 汇总 `qc3_threshold.csv` |
| 4 | SNR 框级           | `postprocessing/snr_filter.py`      | 框外 SNR、框外噪声点数联合判定                                                                                         | `--snr_min`(10.0)、`--snr_min_noise_points`(5)                          | 各样品 `qc4_snr_<样本名>.csv` → 汇总 `qc4_snr.csv`             |
| 5 | 精修框级             | `postprocessing/peak_refinement.py` | 精修置信度、SNR、次峰比例、框宽上限等门控                                                                                    | `--post_min_confidence`(0.99)、`--post_min_snr`(10.0) 等                 | 各样品 `qc5_refined_<样本名>.csv` → 汇总 `qc5_refined.csv`     |

### 标注 RT 一致性检查（防线 1 详述）

两类检查的物理依据：同一色谱方法下同化合物 RT 高度稳定（连续进样漂移通常 <0.1 min）；定量/定性离子必然共流出。**极差 >1 min** 说明某样品标注画错、通道张冠李戴或仪器 RT 异常漂移。

剔除粒度：

| 场景           | 策略                         |
| ------------ | -------------------------- |
| 跨样品、组内样品数 ≥3 | 仅剔偏离组中位数 >1 min 的样品（多数派可信） |
| 跨样品、组内样品数 =2 | 两行都剔（无法仲裁谁错）+ 警示人工复核       |
| 双离子          | 两通道都剔（无法判断定量/定性谁错）         |

推理侧 `roi`/`pipeline` 模式必填 `--labels`，该检查随之启用（`massnova` 模式不依赖标注，自动跳过）；训练数据构建（`coco_annotation`）默认启用。

### 统一 QC 输出

所有环节的 QC 结果表统一写入 **`../output/QC/<run_name>/`**（run\_name 为本次运行输出目录名，训练侧为 `coco_<数据集名>`，同数据集多次构建汇总/覆盖到同一处）：

```
output/QC/<run_name>/
├── qc1_label_rt.csv          # 标注 RT 一致性（含保留行，便于复核）
├── qc2_roi.csv               # ROI 通道级剔除汇总（含 reason）
├── qc3_threshold.csv         # 预测阈值剔除统计（样本级 qc3_threshold_<样本名>.csv 汇总）
├── qc4_snr.csv               # SNR 逐框明细汇总（样本级 qc4_snr_<样本名>.csv 汇总）
├── qc5_refined.csv           # 精修门控剔除明细（样本级 qc5_refined_<样本名>.csv 汇总）
├── qc_summary.md             # 各环节检查数/剔除数/人工复核清单
└── qc_alert.md               # 全防线人工预警报告（需人工复核清单）
```

> **当前状态**：五道防线与统一 QC 输出目录已全部实施运行。2026-08-20 防线 1 在 20260715 实验标注中查出 20 项需人工复核（跨样品/双离子 RT 极差 12.7\~25.9 min）。

***

## 快速开始

### 环境要求

| 项目     | 要求                                                |
| ------ | ------------------------------------------------- |
| Python | **3.11**（Conda 环境名固定为 `gamstekpeaking`            |
| 包管理器   | Conda                                             |
| R（可选）  | 4.0+，仅 Untargeted 模式需要（⚠️ Untargeted 模式暂不开发，可不安装） |

**PyTorch 版本**（按 GPU 选择）：

| GPU 系列                    | CUDA         | torch  | torch vision |
| ------------------------- | ------------ | ------ | ------------ |
| RTX 50 (5060–5090)        | 12.8 (cu128) | ≥2.7.0 | ≥0.22.0      |
| RTX 40 / 30 / 20          | 12.4 (cu124) | 2.6.0  | 0.21.0       |
| CPU / Apple Silicon (MPS) | —            | 2.6.0  | 0.21.0       |

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

***

## 前处理（Preprocessing）

前处理解决「原始数据如何变成模型可用的训练/推理输入」，包含三个环节：

| 环节 | 做什么 | 入口 | 产物 |
| --- | --- | --- | --- |
| ① 格式转换 | 厂商原始文件（`.msdata` / `.wiff`）→ 标准 `.mzML` | `converters/` | `data/mzml/` |
| ② 标注 QC（防线1） | 标注 RT 一致性检查，可疑标注行剔除 + 人工复核 | `preprocessing/label_qc.py` | `qc1_label_rt.csv`、`qc_alert.md` |
| ③ EIC/ROI 提取（防线2） | 标注驱动：mzML + 标注 → ROI 图像 / 特征矩阵 / 窗口映射 | `preprocessing/xic_extraction.py` | `xic_roi/<样品>/`（ROI jpeg + feature.csv + roi_windows.csv） |

### ① 格式转换（原始数据 → mzML）

仪器厂商导出的原始文件（`.msdata` / `.wiff` / `.wiff2`）需先转换为标准 `.mzML` 格式，转换工具位于 `converters/`：

| 脚本                     | 输入                 | 输出      | 工具链                                              |
| ------------------------- | ------------------ | ------- | ------------------------------------------------ |
| `converters/msdata.py`    | `.msdata`          | `.mzML` | `msdata2mzml.exe`（OpenMS，运行时内置于 `msdata_bin/`）   |
| `converters/wiff.py`      | `.wiff` / `.wiff2` | `.mzML` | `msconvert.exe`（ProteoWizard，运行时内置于 `wiff_bin/`） |

```powershell
cd converters

# 1. 将原始文件放入项目根目录 data/ 下（读取路径：data/msdata、data/wiff）
# 2. 预览待转换文件（不执行转换）
python msdata.py --dry-run          # .msdata
python wiff.py --dry-run            # .wiff / .wiff2

# 3. 批量转换，输出统一生成于 data/mzml/<文件名>/ 子目录
python msdata.py                    # .msdata → .mzML
python wiff.py                      # .wiff → .mzML（默认带峰检测）
python wiff.py --no-peak-picking    # 保留 profile 原始轮廓
```

**注意事项**：

- 项目路径不得含中文（OpenMS C++ 层限制）
- WIFF 文件需要同名 `.wiff.scan` 配套文件
- `msdata2mzml.exe` 仅接受位置参数，且即使成功也可能返回退码 858，脚本以「是否生成 .mzML 文件」判定成败

### ② 标注文件与标注 QC（防线 1）

标注文件为 `data/label/<实验>.xlsx`，每行一个通道：`compound` / `channel`（定量/定性离子）/ `rt`（实测保留时间）/ `peak_label`（0=负样本，1=正样本）/ `peak_start1-3` / `peak_end1-3` / `area1-3`（多峰）等。

标注 QC 对标注本身做两道一致性检查（详见「[质量控制](#质量控制qc)」）：

- **跨样品极差**：同化合物同通道跨样品 RT 极差 >1 min 判疑似标注错误，组内 ≥3 时只剔偏离组中位数的行（多数派可信）；
- **样品内双离子极差**：同一样品中定量/定性离子 RT 极差 >1 min 判通道张冠李戴，两通道都剔。

被剔行的 `_qc_excluded` 标记随数据全链传导：不生成 ROI、不进训练 bbox、不参与评估指标。

### ③ EIC/ROI 提取（B 范式：标注驱动）

对每个标注命中通道，从 mzML 提取 EIC 并渲染 400×300 ROI 图：

- **窗口中心 = 标注 `rt` 字段**（而非谱图最高强度点），±1 min 裁剪，与训练图像素级一致；
- 同时写 `roi_windows.csv` 记录每张图实际 `[rt_lo, rt_hi]`，作为后续「像素 → RT 分钟」映射的唯一基准；
- 平滑后最大强度过低 / RT 点数过少的通道在防线 2 剔除（`pipeline_qc_excluded.csv`）；
- 输出 `feature.csv`（通道元数据 + native_id）与 `xic_matrix.npy`（对齐强度矩阵），供推理与训练共用。

### 前处理创新点

1. **标注驱动 ROI（B 范式）**——窗口中心取标注 rt 而非谱图最高点，训练/推理/评估共用同一提取路径，消除「训练窗口中心偏移导致模型只见峰片段」的数据不一致问题；
2. **双维度标注 RT 一致性 QC + 多数派仲裁**——跨样品与样品内双离子极差双通道抓标注错误，n≥3 只剔离群行避免误杀整组，把「错误标注」在源头拦截，不污染训练 bbox 与评估 GT；
3. **像素 → RT 映射基准前置**——ROI 提取时即记录 `roi_windows.csv`，后续预测框、积分、评估全部复用同一映射，避免各环节各自假设导致的 RT 偏移；
4. **多实验标注合并隔离**——多文件合并时 `sample_id` 加「实验名__」前缀防跨实验同名混淆，QC 按实验分别执行。

***

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

| 参数         | 默认           | 说明                 |
| ---------- | ------------ | ------------------ |
| m/z 范围     | 50 – 2000 Da | 仅处理该质量范围内的离子       |
| 容差 (ppm)   | 10.0         | 质荷比相对容差，用于离子聚合     |
| 容差 (Da)    | 0.01         | 质荷比绝对容差，与 ppm 同时生效 |
| 强度下限 / 上限  | 无            | 过滤低/高于该强度的信号，0=不过滤 |
| 最大谱图数      | 0（全部）        | 限制处理的谱图数           |
| 重建 mzML 索引 | 关            | mzML 缺少索引时自动重建     |

1. 点击「▶ 开始运行」，实时显示已扫描谱图数与聚合峰数
2. 完成后显示离子总数与耗时，可点击「📂 打开所在目录」定位产物 CSV

***

## 推理（含后处理）

推理板块 = **模型推理 + 后处理**（SNR 筛选、峰区间精修、面积积分），两者共用统一入口 `model/inference/cli.py`，端到端产出最终峰表与定量面积。

> 💡 四种分析模式（Targeted / Untargeted × Centroided / Profile）的原理、适用场景与操作流程，详见 [User\_Tutorials.md](User_Tutorials.md)。
> ⚠️ **当前项目仅开发 Targeted × Centroided（MRM）模式**，其余三模式保留现状、暂不开发。

统一入口 `model/inference/cli.py`（在 `model/` 目录下用 `python -m inference.cli` 调用），通过 `--mode` 切换 4 种运行模式（`roi` / `pipeline` 均支持 `--mzml` 单文件或 `--batch_dir` 目录递归扫描，含子目录）：

| 模式              | 说明                                                                         | 适用场景                                            |
| --------------- | -------------------------------------------------------------------------- | ----------------------------------------------- |
| `pipeline`      | 完整管线：ROI 提取 → 预测 → SNR 筛选 → 精修（单文件或目录递归）                                   | ⭐ 生产环境（推荐）                                      |
| `roi`           | 仅 EIC/ROI 提取（无需 `--model`，无预测 CSV）                                         | 检查 XIC/ROI 质量                                   |
| `roi2inference` | 对已有 XIC/ROI 中间结果目录批量预测+积分                                                  | 续跑 / 断点恢复 / ROI 复用                              |
| `massnova`      | 整谱 XIC 全峰识别：全部 transition、不依赖标注；信号级找峰 + 可选模型验证（`--model` 提供则按峰切 ±1min 窗验证） | MassNova 集成 / 仅提供时序数据（参数模板：`configs/massnova.json`） |

***

### 完整管线（推荐）

端到端：ROI 提取 → 模型预测 → SNR 筛选 → 峰区间精修。

```powershell
cd model

# 批量 mzML（最常用；目录递归含子目录）
python -m inference.cli --mode pipeline `
  --model checkpoint/quanformer.pth `
  --labels ../data/label/<实验>.xlsx `
  --batch_dir ../data/mzml/<实验> `
  --output_dir ../output/pipeline_batch `
  --threshold 0.5 --plot `
  --snr_min 10.0 `
  --pipeline_min_max_intensity 1000 `
  --pipeline_min_chrom_points 10

# 单个 mzML
python -m inference.cli --mode pipeline `
  --model checkpoint/quanformer.pth `
  --labels ../data/label/<实验>.xlsx `
  --mzml ../data/mzml/<实验>/<文件>.mzML `
  --output_dir ../output/pipeline_single_test `
  --threshold 0.5 --plot
```

> **ROI 生成方式（B 范式）**：`roi` / `pipeline` 模式**必须传** **`--labels`**（标注驱动）——仅标注命中通道生成 ROI，窗口中心 = 标注 `rt` 字段，并由防线 1（标注 RT 一致性）把关（剔除涉事通道）；不再支持 apex（最高强度点）通道驱动。`massnova` 模式不依赖标注。

**输出结构**（以 `--output_dir ../output/pipeline_batch` 为例，`<样品>` 为 mzML 文件名去后缀；`<run_name>` = 输出目录名）：

```
../output/pipeline_batch/
├── xic_roi/                             # EIC 提取 + ROI 图像（每样品一个子目录）
│   └── <样品>/
│       ├── <ROI 图>.jpeg                # 每通道一张 ROI（N_mz{母离子}_q3{子离子}_<slug>.jpeg）
│       ├── feature.csv                  # 通道元数据（Compound Name/native_id/mz/q3/RT）
│       ├── roi_windows.csv              # 每张 ROI 的实际绘图窗口（像素→RT 映射基准）
│       ├── pipeline_qc_excluded.csv     # 被 QC 剔除的通道明细
│       └── xic_matrix.npy               # 对齐强度矩阵（第 0 行为公共 RT 轴）
├── predictions_model/                   # 模型预测 CSV（推理阶段）
│   ├── <样品>/
│   │   ├── model_prediction_<样品>.csv  # ⭐ 模型逐框预测（框坐标/置信度/RT 区间/面积/SNR）
│   │   └── qc3_threshold_<样品>.csv     # 低于置信度阈值被丢弃的框数统计
│   ├── predictions_model_all.csv        # 全部样品合并（首列 stem）
│   └── predictions_model_report.md      # 模型输出阶段报告
├── prediction_refined/                  # SNR 筛选 + 峰区间精修（后处理阶段）
│   ├── <样品>/
│   │   ├── prediction_snr.csv           # SNR 筛选后预测（仅通过 SNR 阈值的行）
│   │   ├── prediction_refined.csv       # ⭐ 精修结果（主峰+次峰宽表：RT/高度/置信度/SNR）
│   │   ├── prediction_refined_with_area.csv  # 精修框补算峰面积（推理报告自动生成）
│   │   ├── qc4_snr_<样品>.csv           # SNR 逐框明细（含未通过行）
│   │   ├── qc5_refined_<样品>.csv       # 精修门控列子集
│   │   ├── feature.csv / roi_windows.csv / xic_matrix.npy  # 精修阶段子集
│   │   └── refined_plots/               # 精修标注图（--plot 时生成）
│   ├── prediction_refined_all.csv       # 全部样品精修合并（首列 stem）
│   └── prediction_refined_report.md     # 精修输出阶段报告
├── all.csv                              # ⭐ 最终跨样品合并明细（每样品×每 ROI 一行）
├── inference_report_<实验名>.md         # ⭐ 推理报告（跑完自动生成：样本摘要/化合物×样品面积矩阵/QC/列说明）
├── pipeline_timing.log                  # 阶段计时日志
└── pipeline_timing_runs.jsonl           # 计时记录（JSONL）

../output/QC/<run_name>/                 # 统一 QC 汇总（见「质量控制」节）
└── qc1_label_rt.csv / qc2_roi.csv / qc3_threshold.csv / qc4_snr.csv / qc5_refined.csv
    + qc_summary.md / qc_alert.md
```

> 跑完 pipeline 后自动生成推理报告（`--no_report` 可关闭）：`inference_report_<实验名>.md` 为可读报告（实验名由 `--exp_name` 指定，缺省回退单 mzML→文件名、目录→目录名），`all.csv` 为最终合并明细（每样品×每 ROI 一行），同时为每个样品补算精修峰面积（`prediction_refined/<样品>/prediction_refined_with_area.csv`）。

### 推理 + 后处理设计要点与创新点

1. **信号搜索 + 模型验证的混合整谱扫描（massnova）**——先以纯信号算法（`find_peaks` + 双门槛 prominence + 边界外推）在**全部 transition、完整 RT 轴**上枚举候选峰（不依赖标注、不漏峰），再用模型按峰切 ±1min 窗批量验证并精修边界：验证通过峰用模型框 RT（`boundary_source=model`），未验证峰保留信号外推边界。信号与模型互为补充，回答「整条 XIC 上到底有哪些峰」；
2. **峰边界精修三件套**（`two_round_detection.py`）——边界外推：先走到强度 ≤ 阈值，再要求外推方向 lookahead 点均值 ≤ 阈值×倍数（抑制单点噪声下穿早停）；**peer 防撞**：外推窗口与相邻候选峰重叠且重叠段强度仍高时判定撞入邻峰继续外推；宽度上限约束：≤ min(预测宽×1.08, ROI 跨度×0.45)。解决峰谷粘连时边界互相侵占的问题；
3. **次峰找回 + 谷值回退双通道补峰**（`peak_refinement.py`）——模型漏掉的次要峰/肩峰由信号规则找回（主框左/右侧段按 ROI 规则重搜），或以谷值切分 + 比例折算置信度，解决共流出峰漏检；
4. **五道 QC 防线 + 全防线 `qc_alert.md`**（防线3~5 属本板块：预测阈值、SNR 框级、精修框级）——从标注 → ROI → 预测 → SNR → 精修逐层留痕、逐层剔除并统一汇成人工复核报告，解决「黑盒检出结果无法审计」的问题；
5. **两遍精修防相邻截停**（整谱场景）——首遍无 peer 粗走、第二遍以首遍结果为 peer 区间，消除精修前宽区间互相截停的循环依赖；
6. **像素 → RT 线性映射**——预测框 x 坐标经 `roi_windows.csv` 窗口线性映射回 RT 分钟（y 轴不参与映射），与训练渲染严格一致，规避旧式固定像素假设导致的积分偏移。

***

### 轻量模式

不需要完整管线（SNR 筛选和区间精修）时使用。
注意：`roi` 仅做 EIC/ROI 提取（不输出预测 CSV，也无需 `--model`）；想看预测框标注请两步走：先 `roi` 生成 ROI，再 `roi2inference --plot` 画图；需要完整预测结果请使用 `pipeline`。

```powershell
# 单个 mzML（输出到 <output_dir>/<文件名>/；无需 --model，但 --labels 必填）
python -m inference.cli --mode roi `
  --labels ../data/label/<实验>.xlsx `
  --mzml ../data/mzml/<实验>/<文件>.mzML --output_dir ../output/test/roi_check

# 批量 mzML（目录递归含子目录）
python -m inference.cli --mode roi `
  --labels ../data/label/<实验>.xlsx `
  --batch_dir ../data/mzml/<实验> --output_dir ../output/test/roi_check

# 对已有 ROI 目录批量预测+积分（--plot 时同时生成预测框标注图；roi2inference 无需 --labels）
python -m inference.cli --mode roi2inference `
  --model checkpoint/quanformer.pth `
  --batch_dir ../output/test/roi_check --output_dir ../output/test/pred_check --plot
```

> **输出目录约定**：所有推理输出统一写到 `../output/`（相对 `model/` 目录）。各模式 `--output_dir` 默认值：
>
> | 模式              | 默认 `--output_dir`                       | 说明                                   |
> | --------------- | --------------------------------------- | ------------------------------------ |
> | `roi`           | `../output/inference/xic_roi`           | EIC/ROI 提取产物，可供 `roi2inference` 模式复用 |
> | `roi2inference` | `../output/inference/predictions_model` | 对已有 ROI 目录的批量预测                      |
> | `pipeline`      | `../output/inference/<实验模式>_<实验名>` | 完整管线（ROI + 预测 + SNR + 精修）；实验名由 `--exp_name` 指定或自动回退 |
> | `massnova`      | `../output/inference/massnova`         | 整谱全峰扫描                               |
>
> **测试/试跑请显式指定** **`../output/test/<名称>`** 单独存放，不与正式产物混放。

***

### 推理参数速查

> 以下为 `model/inference/cli.py` **全部**命令行参数（与 argparse 定义一一对应）。
> 「完整参数模板」可直接复制到终端，按注释填写/删减；除 `--model`（roi 外必填）与 `--labels`（roi/pipeline 必填）外，其余参数均可省略（使用默认值）。

**完整参数模板**（注释即填写说明）：

```powershell
python -m inference.cli `
  # ==================== 基础参数 ====================
  # 运行模式（默认 pipeline），可选：roi / roi2inference / pipeline / massnova
  --mode pipeline `
  # 【必填】模型权重 .pth 路径（相对 model/ 目录；roi 模式非必填，massnova 可选）
  --model checkpoint/quanformer.pth `
  # 【必填】人工标注 xlsx（roi/pipeline 模式标注驱动生成 ROI；massnova 模式忽略）
  --labels ../data/label/<实验>.xlsx `
  # 模型峰阈值（默认 0.5；仅 score > 0.5 的模型框通过）
  --threshold 0.5 `
  # 积分方式（默认 linear）：linear / raw / external_baseline
  --integration_method linear `
  # 高斯平滑 sigma（默认 0.0，越大峰越平滑但可能合并近邻峰）
  --smooth_sigma 0.8 `
  # 输出目录（默认按模式自动生成，见「输出目录约定」）
  --output_dir ../output/pipeline_batch `
  # 实验名（用于输出目录与推理报告名 inference_report_<实验名>.md；缺省自动回退）
  --exp_name <实验名> `
  # [roi / pipeline] 输入 mzML 文件路径，或包含 mzML 的目录（递归扫描）
  --mzml ../data/mzml/<实验>/<文件>.mzML `
  # [roi / pipeline] mzML 目录（递归扫描含子目录）；
  # [roi2inference] testXIC 输出目录
  --batch_dir ../data/mzml/<实验> `
  # [pipeline/roi2inference] 生成预测可视化图（roi 模式不支持）
  --plot `
  # [pipeline/roi2inference] 预测图型：xic=XIC 曲线+多 query 阴影（默认）/ roi=ROI 原图叠红框
  --plot_style xic `
  # ==================== Pipeline QC 参数 ====================
  # [QC] XIC 平滑后最大强度低于此值 → 不生成 ROI（默认 1000；0=关闭）
  --pipeline_min_max_intensity 1000 `
  # [QC] 单条色谱 RT 点数少于此值 → 剔除（默认 10；0=关闭）
  --pipeline_min_chrom_points 10 `
  # [QC] 标注 RT 一致性阈值 min（默认 1.0；0=关闭）
  --qc_label_rt_tol 1.0 `
  # ==================== SNR 筛选参数 ====================
  # 框外 SNR 最低阈值（默认 10.0，越高要求信噪比越严）
  --snr_min 10.0 `
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
  # 精修后最低 SNR（默认 10.0）
  --post_min_snr 10.0 `
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
  # 跑完后不自动生成推理报告（默认生成 inference_report_<实验名>.md + all.csv）
  --no_report `
  # SNR 筛选时生成 snr_kept/snr_dropped/ 红框标注 jpeg（默认关闭，省磁盘）
  --save_snr_jpeg `
  # INFO 级日志 / 仅 ERROR 日志（默认 WARNING）
  --verbose / --quiet `
  # 用 JSON 配置作为默认参数（CLI 显式传参优先）：
  #   pipeline 用 configs/inference_pipeline.json，massnova 用 configs/massnova.json
  --config configs/inference_pipeline.json
```

> massnova 模式还有独立的一组 `--scan_*` 参数（基线分位/模式、prominence、峰宽、边界截停、SNR 门、验证窗口等，默认见 [configs/massnova.json](model/configs/massnova.json)），此处不再逐条展开。

**常用参数速查**：

**通用参数**：

| 参数                     | 默认               | 说明                                                     |
| ---------------------- | ---------------- | ------------------------------------------------------ |
| `--model`              | 必填（roi 除外）       | 模型 `.pth` 路径                                           |
| `--labels`             | 必填（roi/pipeline） | 人工标注 xlsx（标注驱动 ROI；massnova 忽略）                        |
| `--mode`               | `pipeline`       | 运行模式：`roi` / `roi2inference` / `pipeline` / `massnova` |
| `--threshold`          | `0.5`            | 模型峰阈值（严格 score > threshold）                        |
| `--integration_method` | `linear`         | `linear` / `raw` / `external_baseline`                 |
| `--smooth_sigma`       | `0.8`            | 高斯平滑 sigma                                             |
| `--plot`               | —                | 生成预测框标注图                                               |
| `--plot_style`         | `xic`            | 预测图型：`xic`（XIC 曲线）/ `roi`（原图叠框）                       |
| `--exp_name`           | 自动回退             | 实验名（报告名/输出目录用）                                        |
| `--output_dir`         | 按模式自动            | 输出目录                                                   |
| `--config`             | —                | JSON 配置作为默认参数（CLI 显式传参优先）                             |
| `--verbose` / `--quiet` | —                | INFO / 仅 ERROR 日志（默认 WARNING）                        |

**Pipeline 参数**：

| 参数                             | 默认     | 说明                  |
| ------------------------------ | ------ | ------------------- |
| `--snr_min`                    | `10.0` | 框外 SNR 最低阈值         |
| `--pipeline_min_max_intensity` | `1000` | XIC 最大强度低于此值跳过      |
| `--pipeline_min_chrom_points`  | `10`   | 色谱点数少于此跳过           |
| `--qc_label_rt_tol`            | `1.0`  | 标注 RT 一致性阈值（min）    |
| `--post_min_confidence`        | `0.99` | 精修后最低置信度            |
| `--no_timing`                  | —      | 不写计时日志文件（终端仍打印）     |
| `--no_report`                  | —      | 跑完后不自动生成推理报告        |
| `--save_snr_jpeg`              | —      | SNR 筛选生成红框标注图（默认关闭） |

***

### 整谱全峰扫描（massnova）

**适用场景：MassNova 集成 / 仅提供时序数据**（无标注）。对全部 transition 通道做信号级找峰（基线分位 / prominence / 边界截停等 `scan_*` 参数），`--model` 提供时按候选峰切 ±1min 窗口做模型验证。参数模板：`configs/massnova.json`。

```powershell
# 纯信号处理（不跑模型）
python -m inference.cli --mode massnova `
  --batch_dir ../data/mzml/<实验> --output_dir ../output/test/massnova_01

# 带模型验证（候选峰切窗推理）
python -m inference.cli --mode massnova `
  --model checkpoint/quanformer.pth `
  --batch_dir ../data/mzml/<实验> --output_dir ../output/test/massnova_01
```

***

### Untargeted 模式

> ⚠️ Untargeted 模式暂不开发。原 `getFeature.py`（R + CentWave 特征提取）已从仓库中删除，如需恢复请从 git history 找回。仅做 MRM（Targeted × Centroided）定量可跳过本节。

***

## 训练

### 生成 COCO 数据集

训练要求数据集为 COCO 格式，由 `preprocessing/coco_annotation.py` 从 **mzML + 标注 xlsx 联合生成**（EIC 图像与推理管线完全一致：400x300、标注 RT ±1min 窗口）：

```powershell
cd model

# 方式一：用现成配置（推荐；数据规模/划分比例见 configs/coco_annotation.json）
python -m preprocessing.coco_annotation --config configs/coco_annotation.json

# 方式二：命令行直接指定
python -m preprocessing.coco_annotation `
  --mzmls <mzML 文件...> `
  --labels ../data/label/<实验>.xlsx `
  --output_dir ../data/coco/<数据集名> `
  --val_ratio 0.3 `
  --force                    # 强制重新提取 XIC（忽略已有 _xic 缓存；B 范式切换后必须 --force 重建）
```

> 参数说明：`--labels` 缺省/`auto` 时按 mzML stem 前缀自动匹配 `--label_dir`（默认 `../data/label`）下的 `<实验>.xlsx`；`--val_stems` 按文件划 val、`--val_ratio` 图像级随机分层划 val（二选一，val_stems 优先）；输出为 `<output_dir>/{train,val}/{train_coco,val_coco}.json` + JPEG，XIC 中间产物在 `<output_dir>/_xic/<stem>/`（再次构建时复用，`--force` 才重新提取）；QC 表写到 `--qc_root`（默认 `../output/QC`）下 `coco_<数据集名>/`。

**ROI 由标注驱动（B 范式）**：每行标注 `(compound, channel)` 经 `label_key` → `native_id`「化合物名-1/-2」匹配 mzML 色谱，**窗口中心 = 标注** **`rt`** **字段**（非谱图最高强度点）；未标注的 mzML 通道不生成 ROI。标注格式（`data/label/<trial>.xlsx`，多峰）：

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

# 方式一：用现成配置（推荐；quanformer 基线 / v2 微调 / v3 微调 / v1 FDR / v1 多源，见下表）
python -m train --config configs/quanformer_baseline.json

# 方式二：命令行直接指定
python -m train `
  --model quanformer `              # 模型变体：quanformer / mrmpformer_v1
  --coco_path ../data/coco/<数据集名> `
  --output_dir ../output/train/<实验名> `
  --device auto `
  --epochs 30 --batch_size 4 `
  --lr 1e-4 --lr_backbone 1e-5
```

### 模型架构与设计创新（MRMPFormer v1）

在 DETR（ResNet-50 + Transformer）基线上，MRMPFormer v1 针对**色谱峰检测的特殊性**做了针对性改进：

1. **1 Encoder + 3 Decoder 残差级联精化**——Encoder 保持 1 层（与 QuanFormer 同构，旧权重可直接迁移），Decoder 扩为 3 层级联：第 2/3 层只输出对边界的**残差偏移**（`z_{k+1}=z_k+Δz`，Δz=0 时为恒等映射），最终框取「第 3 层左右边界 + 第 1 层上下边界」。解决单层回归头对窄峰/肩峰边界预测粗糙的问题；
2. **FDR（Fine-grained Distribution Refinement，边界逐层精化）**——每层独立 FDR Head 输出左右边界的非均匀 Bin 概率分布（33 bins，密集中间、稀疏两端），用「分布期望偏移」解码为边界调整量，全程可导；Head 末层 Linear **零初始化**，训练初期从恒等映射起步、不破坏初始框学习；
3. **边界位置反馈通路**——每层精化出的边界经 `BoundaryPositionMLP` 编码进下一层的 `query_pos`，让深层 Decoder 聚焦「修正边界」而非重复首层检测；
4. **峰专用损失**——Softmax Focal（+eos_coef 背景降权，防峰/背景样本严重不均衡下分类头退化为恒背景）+ **动态 L1**（λ_c=1/w_gt，窄峰中心权重更大）+ **PW-CIoU**（中心距离按峰宽加权，窄峰惩罚更重，权重取训练集统计而非 mini-batch 均值）。解决农残峰宽差异大时小宽峰被「平均化」的问题；
5. **num_queries=3 多峰检出**——单 ROI 最多同时检出 3 个峰，与多峰标注 `peak_start1-3` 对齐，并配套 `gt_exceeds_queries_roi_ratio` 等诊断指标监控 GT 超限比例；
6. **多源联合训练热启动**（mrmpformerv2）——用 v1 架构在 traindata3 + shiyaoyuan_1 + traindata2 + test1 合并数据集（13929 峰）上训练，`resume` 自 v1_fdr 权重 + `reset_optimizer`，解决单源数据过拟合、提升跨仪器/样品域泛化。

### 参数配置外置（--config，推荐）

所有入口（`train.py` / `inference.cli.py` / `preprocessing/coco_annotation.py` / `tools.evaluation.evaluate_baseline.py`）都支持 `--config configs/<名称>.json`：配置文件提供**默认参数**，CLI 参数仍可覆盖；`_comment_*` 键为注释不生效。现成配置（均在 `model/configs/` 下）：

| 配置                                                         | 用途                                                              |
| ---------------------------------------------------------- | --------------------------------------------------------------- |
| `quanformer_baseline.json`                                 | quanformer 基线，从零训练（`coco_path: ../data/test/coco`）              |
| `quanformer_v2_finetune.json`                              | quanformer 微调（`resume: checkpoint/quanformer.pth`，产出 quanformerv2）  |
| `quanformer_baseline_test1ft.json`                         | 对照实验：test1 数据微调基线，验证能否追平 v1_fdr                              |
| `quanformer_v3_finetune.json`                              | quanformer 第 3 轮微调（B 范式合并数据集 merged，产出 quanformerv3）          |
| `mrmpformer_v1_fdr.json`                                   | MRMPFormer v1（traindata3，FDR 边界精化全开，产出 mrmpformerv1）            |
| `mrmpformer_v1_multisrc.json`                              | MRMPFormer v1 多源联合训练（resume 自 v1_fdr，产出 mrmpformerv2）           |
| `inference_pipeline.json`                                  | 完整推理管线（pipeline 模式）参数模板                                         |
| `massnova.json`                                       | 整谱全峰扫描（massnova 模式）参数模板                                         |
| `evaluation_baseline.json`                                 | 基线一键精度评测参数                                                      |
| `coco_annotation.json` / `coco_annotation_traindata3.json` | COCO 数据集构建参数（10 文件合并 / traindata3 全量）                           |

示例：`python -m train --config configs/mrmpformer_v1_fdr.json`

> MRMPFormer v1 的 FDR 结构参数（`num_fdr_bins` / `fdr_bin_power` / `fdr_bin_values` / `fdr_scale_mode` / `fdr_layer_weights` / `fdr_loss_coef` / `fdr_min_width` / `detach_boundary_feedback`）与分类/定位损失参数（Focal、动态 L1、PW-CIoU）均已注册到 `train.py` 的 argparse，配置文件与 CLI 均可控制。

### 关键参数

| 参数                              | 默认                     | 说明                    |
| ------------------------------- | ---------------------- | --------------------- |
| `--coco_path`                   | `data/coco`            | 数据集根目录                |
| `--device`                      | `auto`                 | CUDA > MPS > CPU 自动选择 |
| `--epochs`                      | `30`                   | 训练轮数                  |
| `--batch_size`                  | `4`                    | 批大小                   |
| `--lr` / `--lr_backbone`        | `1e-4` / `1e-5`        | 学习率                   |
| `--enc_layers` / `--dec_layers` | `1` / `1`              | Transformer 编/解码器层数   |
| `--num_queries`                 | `10`（configs 统一为 `3`） | 最大检出峰数                |
| `--resume`                      | `checkpoint.pth`       | 从检查点恢复                |
| `--eval`                        | —                      | 仅评估，不训练               |

### 微调与验证（`--eval` 仅在验证集上跑损失/指标，区别于下方「评估」板块的人工标注评测）

```powershell
# 从预训练权重继续训练
python -m train `
  --coco_path ../data/coco/<数据集名> --output_dir ../output/train/<实验名> `
  --device auto --resume checkpoint/quanformer.pth --epochs 50

# 仅评估
python -m train `
  --coco_path ../data/coco/<数据集名> `
  --resume checkpoint/quanformer.pth --device auto --eval
```

> ⚠️ 恢复训练仅在 checkpoint 与当前模型**同名字段维度不一致**时才跳过该权重（如 COCO 预训练迁移场景）；自训 checkpoint 续训/微调时分类头与 `query_embed` 完整加载，不会静默随机初始化。
> 💡 从旧 QuanFormer 单层权重热启动 MRMPFormer v1 时，自动走 `load_legacy_quanformer_state` 迁移：decoder L1 参数复制初始化到 L2/L3，FDR/边界反馈模块保持新初始化。
> 💡 当前 `quanformer.pth` 的训练参数：`enc_layers=1, dec_layers=1, num_queries=3, hidden_dim=256, nheads=8, dim_feedforward=2048, dropout=0.1`。

***

## 评估

评估板块用**预测结果 vs 人工标注**计算模型效果，覆盖检测与定量两个口径（入口 `model/tools/evaluation/evaluate_baseline.py`）。

### 评估口径

| 维度 | 指标 | 说明 |
| --- | --- | --- |
| 检测 | P / R / F1 | 预测起止与人工起止偏差**均 ≤ ±0.1 min** 判命中；按置信度降序贪心匹配，同一标注至多匹配一个预测；未匹配预测=FP、未匹配标注=FN |
| 定量 | 面积 R² | TP 对预测面积 vs 人工面积的散点线性拟合（准确度） |
| 定量 | RSD | 同化合物通道跨进样预测面积的重复性 std/mean（重现性） |
| 定位 | RT 偏差 | 预测起止与人工起止偏差的均值/中位数（分钟） |

### GT 构建与 QC 联动

- **GT 与训练标注同源**：GT 直接从标注多峰格式重建（`peak_start1-3` / `peak_end1-3` + `area1-3`），`peak_label=0`（负样本/空白）行不产 GT，避免 RT=0 伪峰——评估与训练共用同一份标注；
- **QC 联动**：防线 1（标注 RT 一致性）剔除的 (sample, compound, channel) 整通道退出评估——GT 不计 FN、该通道预测不计 FP，标注错误不会被算成模型漏检/误检；
- **默认口径**：读 `model_prediction_<样品>.csv`（模型原始检测能力，不含 SNR 筛选/精修后处理）；需评估整条管线最终产物时改读 `prediction_refined.csv`。

### 评估命令

```powershell
cd model

# 方式一：复用已有推理产物评估（--run_inference 0，逐样品指定 prediction/feature CSV）
python -m tools.evaluation.evaluate_baseline `
  --labels ../data/label/<实验>.xlsx `
  --run_inference 0 `
  --prediction_csvs <样品1>=../output/<run>/predictions_model/<样品1>/model_prediction_<样品1>.csv <样品2>=...
  --feature_csvs  <样品1>=../output/<run>/xic_roi/<样品1>/feature.csv <样品2>=...
  --output_dir ../output/eval/<实验名>

# 方式二：一键式（默认先调 inference.cli pipeline 推理，再评估）
python -m tools.evaluation.evaluate_baseline `
  --labels ../data/label/<实验>.xlsx `
  --mzmls ../data/mzml/<实验>/<文件>.mzML `
  --model checkpoint/quanformer.pth `
  --output_dir ../output/eval/<实验名> `
  --config configs/evaluation_baseline.json
```

产物：`evaluation_report.json`（P/R/F1、面积 R²、RT 偏差、RSD）、`match_details.csv`（逐对匹配明细）、`area_pairs.csv`（面积配对）。

### 评估创新点

1. **双口径两轮配对**——严格轮（±0.1 min）决定 P/R/F1；宽松轮（±0.2 min）仅用于面积 R²/RSD/RT 偏差，且宽松配对的预测在检测口径仍分别计 FP 与 FN，避免「定量配对了但检测没检出」的口径混用；
2. **命中判据直接对标人工标注语义**——用「起止偏差 ≤ 容差」而非 IoU 阈值判定命中，更贴近「人工画的峰区间」这一标注形态；
3. **GT 由多峰标注直接重建**——与训练 bbox 同源、负样本自动不产 GT，保证评估 GT 与训练标注零口径差、无伪峰；
4. **QC 剔除与评估全联动**——被剔通道整通道退出（GT 不计 FN、预测不计 FP），模型指标不被脏标注污染；
5. **准确度 + 重现性 + 定位三维定量口径**——面积 R²、RSD、RT 偏差三组指标互补，覆盖定量能力的不同侧面。

***

## 项目结构

```
MRMPFormer/
├── requirements.txt              # pip 依赖（model + desktop 合并，GPU 分段：RTX 50 / RTX 40+30+20 / CPU）
├── environment.yml               # Conda 环境（name: gamstekpeaking；在仓库根目录使用）
├── model/                        # ⭐ 核心代码（须在此目录下用 python -m <pkg> 调用）
│   ├── train.py                  # 训练入口（python -m train ...）
│   ├── configs/                  # 参数配置文件（--config 外置默认参数：训练/推理/评测/数据集构建）
│   ├── inference/                # 推理（含后处理前半段）：统一入口 cli.py + 预测器 + 整谱全峰扫描 + 边界精修
│   │   ├── cli.py                #   统一推理入口（python -m inference.cli --mode ...）
│   │   ├── predictor.py          #   单张/批量预测 + 逐框积分
│   │   ├── massnova.py           #   整谱全峰扫描（信号搜索 + 模型验证，massnova 模式）
│   │   └── two_round_detection.py #  峰边界精修（边界外推 / peer 防撞 / 次峰找回）
│   ├── models/                   # 模型定义（quanformer / mrmpformer_v1(FDR 边界精化) / shared 骨干与位置编码）
│   ├── preprocessing/            # 前处理：标注 QC（label_qc）+ EIC/ROI 提取（xic_extraction）+ 数据集构建（coco_annotation）
│   ├── postprocessing/           # 后处理（并入推理板块）：SNR 筛选 / 峰区间精修 / 面积积分 / 谷值切分
│   ├── framework/                # DETR 训练框架（datasets / util / engine / hubconf）
│   ├── utils/                    # 推理辅助（io / quantify / mzml_load / roi_rt_mapping / adaptive_integration 等）
│   ├── tools/                    # 工具集：evaluation（评估板块）/ batch / diagnostics / experiments / maintenance / mzml / benchmark / visualization / tests
│   ├── checkpoint/               # 模型权重（quanformer.pth 基线、quanformerv2/v3.pth、mrmpformer.pth=v3 最终、mrmpformer_special_v2.pth）
│   └── tests/                    # 单元测试
├── desktop/                      # PySide6 桌面 GUI（pages：前处理 / 寻峰 / 设置；workers：转换 / 离子天顶）
├── converters/                   # 格式转换（msdata/wiff → mzML；含运行时 msdata_bin/ wiff_bin/）
├── data/                         # 数据（label/ 标注 xlsx、msdata/、wiff/、mzml/；整体 .gitignore）
├── docs/                         # 项目文档（plan / spec / 实验报告 / Bugs 等）
├── blank_label.md                # 📋 所有生成 CSV/表格的用途与字段含义字典
├── User_Tutorials.md             # 四种分析模式操作教程
└── dev_log.md                    # 开发日志
```

***

## 辅助工具

### converters — 格式转换

支持两种厂商格式 → 标准 `.mzML`：

```powershell
cd converters

# msdata → mzML（OpenMS 工具链）
python msdata.py         # 批量转换（--dry-run 仅预览）

# wiff → mzML（ProteoWizard 工具链）
python wiff.py           # 批量转换（--no-peak-picking 保留 profile 轮廓）
```

> 要求对应 `*_bin/` 包含运行时依赖，项目路径不得含中文。

### 工具集（`model/tools/`）

| 子模块              | 用途                           | 入口示例                                                                                                                                      |
| ---------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `batch/`         | 批量重处理（SNR / post）            | `python -m tools.batch.reprocess --stage snr`                                                                                             |
| `mzml/`          | 色谱图查看/导出/检查                  | `python -m tools.mzml.chromatogram list <file>`；`python -m tools.mzml.inspect --mzml <file>`                                           |
| `benchmark/`     | 性能基准测试                       | `python -m tools.benchmark.runner --help`                                                                                                 |
| `visualization/` | 可视化（GT vs 预测 / RT / 精修 XIC） | —                                                                                                                                         |
| `evaluation/`    | 推理报告 / 框修正消融 / 基线评测 / 对比 | `python -m tools.evaluation.inference_report --output_dir ../output/inference/pipeline_demo --exp_name demo`；`python -m tools.evaluation.refine_ablation`；`python -m tools.evaluation.evaluate_baseline` |
| `diagnostics/`   | 诊断脚本（RT 轴核对 / 案例证据导出等）       | `python -m tools.diagnostics.verify_rt_axis`                                                                                             |
| `experiments/`   | 面积对比实验（江南 / 欧陆）              | —                                                                                                                                         |
| `maintenance/`   | 维护（空白负样本强制 / 结果整理 / XIC 重生成） | —                                                                                                                                         |
| `tests/`         | 测试脚本                         | —                                                                                                                                         |

***

## 平台注意事项

| 平台          | 注意事项                                                     |
| ----------- | -------------------------------------------------------- |
| **Windows** | 路径避免空格（影响 R 调用）；`pycocotools` 可能需 Visual C++ Build Tools |
| **Linux**   | 无桌面环境运行 GUI 需 X11 转发（`ssh -X`）或 `xvfb`                   |
| **macOS**   | Apple Silicon 自动使用 MPS 加速；Intel Mac 仅 CPU                |

***

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

***

> 更多细节：[CSV 字段字典](blank_label.md) · [已知问题](docs/Bugs.md) · [模式操作教程](User_Tutorials.md) · [开发日志](dev_log.md)

> 更多细节：[项目全景](docs/PROJECT_PANORAMA.md) · [已知问题](docs/Bugs.md) · [跨平台部署](docs/MRMPFormer%20跨平台部署指南.md)

---

## MRMPFormer C/C++ 共享库

`cpp/` 提供不依赖 Python 的 C++20 共享库，使用 C ABI（公共头文件：
[`cpp/include/mrmpformer.h`](cpp/include/mrmpformer.h)）调用
`model/checkpoint/mrmpformerv2.onnx`。输入使用 transition 的 `mzq1`、`mzq3`
和 `channel`，ONNX 使用 `scores` 与 `boxes_xyxy` 输出；结果中的
`SIGNAL_FALLBACK`、`CHANNEL_LOW_INTENSITY` 和 `status/alerts` 来自 C++ 的
QC/后处理，而非 ONNX 网络。

构建、部署、纯 C 示例和内存/回调约定见 [cpp/README.md](cpp/README.md)；从
QuanFormer 迁移时请先阅读 [C API 差异与迁移指南](docs/CPP_API_DIFFERENCES.md)。
