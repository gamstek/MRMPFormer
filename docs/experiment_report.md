# 实验报告（Experiment Report）

> 记录模型实验的假设、证据、结论与影响。每篇实验独立成节，按时间倒序排列。
> 数据与命令均可复现，结论均给出证据出处。

---

## 实验日志 001：v1「先成功后失败」之谜 —— 推理取类 bug 与 shadow query 假象

- **日期**：2026-08-17
- **涉及模型**：quanformer.pth（v1，原基线权重，未改动）、quanformerv2.pth（v2，data/test/coco 微调）
- **状态**：根因已定位并修复；量化证据已归档；逐 query 铁证待外部运行（附录 C）

### 1. 现象

同一份 v1 权重、同一批测试数据（20260715 两次进样 ×60 通道），先后两次评测结果天壤之别：

| 评测时间 | 检测口径结果 | 定量口径结果 | 当时结论 |
|---|---|---|---|
| 2026-08-16（bug 未修复） | tIoU>0.95：P/R/F1 = 0.025/0.025/0.025（TP=3） | 面积 R² = 0.99998（n=115）、RT 起止偏差中位 0.063/0.073 min、RSD 中位 1.99% | 「定位与定量能力优秀，检测分低是框偏宽」 |
| 2026-08-17（bug 修复后，±0.1 min 容差口径） | TP=1 / FP=121 / FN=119，F1 = 0.008 | 面积对仅 1 对（无统计意义） | 「v1 检测全灭」 |

同一权重为何前后判若两模？v1 到底是变差了，还是从未好过？

### 2. 根因

**v1 权重从未变过，变的是「每张图选中哪个 query 的框」。** 源头是 `model/utils/predict_utils.py` 的取类 bug：

```python
# bug（修复前）：logits 布局为 [背景, 峰]，[:-1] 取到的是背景概率列
probas = pred_logits.softmax(-1)[0, :, :-1]   # ← 背景概率
# 修复后：排除背景类，取真实类别列
probas = pred_logits.softmax(-1)[0, :, 1:]    # ← 峰概率
```

模型输出 `num_queries=3` 个候选框，`score > threshold` 决定保留哪个。修复前后选择标准正好**互补**：

- **修复前**：留下的是「模型自认为背景」的 query 的框（shadow query）
- **修复后**：留下的是「模型自信认为是峰」的 query 的框

### 3. 证据

#### 3.1 新旧框对比（122 条，同权重/同图/同 GT，出处 `data/evaluation/quanformer/match_details.csv` × `v1_fixed/match_details.csv`）

| 指标 | 旧框（bug 选中，背景 query） | 新框（修复后选中，峰 query） | GT |
|---|---|---|---|
| 起点 vs GT 起点 | **−0.051 min**（中位 −0.061） | **+0.697 min**（中位 +0.665） | — |
| 框宽 | **0.468 min** | 0.241 min | 0.470 min |

抽样示例（同一条通道、新旧是**两个完全不同的框**）：

```
阿维菌素-1   GT[16.43, 16.97] | 旧[16.43, 17.11] | 新[17.29, 17.58]
乙酰甲胺磷-2  GT[ 2.60,  3.21] | 旧[ 2.56,  2.85] | 新[ 3.52,  3.61]
灭螨醌-1     GT[ 6.13,  6.49] | 旧[ 6.03,  6.49] | 新[ 6.73,  6.99]
```

#### 3.2 机理解读

1. **v1 的「背景 query」框反而贴合人工边界**（−0.05 min、框宽≈GT）。DETR 单目标场景下，未被匈牙利匹配选中的 shadow query 仍会回归到目标附近。旧 bug 阴差阳错选中了它——**之前 v1 所有好看的定量数字（R²=0.99998、RT 偏差 0.06 min）全部来自这个 shadow 框**，且 prediction.csv 里的 score≈0.99 实为背景概率，并非峰置信度。
2. **v1 的「峰 query」框才是其真实检测水平**：整体晚 +0.70 min、框宽仅 GT 一半。说明 v1 原训练数据的 bbox 约定与本项目人工积分约定差异巨大。修复取类后暴露的是真面目。
3. **v1 的检测其实从来没好过**：bug 时代 tIoU>0.95 口径 F1 也只有 0.025——旧评测中「检测差」与「定量好」的矛盾，正是「峰 query 框差 × shadow query 框好」这对矛盾体的投影。
4. **v2 不受影响的原理**：微调数据（data/test/coco）的 bbox 直接映射自人工 `peak_start/peak_end`，微调把「峰 query」的框校准到了人工约定上（起点偏差均值 −0.017 min、tIoU 中位 0.72）——自信与正确终于统一。

#### 3.2.1 逐 query 铁证（2026-08-17 用户外部运行 `tools/evaluation/dump_queries.py`）

诊断脚本直接前向模型，打印每张 ROI 图全部 3 个 query 的 `P(峰)/P(背)` 与框偏差。v1 抽样输出：

```
阿维菌素-1  GT[16.428, 16.969]
  q0  P(峰) 0.990  P(背) 0.010  框RT[16.484, 16.714]  起偏 +0.06 / 止偏 -0.26
  q1  P(峰) 0.000  P(背) 1.000  框RT[16.435, 17.113]  起偏 +0.01 / 止偏 +0.14   ← 旧bug选中(背>0.9)
  q2  P(峰) 0.995  P(背) 0.005  框RT[17.295, 17.577]  起偏 +0.87 / 止偏 +0.61   ← 新代码选中(峰>0.9)
```

末尾统计（14 张有 GT 图，按起止总偏差比较）：**背概率最高 query 的框更贴 GT：13 / 14**，峰概率最高 query 仅 1 张——假设坐实。

**v1 的三条 query 分工（系统性规律）**：

| query | P(峰) | 框行为 | 角色 |
|---|---|---|---|
| q0 | ~0.99 | 窄框（宽 ~0.23 min），贴峰左/近 GT | v1 旧训练「紧框包 apex」约定的实例 |
| q1 | ~0.00 | **GT 宽度**（~0.61 min）的宽框，起点几乎零偏 | 唯一符合人工积分约定的框；旧 bug 捡到的就是它 |
| q2 | ~0.99 | 窄框，右偏 +0.8 min | 同一错误约定的另一实例化 |

**用户质疑与解答**：阿维菌素-1 中 q0 显然比 q2 贴 GT，为何最终输出 q2？答案在选框代码（`model/inference/predictor.py`）：

```python
top_idx = int(np.argmax(scores[:, 0]))   # 每图只保留「峰概率 argmax」的那一个 query
top_score = scores[top_idx:top_idx + 1]
top_box = boxes[top_idx:top_idx + 1]
```

- q0（P=0.990）与 q2（P=0.995）都过了 0.9 阈值（dump 均标「新代码选中」），但下游每图只取 argmax——q2 胜出。
- 铁证：prediction.csv 中阿维菌素-1 行为 `[17.2948, 17.5770]`、score=0.99463，与 q2 的框逐位吻合（而非 q0）。
- **更深一层**：v1 的置信度排序与框质量反相关（错 1.5 min 的 q2 比错 0.3 min 的 q0 更自信 0.005，阈值提到 0.99 也无法仲裁）。推理端无论 argmax / NMS / 提阈值，都选不出「它没学过的约定」——**这不是选框策略问题，是模型权重本身的问题**，只能靠重训/微调（v2 路线）解决。
- 顺带澄清：q0 也并非「好框」（止边偏 −0.26 min，±0.1 容差下仍不 TP），它只是矮子里拔将军。

#### 3.3 v1/v2 统一口径复测（±0.1 min 起止容差，score≥0.90，2026-08-17）

| 指标 | v1（修复取类后） | v2（微调） |
|---|---|---|
| TP / FP / FN | 1 / 121 / 119 | 55 / 67 / 65 |
| P / R / F1 | 0.008 / 0.008 / 0.008 | **0.451 / 0.458 / 0.455** |
| 面积 R² | —（n=1） | **0.99999（n=106）** |
| RT 起止偏差中位 | — | 0.059 / 0.080 min |
| RSD 中位 | — | 1.91%（n=48） |

报告存档：`data/evaluation/v1_fixed_dev/`、`data/evaluation/v2_fixed_dev/`。

### 4. 结论

1. **v1 从未真正成功过。** 此前的「定量成功」是取类 bug 借 shadow query 之手制造的假象；修复后 v1 检测 F1=0.008 才是其真实水平。
2. **v2 的提升（F1 0.008 → 0.455，面积 R² 保持 0.99999）是真实的**，且是在正确取类口径下取得。
3. **2026-08-16 基线参考分数中 v1 的定量指标作废**（来源框非法）；检测指标（F1=0.025）仍有效但需注明为 shadow 框口径。improve.md 第 3 项的基线表已不适用，应以本报告 3.3 节为准。
4. **方法论教训**：
   - DETR 类多 query 模型评测时，必须核验「选框依据」与「置信度语义」是否一致（本例 score 是背景概率却当峰置信度用）；
   - 单一指标好（面积 R²）不等于流程正确，需多口径交叉验证；
   - 修复取类 bug 应视为**评测口径修正**，v1 修复前后不是「模型变差」，是「显形」。

### 5. 影响范围与后续动作

| 项 | 处置 |
|---|---|
| `predict_utils.py` 取类修复 | 已完成（`[1:]`），v1/v2 推理统一为峰概率列 |
| 旧基线分数（improve.md 第 3 项） | 需按 3.3 节口径重录 |
| 逐 query 铁证 | 附录 C 命令待外部运行，预期「背景概率最高 query 的框更贴 GT」占多数即坐实 |
| v2 剩余提升空间 | 起止残留偏差 0.05~0.08 min（±0.1 容差边缘），可试：容差敏感性扫描（0.08/0.1/0.12/0.15）、更大 lr 或更多 epoch 微调、bbox loss 加权 |

---

### 附录 A：事件时间线

| 日期 | 事件 |
|---|---|
| 2026-08-16 | v1 首次基线评测：定量指标优秀、检测 F1=0.025（shadow 框口径，当时未知） |
| 2026-08-16 | v2 微调两次失败（transforms 错位 → 修；仍 0 检出）→ 定位 `predict_utils.py` 取类 bug（`[:-1]`→`[1:]`），v1/v2 统一修正后推理均恢复 61/61 检出 |
| 2026-08-17 | tIoU>0.95 口径评测：v1 F1=0、v2 F1=0.017（阈值过严，v2 tIoU 中位 0.72） |
| 2026-08-17 | 评测协议改「起止偏差容差」口径（检测 ±0.1 min / 定量宽松 ±0.2 min），删除 tIoU 判据 |
| 2026-08-17 | v1 F1=0.008 vs v2 F1=0.455；本报告根因分析完成 |

### 附录 B：涉及文件

- 修复：`model/utils/predict_utils.py`（取类 `[1:]`）
- 评测协议：`model/tools/evaluation/evaluate_baseline.py`（起止偏差口径 + `--config` 外置）
- 可视化复核：`model/tools/evaluation/visualize_compare.py`（GT/v1/v2 三色叠加画廊）
- 逐 query 诊断：`model/tools/evaluation/dump_queries.py`
- 数据：`data/evaluation/{quanformer,v1_fixed,v1_fixed_dev,v2_fixed_dev}/`、`data/test/pred_{v1,v2}_fixed/`

### 附录 C：逐 query 铁证命令（待运行）

```powershell
cd D:\work\MRMPFormer\model

# v1：每样品前 8 张，打印每个 query 的 P(峰)/P(背) 与框偏差
D:\Anaconda3\envs\gamstekpeaking\python.exe -m tools.evaluation.dump_queries --model checkpoint/quanformer.pth --xic_root ../data/test/coco/_xic --labels ../data/label/20260715_shiyaoyuan_test.xlsx --limit 8

# v2 对照（预期：峰 query 又自信又贴 GT）
D:\Anaconda3\envs\gamstekpeaking\python.exe -m tools.evaluation.dump_queries --model checkpoint/quanformerv2.pth --xic_root ../data/test/coco/_xic --labels ../data/label/20260715_shiyaoyuan_test.xlsx --limit 8
```

末尾统计块若显示「背概率最高 query 的框更贴 GT」显著占优，即为最终铁证；v2 应呈相反格局。

---

## 附录 D：推理管线模式与输出清单（2026-08-18）

> 统一入口 `model/inference/cli.py`（`python -m inference.cli --mode <mode> ...`），3 种模式各阶段输出如下。
> `roi` / `pipeline` 支持 `--mzml` 单文件或 `--batch_dir` 目录（递归含子目录）。

### 模式总览

| 模式 | 用途 | 输入 | 输出 |
|---|---|---|---|
| `roi` | mzML → ROI（仅阶段①） | `--mzml` 文件/目录 或 `--batch_dir` 目录（递归） | `<out>/<key>/` ROI 目录 |
| `batch_dir` | 已有 ROI 目录批量预测 | `--batch_dir`（每子目录一套 ROI） | `<out>/<子目录>/prediction.csv` |
| `pipeline`（默认） | 完整管线①~④（单文件或批量） | 同 `roi` | `base_out/` 四阶段产物 |

> `<key>` = mzML 文件名 stem；目录递归下不同子目录同名 stem 自动改为路径展平（如 `子目录A__样品1`）避免覆盖。

### 各模式详细输出

**① `roi`**：`extract_xic_with_pyopenms` 逐 mzML 读色谱 → `<out>/<key>/`（仅 ROI 生成，无需 `--model`，不支持 `--plot`）：
- 每通道 ROI jpeg（命名 `N_mz{母离子}_q3{子离子}.jpeg`）
- `feature.csv`、`roi_windows.csv`、`xic_matrix.npy`
- 被 QC 剔除的通道记录 `pipeline_qc_excluded.csv`
- 想看预测框标注：`roi` 生成 ROI 后，再对该目录跑 `batch_dir --plot`（模型仅加载一次）

**② `batch_dir`**：复用 `predictor.main()` 批量模式，逐子目录：
- `<out>/<子目录>/prediction.csv`（积分方式非 linear 时为 `prediction_{method}.csv`）
- `<out>/<子目录>/predicted_plots/`（`--plot`）

**③ `pipeline`**（`base_out = --output_dir`，默认 `results/full_pipeline`；单文件与批量统一走批量路径，产物布局一致）：

| 阶段 | 输出目录 | 产物 |
|---|---|---|
| ① ROI 生成 | `base_out/xic-roi-batch/<key>/` | ROI jpeg、`feature.csv`、`roi_windows.csv`、`xic_matrix.npy` |
| ② 模型预测 | `base_out/batch_predictions/<key>/` | `prediction.csv`（或 `prediction_{method}.csv`）、`predicted_plots/`（`--plot`） |
| ③ SNR 筛选 | `base_out/snr_filtered/<key>/SNR_box_<thr>/` | `prediction_snr.csv`（保留行，旧版名 `prediction.csv`）、`feature.csv`、`roi_windows.csv`、`xic_matrix.npy`、`box_outside_snr_report.csv`；`--save_snr_jpeg` 时 `筛选保留/`、`筛选剔除/` 红框标注图 |
| ④ 框修正 | 同上 `SNR_box_<thr>/` | `prediction_refined.csv`（`--post_output_name` 可改）、`refined_plots/`（`--plot`） |
| 计时 | `base_out/` | `pipeline_timing.log`、`pipeline_timing_runs.jsonl`（阶段耗时 + 资源统计） |

### 输出差异说明

- **prediction.csv 列**：image, image_path, compound_name, mz, q3, old_rt, box_x1/y1/x2/y2, score, rt_min, rt_max, retention_time, intensity_max, area, point_counts, snr, noise_std, baseline_slope, peak_width_ratio, dynamic_range, integration_method_used（每 (mz,q3) 只保留面积最大行）。
- **SNR 阶段产物**：`prediction_snr.csv` 为通过 SNR 门槛的检测框（供 post 读取，2026-08-19 起由 `prediction.csv` 更名而来，读方均带旧名回退），`box_outside_snr_report.csv` 为逐框 SNR 明细；两者均含 `image_path` 指向 ROI 图。
- **精修 `prediction_refined.csv`**：为主峰+次峰识别/谷值拆分后的最终峰列表，含框修正后 RT 边界与置信度。
- **积分方法**：cli `--integration_method`（linear/raw/external_baseline）非 linear 时预测文件名为 `prediction_{method}.csv`，不覆盖默认；predictor 底层另支持 peak_adaptive/adaptive/minval_noise_right 等仅供内部调用。
- **QC 门槛**（pipeline 模式）：`--pipeline_min_max_intensity`（默认 1000）、`--pipeline_min_chrom_points`（默认 10），未达标的通道不生成 ROI、不参与预测。

---

## 附录 E：三段式工作流（训练 → 推理 → 评估）（2026-08-19 确认）

> 本项目标准工作流为「训练 → 推理 → 评估」三段式闭环（前置一环为「构建数据集」）。本文档记录各环节的入口、产物与评估口径，作为实验基准。

### E.1 环节总览

| 环节 | 做什么 | 入口 | 产物 |
|---|---|---|---|
| ① 构建数据集 | 标注 xlsx + mzML → COCO 格式 bbox（bbox 直接映射人工 `peak_start/peak_end`） | `model/preprocessing/coco_annotation.py` | `data/test/coco/`（train/val + `train_coco.json`/`val_coco.json`） |
| ② 训练 | COCO 数据集训练/微调 DETR | `python -m train --config configs/quanformer_baseline.json`（从零）/ `quanformer_v2_finetune.json`（微调） | `checkpoint/quanformer.pth`、`quanformerv2.pth` |
| ③ 推理 | 测试 mzML + 已训练模型 → 预测值 | `python -m inference.cli --mode pipeline --model checkpoint/quanformer.pth` | `prediction.csv` → `prediction_snr.csv` → `prediction_refined.csv` |
| ④ 评估 | 预测值 + 人工标注 → 模型效果 | `python -m tools.evaluation.evaluate_baseline --labels ../data/label/20260715_shiyaoyuan_test.xlsx` | `evaluation_report.json`、`match_details.csv`、`area_pairs.csv` |

### E.2 评估口径（重要）

- **评估读阶段②原始 `prediction.csv`**（`evaluate_baseline.py` 取 `batch_predictions/<样品>/prediction.csv`）——衡量**模型原始检测能力**，**不含** SNR 筛选/精修等后处理；
- 指标含双口径：**检测**（RT 容差内匹配的 P/R/F1，`--tiou` 默认 0.95，2026-08-17 后改为起止偏差容差）与**定量**（面积 R²、RT 边界偏差、RSD，`--quant_tiou` 默认 0.5）；
- 若需评估「整条管线最终产物」（含后处理），目标应改为 `prediction_refined.csv`——**两套口径不可混用**（见实验日志 001：同一模型不同口径结果天差地别）；
- **基线分数有效期**：`imporove.md` 第 3 项中的 v1 定量指标已因取类 bug 作废（实验日志 001），现行有效基线见 `data/evaluation/v1_fixed_dev/`、`v2_fixed_dev/`。

### E.3 数据泄漏防护

- 训练/评估分样品：train=`20260715_shiyaoyuan_test_1`、val=`20260715_shiyaoyuan_test_2`（同一批仪器数据两次进样，通道一致）——不在同一样品上既训又评；
- 标注数据集 `data/label/20260715_shiyaoyuan_test.xlsx` 双重身份（训练 bbox 来源 + 评估 GT）为**预期设计**，但新增实验数据时必须维持「训练/评估样品隔离」原则。

### E.4 典型命令

```powershell
cd D:\work\MRMPFormer\model

# ① 构建 COCO 数据集（mzML + 标注 → data/test/coco）
D:\Anaconda3\envs\gamstekpeaking\python.exe -m preprocessing.coco_annotation --help

# ② 训练（从零 / 微调）
D:\Anaconda3\envs\gamstekpeaking\python.exe -m train --config configs/quanformer_baseline.json
D:\Anaconda3\envs\gamstekpeaking\python.exe -m train --config configs/quanformer_v2_finetune.json

# ③ 推理（完整管线，输出 ../output/test/<名称> 测试目录）
D:\Anaconda3\envs\gamstekpeaking\python.exe -m inference.cli --mode pipeline --config configs/inference_pipeline.json --model checkpoint/quanformer.pth --mzml ..\data\test\mzml\20260715_shiyaoyuan_test_1.mzML --output_dir ..\output\test\eval_check

# ④ 评估（--run_inference 1=先跑推理；0=复用已有 prediction.csv）
D:\Anaconda3\envs\gamstekpeaking\python.exe -m tools.evaluation.evaluate_baseline --labels ..\data\label\20260715_shiyaoyuan_test.xlsx --run_inference 0
```

---

## 附录 F：数据目录规范（2026-08-20 实施）

> `data/` 按「实验隔离 + 数据类型五分法」组织；正式实验数据放 `data/<实验名>/`，测试数据放 `data/test/`，两者内部结构一致。此前散放的文件（`data/coco`、`data/test/*.mzML`、`data/test/testcase_data.xlsx` 等）已全部迁入规范位置；**标注文件统一存 `data/label/<trial>.xlsx`**（2026-08-20 起，不再随实验子目录）。

### F.1 目录结构

```
data/
├── coco/     # 对应实验构造好的 COCO 训练数据集（train/ + train_coco.json、val/ + val_coco.json；_xic/ 为构建时的 XIC 中间产物）
├── label/    # 各实验的人工标注数据（统一存 data/label/<trial>.xlsx；布局：化合物/通道/rt/peak_label/多峰起止/面积）
├── mzml/     # 原始数据转换得到的 mzML 文件
├── msdata/   # 原始 msdata 文件
└── wiff/     # 原始 wiff 文件
```

当前实际内容（20260715 试药园实验，位于 `data/test/` 下）：

| 目录 | 内容 |
|---|---|
| `data/test/coco/train|val/` | COCO 数据集（train=test_1 样品 61 图，val=test_2 样品 61 图） |
| `data/label/` | `20260715_shiyaoyuan_test.xlsx`（60 化合物 ×2 离子标注） |
| `data/test/mzml/` | `20260715_shiyaoyuan_test_1.mzML`、`_2.mzML` |
| `data/test/msdata/` | `20260715_shiyaoyuan_test.msdata` |
| `data/test/wiff/` | （空） |

`data/` 顶层同名五目录暂为空，新实验按同结构创建。

### F.2 数据流转关系

```
wiff/ + msdata/ ──(converters 格式转换)──> mzml/ ──(coco_annotation + label/)──> coco/
```

- `wiff/msdata` 是原始仪器数据（原料），经 `converters/` 转为 mzML；
- `mzml + label` 经 `preprocessing/coco_annotation.py` 构建为 COCO 训练数据集；
- 推理/评测读 `mzml`（输入）与 `label`（GT），**中间产物一律写 `../output/`，不写入 data/**。

### F.3 本次迁移的路径变更（代码与配置已同步）

| 旧路径 | 新路径 |
|---|---|
| `data/coco`（训练配置 `coco_path`） | `data/test/coco` |
| `data/test/20260715_shiyaoyuan_test/*.mzML` | `data/test/mzml/*.mzML` |
| `data/test/testcase_data.xlsx` | `data/label/20260715_shiyaoyuan_test.xlsx`（2026-08-20：标注统一存 `data/label/<trial>.xlsx`） |
| `data/coco/_xic`（评测 feature.csv 来源） | `data/test/coco/_xic` |
| `data/test/pred_v1_fixed` 等历史评测产物 | `../output/test/`（输出目录约定） |

同步更新的文件：`configs/quanformer_baseline.json`、`configs/quanformer_v2_finetune.json`、`configs/evaluation_baseline.json`、`preprocessing/coco_annotation.py`、`tools/evaluation/{evaluate_baseline,dump_queries,visualize_compare}.py`、`inference/cli.py`（docstring 示例）。

---

## 附录 G：标注数据质量 QC 设计（2026-08-20 定稿；P1 已实施）

> 背景：错误标注的传导链——污染训练 bbox（v2 类微调依赖 bbox=人工边界，实验日志 001 已证边界约定差异的直接后果）+ 误导评估 GT。新增「标注 RT 一致性」防线，并将各环节 QC 结果表统一到 `../output/QC/`。实施蓝图见 `docs/plan_qc.md`。

### G.0 P1 实施结果（2026-08-20）与真实数据发现 ⚠️

P1（训练/评估防线）已实施：`preprocessing/label_qc.py` 新模块 + `coco_annotation.py` 挂点（`--qc_label_rt_tol` 默认 1.0）。

**真实标注回归（data/label/20260715_shiyaoyuan_test.xlsx，120 行）首跑即发现 8 组双离子 RT 异常，16 行判「疑似实验有误」**：

| 化合物（两样品均异常） | 定量离子 RT | 定性离子 RT | 极差 (min) |
|---|---|---|---|
| 乙酰甲胺磷 | 28.478 / 28.488 | 2.705 / 2.634 | 25.77 / 25.85 |
| 灭螨醌 | 6.251 | 22.157 / 22.197 | 15.91 / 15.95 |
| 甲羧除草醚 | 10.458 / 10.488 | 23.239 | 12.78 / 12.75 |
| 羟基-灭螨醌 | 6.240 / 6.251 | 22.167 / 22.197 | 15.93 / 15.95 |

（跨样品检查 0 异常——同通道在各样品间 RT 稳定；异常全部集中在**样品内定量/定性离子不共流出**）

**影响判定**：这些通道的定量/定性离子 RT 差 12.8~25.9 min，物理上不可能共流出，极可能是定性离子通道标注到了干扰峰（或通道归属错误）。此前 v1/v2 评估的 GT 与 COCO 训练 bbox 均包含这 16 行可疑标注——**v2 微调与历次评估结果中涉及上述 4 类化合物的通道需在人工复核后重新审视**。

QC 表已可复现：`coco_annotation --qc_label_rt_tol 1.0` 运行后查看 `output/QC/coco_<实验名>_<时间戳>/qc_label_rt.csv`（excluded 行 + suggest_review 列）。

### G.1 判定规则

| 检查 | 分组键 | 统计量 | 阈值 | 含义 |
|---|---|---|---|---|
| A 跨样品 | `(compound, channel)` × 各 sample | `max(rt)−min(rt)` | >1.0 min | 某样品标注画错 / 仪器 RT 异常漂移 |
| B 双离子 | `(sample_id, compound)` × 各 channel | `max(rt)−min(rt)` | >1.0 min | 定量/定性通道张冠李戴 / 干扰峰误标 |

极差超阈值 → 判「疑似实验有误」：终端 WARN 警示人工复核 + 涉事行剔除（不生成 ROI、不进训练 bbox）+ 记入 QC 表。

物理依据：同色谱方法下同化合物 RT 稳定（连续进样漂移 <0.1 min）；定量/定性离子必然共流出。

### G.2 剔除粒度（无法仲裁时的宁缺毋滥策略）

- 跨样品、n≥3：仅剔偏离组中位数 >1 min 的样品行（多数派可信）；
- 跨样品、n=2：两行都剔（双方各偏 >0.5 min，无法判断谁错）；
- 双离子：两通道都剔（同理）。

### G.3 高效计算（两遍向量化 groupby，labels 单次解析三方复用）

```python
cross = df.groupby(["compound","channel"])["rt"].agg(rt_min="min", rt_max="max", rt_median="median")
cross["rt_range"] = cross["rt_max"] - cross["rt_min"]
inner = df.groupby(["sample_id","compound"])["rt"].agg(rt_min="min", rt_max="max")
inner["rt_range"] = inner["rt_max"] - inner["rt_min"]
```

### G.4 挂点与统一输出

| 挂点 | 行为 |
|---|---|
| `coco_annotation`（训练/评估数据构建） | 默认启用；命中行不参与 bbox 映射，ROI 降级负样本 |
| `inference.cli --mode pipeline` | 可选 `--labels` 启用（不传则跳过，保持推理无标注契约）；exclude 集 → `extract_xic_with_pyopenms(exclude_native_ids=...)`，reason 记入现有 `pipeline_qc_excluded.csv` 结构 |

统一 QC 输出 `../output/QC/<run_name>/`：`qc_label_rt.csv`、`qc_roi_channels.csv`、`qc_prediction_threshold.csv`、`qc_snr_boxes.csv`、`qc_post_refinement.csv`、`qc_summary.md`（五道防线全表，详见 README「质量控制（QC）」章节）。

### G.5 现有 QC 环节盘点（实施前的基线）

| 环节 | 位置 | 现有产物 |
|---|---|---|
| ROI 通道级（强度/点数） | `xic_extraction.py`（cli `_pipeline_qc_kwargs` 下发） | 各样品 `pipeline_qc_excluded.csv`（含 reason 列） |
| 预测框级（score 阈值） | `predictor.py` | 无记录表（仅丢弃） |
| (mz,q3) 去重 | `predictor.py` | 终端 INFO |
| SNR 框级 | `snr_filter.py` | `box_outside_snr_report.csv`（含 passed_* 列） |
| 精修框级（20+ 门控） | `peak_refinement.py` | 无记录表（不出现在 refined 输出即被剔） |
| 训练标注级（RT 越窗/窄框） | `coco_annotation.py` | 终端 WARN + 降级负样本 |

## 附录 H：B 范式（标注驱动 ROI）+ 训练侧 peak_label 正负样本（2026-08-20）

### H.1 背景与动机

原 ROI 生成是**通道驱动**：mzML 里每一条 chromatogram 生成一张 ROI，窗口中心取该通道平滑后**最高强度点**。这带来两个问题：

1. 推理与训练窗口中心不一致（训练用标注 RT 覆盖、推理用 apex），图像口径不统一；
2. 训练负样本来自「未标注通道」（TIC 等），与「训练识别峰、算峰面积」的目标脱节。

决策（与用户确认）：ROI 改为**标注驱动（B 范式）**，训练侧负样本改用标注文件的 `peak_label` 字段显式区分。

### H.2 ROI 生成范式变更（xic_extraction.py）

`extract_xic_with_pyopenms` 新增 `labels` 参数，提供时进入 label 驱动模式：

- 每行标注 `(compound, channel)` → `label_key()` → `native_id`「化合物名-1/-2」匹配 mzML 色谱；**未标注的通道不生成 ROI**
- **窗口中心 = 标注 `rt` 字段**（替代最高强度点）；`rt` 缺失/非法 → 剔除（reason=`label_rt_missing`）
- 标注了但 mzML 无对应通道 → 剔除并记录（reason=`label_no_channel`）
- 不传 `labels` 维持原 apex 行为（推理无需标注契约不变）

推理侧 `inference.cli --mode pipeline --labels <xlsx>` 触发；多样品标注按 `sample_id` 出现顺序对应 mzML（单样品直接全量）。

### H.3 训练侧正负样本方案（coco_annotation.py）

```
peak_label = 0  → 负样本：生成 ROI 图（窗口中心=rt）但无 bbox
peak_label = 1  → 正样本：遍历 peak_start1-3/peak_end1-3，每个有效区间一个 bbox（最多 3 个）
peak_label 缺失 → 按正样本（兼容无该列的文件）
其余值（如 2）  → 不入数据集
```

- RT 一致性 QC（附录 G 防线 1）仍在构建时启用：跨样品/双离子极差可疑行剔除、不入数据集
- bbox 由 `peak_start/peak_end`（分钟）经 ROI 窗口线性映射为像素；区间完全在窗口外或映射宽 <1px 的峰跳过
- 效果链：正样本=标注有峰区间 → bbox；负样本=标注无峰 → 无 bbox（模型学"图上无峰"）

### H.4 真实标注文件适配（阻断修复）

真实标注 `data/label/20260715_shiyaoyuan_test.xlsx`（120 行）为**多峰格式**：`peak_label`/`peak_count`/`peak_start1-3`/`peak_end1-3`/`area1-3` 等，**无单数 `peak_start/peak_end`**——原 `parse_labels_xlsx` 解析会直接报错（推理侧 `--labels` 同样受影响）。已扩展 `_LABEL_COLS` 并收紧必需列检查为 `compound/channel`；单数 `peak_start/peak_end` 保留兼容旧文件。

### H.5 验证与注意事项

- 真实文件解析：120 行通过；`peak_label` 分布 118 正 + 2 负；RT QC 240 项中 20 项需人工复核（标注质量待修）
- 训练侧重建数据集**必须 `--force`**（旧缓存为旧格式生成）
- ⚠️ 训练数据不再有「未标注通道负样本」——负样本仅来自 `peak_label=0`，当前只有 2 个，正负不平衡需关注（后续可补充负样本标注）
