# 实验报告（Experiment Report）

> 记录模型实验的假设、证据、结论与影响。每篇实验独立成节，按时间倒序排列。
> 数据与命令均可复现，结论均给出证据出处。

---

## 实验日志 001：v1「先成功后失败」之谜 —— 推理取类 bug 与 shadow query 假象

- **日期**：2026-08-17
- **涉及模型**：quanformer.pth（v1，原基线权重，未改动）、quanformerv2.pth（v2，data/coco 微调）
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
4. **v2 不受影响的原理**：微调数据（data/coco）的 bbox 直接映射自人工 `peak_start/peak_end`，微调把「峰 query」的框校准到了人工约定上（起点偏差均值 −0.017 min、tIoU 中位 0.72）——自信与正确终于统一。

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
D:\Anaconda3\envs\gamstekpeaking\python.exe -m tools.evaluation.dump_queries --model checkpoint/quanformer.pth --xic_root ../data/coco/_xic --labels ../data/test/testcase_data.xlsx --limit 8

# v2 对照（预期：峰 query 又自信又贴 GT）
D:\Anaconda3\envs\gamstekpeaking\python.exe -m tools.evaluation.dump_queries --model checkpoint/quanformerv2.pth --xic_root ../data/coco/_xic --labels ../data/test/testcase_data.xlsx --limit 8
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

**① `roi`**：`extract_xic_with_pyopenms` 逐 mzML 读色谱 → `<out>/<key>/`：
- 每通道 ROI jpeg（命名 `N_mz{母离子}_q3{子离子}.jpeg`）
- `feature.csv`、`roi_windows.csv`、`xic_matrix.npy`
- 被 QC 剔除的通道记录 `pipeline_qc_excluded.csv`
- `--model + --plot`：`generate_prediction_plots` 预测框标注图

**② `batch_dir`**：复用 `predictor.main()` 批量模式，逐子目录：
- `<out>/<子目录>/prediction.csv`（积分方式非 linear 时为 `prediction_{method}.csv`）
- `<out>/<子目录>/predicted_plots/`（`--plot`）

**③ `pipeline`**（`base_out = --output_dir`，默认 `results/full_pipeline`；单文件与批量统一走批量路径，产物布局一致）：

| 阶段 | 输出目录 | 产物 |
|---|---|---|
| ① ROI 生成 | `base_out/xic-roi-batch/<key>/` | ROI jpeg、`feature.csv`、`roi_windows.csv`、`xic_matrix.npy` |
| ② 模型预测 | `base_out/batch_predictions/<key>/` | `prediction.csv`（或 `prediction_{method}.csv`）、`predicted_plots/`（`--plot`） |
| ③ SNR 筛选 | `base_out/snr_filtered/<key>/SNR_box_<thr>/` | `prediction.csv`（保留行）、`feature.csv`、`roi_windows.csv`、`xic_matrix.npy`、`box_outside_snr_report.csv`；`--save_snr_jpeg` 时 `筛选保留/`、`筛选剔除/` 红框标注图 |
| ④ 框修正 | 同上 `SNR_box_<thr>/` | `prediction_refined.csv`（`--post_output_name` 可改）、`refined_plots/`（`--plot`） |
| 计时 | `base_out/` | `pipeline_timing.log`、`pipeline_timing_runs.jsonl`（阶段耗时 + 资源统计） |

### 输出差异说明

- **prediction.csv 列**：image, image_path, compound_name, mz, q3, old_rt, box_x1/y1/x2/y2, score, rt_min, rt_max, retention_time, intensity_max, area, point_counts, snr, noise_std, baseline_slope, peak_width_ratio, dynamic_range, integration_method_used（每 (mz,q3) 只保留面积最大行）。
- **SNR 阶段产物**：`prediction.csv` 为通过 SNR 门槛的检测框（供 post 读取），`box_outside_snr_report.csv` 为逐框 SNR 明细；两者均含 `image_path` 指向 ROI 图。
- **精修 `prediction_refined.csv`**：为主峰+次峰识别/谷值拆分后的最终峰列表，含框修正后 RT 边界与置信度。
- **积分方法**：cli `--integration_method`（linear/raw/external_baseline）非 linear 时预测文件名为 `prediction_{method}.csv`，不覆盖默认；predictor 底层另支持 peak_adaptive/adaptive/minval_noise_right 等仅供内部调用。
- **QC 门槛**（pipeline 模式）：`--pipeline_min_max_intensity`（默认 1000）、`--pipeline_min_chrom_points`（默认 10），未达标的通道不生成 ROI、不参与预测。
