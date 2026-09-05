# 项目 CSV / 表格 生成说明（blank_label.md）

> 本文件记录当前项目中**所有生成 CSV/表格**的用途与字段含义。
> **维护约定：以后每新增一个表格/CSV，必须在本文件中追加一节**，说明：
> 1. 生成脚本（文件路径 + 函数/行号）
> 2. 为什么生成（用途）
> 3. 每个字段（列）的含义

---

## 目录

1. [管线总览](#管线总览)
2. [预处理阶段（XIC 提取 / ROI 生成）](#1-预处理阶段xic-提取--roi-生成)
3. [标注 RT 一致性 QC](#2-标注-rt-一致性-qc)
4. [COCO 训练集构建 QC](#3-coco-训练集构建-qc)
5. [模型预测（阶段 2）](#4-模型预测阶段-2)
6. [SNR 框外噪声滤波（阶段 3）](#5-snr-框外噪声滤波阶段-3)
7. [峰精修（阶段 4）](#6-峰精修阶段-4)
8. [面积积分（阶段 5）](#7-面积积分阶段-5)
9. [管线汇总输出](#8-管线汇总输出)
10. [全谱扫描（massnova）](#9-全谱扫描massnova)
11. [诊断工具](#10-诊断工具)
12. [评估工具](#11-评估工具)
13. [实验脚本](#12-实验脚本)
14. [mzML 工具](#13-mzml-工具)
15. [常用字段字典](#14-常用字段字典)

---

## 管线总览

> 推理入口：`model/inference/cli.py`（`--mode`：`roi` / `roi2inference` / `pipeline` / `massnova`）。
> `pipeline` = 完整管线（XIC → 预测 → SNR → 精修）；`roi` = 仅 ROI 生成；`roi2inference` = 对已有 ROI 目录批量预测+积分；`massnova` = 整谱全峰识别。

```
mzML 数据
  │  (1) XIC 提取 / ROI 生成      → feature.csv / roi_windows.csv / pipeline_qc_excluded.csv / xic_matrix.npy
  │  (2) 模型预测                 → model_prediction_<样本>.csv / prediction.csv / qc3_threshold_<样本>.csv
  │  (3) SNR 框外噪声滤波         → prediction_snr.csv / qc4_snr_<样本>.csv
  │  (4) 峰精修                   → prediction_refined.csv / qc5_refined_<样本>.csv
  │  (5) 面积积分                 → prediction_refined_with_area.csv
  │  (汇总)                      → all.csv / predictions_model_all.csv / prediction_refined_all.csv
  │  (阶段报告)                  → predictions_model_report.md / prediction_refined_report.md / inference_report_<实验名>.md
  │  (QC 汇总, output/QC/<run>)  → qc1_label_rt.csv / qc2_roi.csv / qc3_threshold.csv / qc4_snr.csv / qc5_refined.csv
  │  (可选) 全谱扫描              → massnova_peaks.csv / scan_summary.csv / scan_qc_excluded.csv
  └─ (评估/实验工具)               → match_details.csv、area_pairs.csv 等
```

- 样本级明细表落在 `output/<run>/<阶段目录>/<样本>/`（pipeline 模式下阶段目录为 `xic_roi/`、`predictions_model/`、`prediction_refined/`），QC 汇总表落在 `output/QC/<run>/`。
- 所有 CSV 均使用 `encoding="utf-8-sig"`（Excel 打开中文不乱码）。
- 汇总表通常在第 0 列插入 `stem`（样本目录名）。

---

## 1. 预处理阶段（XIC 提取 / ROI 生成）

生成脚本：`model/preprocessing/xic_extraction.py`
用途：从 mzML 中按标注提取每条 MRM 通道的色谱（XIC），生成与图/矩阵一一对应的元数据表，供下游模型推理使用。

### 1.1 feature.csv

- 位置：`<output>/xic_roi/<样本>/feature.csv`（及 pipeline 各阶段目录）
- 用途：与 `xic_matrix.npy`、ROI 图一一对应的**通道/化合物元数据表**。ROI 文件名 `N_mz{mz}_q3{q3}.jpeg` 中的 `N` 即 `Compound Name`（1-based 行号）。

| 列名 | 含义 |
|---|---|
| `Compound Name` | 化合物序号（int，1-based，= feature 行号 = xic_matrix.npy 第 N+1 行） |
| `native_id` | mzML chromatogram 的 native id（如 `6-涕灭威-1`） |
| `mz` | 母离子 m/z（Q1），解析自 native_id 或 pyopenms precursor |
| `q3` | 子离子 m/z（Q3） |
| `RT` | ROI 窗口中心保留时间（分钟）：有标注=标注 rt，否则=平滑后强度最高点 apex RT |

### 1.2 roi_windows.csv

- 位置：`<output>/xic_roi/<样本>/roi_windows.csv`
- 用途：记录每张 ROI 图的**实际绘图窗口**（像素→RT 映射基准），保证下游 `box_to_rt_range` 与图完全一致。

| 列名 | 含义 |
|---|---|
| `image` | ROI 图像文件名（`N_mz{...}_q3{...}_<slug>.jpeg`） |
| `rt_lo` | 该 ROI 图 x 轴实际左边界（分钟） |
| `rt_hi` | 该 ROI 图 x 轴实际右边界（分钟） |

### 1.3 pipeline_qc_excluded.csv（样本级，中间台账）

- 位置：`<output>/xic_roi/<样本>/pipeline_qc_excluded.csv`
- 用途：记录**被 QC 剔除的通道明细**。消费方：`coco_annotation.py` / `inference/cli.py` 聚合为数据集级 `qc2_roi.csv` 后删除本文件。

| 列名 | 含义 |
|---|---|
| `chrom_index` | 被剔除通道在 mzML 中的色谱索引（label_no_channel 行为 -1） |
| `native_id` | 通道 native id |
| `q1` | 母离子 m/z |
| `q3` | 子离子 m/z |
| `reason` | 剔除原因：`tic_excluded`（TIC 类无 Q1/Q3）、`label_rt_missing`（标注 rt 缺失）、`label_rt` / `label_rt_cross_sample` / `label_rt_ion_pair`（标注 RT 一致性 QC）、`too_few_points`（点数不足）、`low_max_intensity`（强度过低）、`label_no_channel`（标注了但无对应通道） |
| `n_points` | 该色谱数据点数 |
| `max_intensity_smoothed` | 平滑后最大强度 |

### 1.4 xic_matrix.npy（非 CSV，配套矩阵）

- 用途：numpy 数组 `(N+1, S)`，第 0 行是公共 RT 轴（分钟），第 1..N 行是各通道对齐强度。

---

## 2. 标注 RT 一致性 QC

生成脚本：`model/preprocessing/label_qc.py`（调用方：训练侧 `coco_annotation.py`、推理侧 `inference/cli.py`）
用途：检查标注 RT 一致性——A 跨样品（同 compound+channel 跨样品 rt 极差>tol）、B 双离子（同 sample+compound 的定量/定性离子 rt 极差>tol），超阈值标记剔除。

### 2.1 qc1_label_rt.csv（防线 1）

- 位置：训练侧 `output/QC/coco_<数据集>/qc1_label_rt.csv`；推理侧 `output/QC/<run>/qc1_label_rt.csv`
- 用途：把标注 RT 一致性检查的**全量逐行结果**（含保留行与剔除行）落成台账，供人工复核；命中剔除的标注行最终不生成 ROI。

| 列名 | 含义 |
|---|---|
| `check_type` | 检查类型：`cross_sample`（A 跨样品极差）/ `ion_pair`（B 样品内双离子极差） |
| `sample_id` | 样品 ID |
| `compound` | 化合物名 |
| `channel` | 通道（定量离子 / 定性离子） |
| `rt` | 该行标注 RT（分钟） |
| `group_median` | 所在分组 RT 中位数（组内仅 1 行时为 None） |
| `rt_range` | 组内极差 max−min（分钟） |
| `n_group` | 组内行数 |
| `action` | `excluded`（剔除）/ `kept`（保留） |
| `suggest_review` | 是否建议人工复核（bool） |

---

## 3. COCO 训练集构建 QC

生成脚本：`model/preprocessing/coco_annotation.py`
用途：构建训练数据集时生成的 QC 阶段表。

### 3.1 qc2_roi.csv（防线 2）

- 位置：`<qc_root>/coco_<数据集名>/qc2_roi.csv`
- 用途：把各样品 `pipeline_qc_excluded.csv` 聚合为**数据集级 ROI 通道剔除汇总表**。

| 列名 | 含义 |
|---|---|
| `stem` | 来源 mzML 文件名（无扩展名）= 样品子目录名 |
| `chrom_index` | mzML 内色谱索引 |
| `native_id` | 被剔除通道的 native id |
| `q1` | 母离子 m/z |
| `q3` | 子离子 m/z |
| `reason` | 剔除原因（同 1.3 的 reason 枚举） |
| `n_points` | 该色谱数据点数 |
| `max_intensity_smoothed` | 平滑后最大强度 |

### 3.2 build_log.txt（COCO 构建日志）

生成脚本：`model/preprocessing/coco_annotation.py`
- 位置：`<output_dir>/build_log.txt`
- 用途：记录数据集构建过程日志（mzML 处理、标注匹配、QC 剔除计数等）。非 CSV，属配套日志。

---

## 4. 模型预测（阶段 2）

生成脚本：`model/inference/predictor.py`（批量模式由 `inference/cli.py --mode roi2inference` 透传调用，`roi2inference` 即"对已有 ROI 目录批量预测"）
用途：模型推理的**逐框积分明细**——每个预测框（每峰）一行。是整条 pipeline 的"第一份定量结果"，下游 SNR 滤波、精修都以它为输入。

### 4.1 model_prediction_<样本>.csv / prediction.csv

- 位置：批量模式（`roi2inference`/`pipeline`）`<batch_output>/<样本>/model_prediction_<样本>.csv`；单目录模式默认 `../output/inference/prediction.csv`
- 用途：逐框记录框坐标、置信度、峰 RT 区间、峰顶、面积与质量参数。
- 列（按落盘顺序）：

| 列名 | 含义 |
|---|---|
| `image` | ROI 图像文件名（如 `1_mz142.0000.jpeg`）；无检测时为占位名 |
| `compound_name` | 化合物标签（优先 feature 表 `native_id`） |
| `mz` | 母离子 m/z（Q1） |
| `old_rt` | feature 表 RT（**参考 RT**，非实测；实测峰顶在 `retention_time`） |
| `box_x1` / `box_y1` / `box_x2` / `box_y2` | 模型预测框像素坐标（无检测为 NaN） |
| `score` | 模型置信度 0~1（无检测行为 0.0） |
| `peak_start` / `peak_end` | 峰起点/终点 RT（分钟），由像素框经 ROI 窗口映射 |
| `retention_time` | 实测峰顶 RT（分钟）= 框内最高强度点 RT |
| `intensity_max` | 峰顶强度（框内最高强度） |
| `area` | 峰面积（按 `integration_method` 积分） |
| `point_counts` | 峰内连续高于基线的点数 |
| `snr` | 信噪比（compute_roi_quality_params） |
| `noise_std` | 噪声标准差 |
| `baseline_slope` | 基线斜率 |
| `peak_width_ratio` | 峰宽比（宽峰/双峰判定） |
| `dynamic_range` | 动态范围 |
| `integration_method_used` | 实际使用的积分方法名 |
| `peak_index` | 同图内按 RT 升序的峰序号 1..n |

### 4.2 qc3_threshold_<样本>.csv（防线 3）

- 位置：`<prediction_output 所在目录>/qc3_threshold_<样本>.csv`
- 用途：记录每张 ROI 图上**低于置信度阈值被丢弃的检测框数量**。

| 列名 | 含义 |
|---|---|
| `image` | ROI 图像文件名 |
| `n_queries` | 该图模型输出的检测框总数 |
| `n_kept` | 置信度 ≥ threshold 保留的框数 |
| `n_dropped` | 低于阈值丢弃的框数 |
| `max_confidence` | 该图最高置信度 |

---

## 5. SNR 框外噪声滤波（阶段 3）

生成脚本：`model/postprocessing/snr_filter.py`
用途：对每个预测框计算"框外噪声 SNR"，滤除低信噪比假阳性。

### 5.1 qc4_snr_<样本>.csv（防线 4，样本级）

- 位置：`<样本目录>/qc4_snr_<样本>.csv`（与 prediction 同目录）
- 用途：对输入 prediction.csv **每一行**逐行计算框外噪声 SNR（无论是否通过阈值都记录），供人工 QC 审计。
- 列：先复制输入 prediction 的全部原列，再追加：

| 追加列名 | 含义 |
|---|---|
| `snr_outside_box` | 框外噪声 SNR = 信号 / 框外残差 RMS（信号 = 框内最大强度 − 框外噪声均值） |
| `noise_rms_outside_box` | 框外噪声 RMS（σ） |
| `signal_outside_mean` | 框内峰高相对框外均值的超出量（信号） |
| `noise_mean_outside_box` | 框外噪声均值 |
| `n_noise_points` | 框外参与噪声估计的点数（过少则 SNR 为 NaN） |
| `rt_window_lo_used` / `rt_window_hi_used` | 实际使用的 ROI RT 窗口（分钟） |
| `passed_snr_threshold` | 是否 snr ≥ min_snr（bool） |
| `passed_min_chrom_points` | 是否满足最少色谱点数 |
| `passed_min_chrom_max_intensity` | 是否满足最小最大强度 |
| `pred_row_index` | 该行在输入 prediction.csv 中的行号 |

### 5.2 prediction_snr.csv

- 位置：`<样本目录>/prediction_snr.csv`
- 用途：**仅保留通过 SNR ≥ 阈值**的行，作为下游精修输入。同步重写 feature.csv / roi_windows.csv / xic_matrix.npy（仅保留行）与 snr_kept/、snr_dropped/ 标注图。
- 列：输入 prediction 全部列 + 追加 `snr_outside_box`。注意两处改写：
  - `image` 改为 `<原stem>_snr<SNR值>.jpeg`（对应 snr_kept/ 中的图名）；
  - `compound_name` 重写为 1-based 压缩行号，与 xic_matrix.npy 行号、feature.csv 的 `Compound Name` 严格对齐（否则下游映射会错位、图空白）。

---

## 6. 峰精修（阶段 4）

生成脚本：`model/postprocessing/peak_refinement.py`
用途：对预测框做**主峰区间修正**（interval correction + lookahead + 同伴框防撞）、**次峰识别**（模型小框 / ROI 次峰 / 谷分裂 / 主框双峰劈裂 / 左右重检），输出 main + small/small2/small3 宽表。

### 6.1 prediction_refined.csv（post_newtest 模式，默认）

- 位置：`<样本目录>/prediction_refined.csv`（`--output_name`）；`--keep_all_pred_boxes` 时另出 `prediction_refined_all.csv`（每行一个峰）
- 用途：精修后的主峰 + 次峰宽表，是面积积分、all.csv 的直接输入。

| 列名 | 含义 |
|---|---|
| `image` / `compound_name` / `mz` / `q3` | 来源标识 |
| `main_rt_min` / `main_rt_max` | 主峰精修后 RT 区间（分钟） |
| `main_rt_peak` | 主峰峰顶 RT（区间内强度最大点） |
| `main_height` | 主峰峰高（区间内最大强度） |
| `main_score_ai` | 主峰的模型 AI 置信度（输入行 `score`） |
| `main_snr` | 主峰 SNR（对精修后区间按框外噪声口径计算） |
| `main_skew` / `main_skew_local` | 主峰区间加权偏度 / 峰顶 ±std_skew_window 内局部偏度 |
| `small_rt_min/max/peak`、`small_height`、`small_score_ai_raw`、`small_score_ai_discounted`、`small_height_ratio`、`small_source` | 第一个次峰：区间、峰高、原始/折减 AI 分、高比（小峰高/主峰高）、来源（`model`/`roi_secondary`/`main_double_split`/`valley_split`/`lr_repredict`/`none`/`sample_recover_roi`/`sample_recover_valley`） |
| `small2_*`、`small3_*` | 第二、第三个候选次峰（同 small 字段） |
| `has_secondary_gate` | ROI 是否满足 has_secondary_peak_in_roi 判据（0/1） |
| `gate_ok_for_adjustment` | 主峰修正门控（score≥min_confidence 且 SNR≥min_snr 且次峰判据通过）（0/1） |
| `lr_repredict_applied` | 是否发生左右重检替换主框（0/1） |
| `noise_avg_last25pct` | ROI 后 25% 平均噪声 |
| `small_min_height_required` | 次峰最低高度门槛 |

### 6.2 qc5_refined_<样本>.csv（防线 5，精修门控列子集）

- 位置：`<样本目录>/qc5_refined_<样本>.csv`（`_emit_qc5_sample` 从 prediction_refined.csv 抽取）
- 用途：把精修门控关键列单独成表，供防线 5 QC 汇总审计。
- 列（`_QC_POST_GATE_COLS`，仅保留实际存在的列）：

| 列名 | 含义 |
|---|---|
| `image` / `compound_name` / `mz` / `q3` | 来源标识 |
| `rt_min` / `rt_max` / `rt_peak` | 精修区间与峰顶 RT（兼容列） |
| `score_ai` | AI 置信度（keep_all 模式兼容列） |
| `main_score_ai` | 主峰 AI 置信度 |
| `main_snr` | 主峰 SNR |
| `main_interval_width` | 主峰区间宽度 |
| `main_conf_composite` | 主峰复合置信度（0.45·ai + 0.25·snr_norm + 0.20·rt_term + 0.10·skew_term） |
| `main_conf_final` / `small_conf_final` | 主峰/小峰最终置信度（ai × RT 非线性衰减 × ... × 偏度加权） |
| `final_conf_best` | 最终置信度（保留小峰时取 main/small 较大者） |
| `need_manual_review` | 最终置信度 < 阈值（默认 0.90）→ 1，需人工复核 |
| `keep_small_by_standard` | 小峰是否按标品规则保留（RT≤容差 且 (偏度容差 或 AI 分达标)） |
| `small_rt_gate_pass` / `small_skew_gate_pass` / `small_ai_gate_pass` | 小峰 RT/偏度/AI 门控是否通过 |
| `gate_ok_for_adjustment` / `has_secondary_gate` | 同 6.1 |
| `main_refine_applied` | 主峰是否应用精修 |
| `small_recover_trigger` | 小峰恢复触发：`none`/`missing`/`weak_retry`/`weak_retry_failed` |

### 6.3 standard_mode 标品校准输出（标品模式）

生成脚本：`model/postprocessing/peak_refinement.py`（`run_standard_mode`），读取所有标品 prediction_refined.csv，按 compound_key（mz+q3）分组选主峰、低 R² 修复、mz 级预期 RT。位置：`--output_dir`。

| 文件 | 用途 | 关键列 |
|---|---|---|
| `standard_selected_peaks.csv` | 每个 (compound_key, 浓度) 选定主峰的行 | `compound_key`、`mz`、`q3`、`concentration_ppb`、`selected_peak_tag`、`selected_rt_peak`、`selected_height`、`selected_score_ai`、`selected_skew`、`area_proxy`、`source_file`、`source_image`、`small_rt_peak_raw`、`small_height_raw`、`small_score_ai_raw`、`small_source_raw`、`r2_compound`、`r2_before_compound`、`r2_after_compound`、`slope`、`intercept`、`small_keep_after_main`、`small_rt_peak_selected`、`small_height_selected`、`small_score_selected` |
| `standard_refs.csv` | 每通道标品参考 | `compound_key`、`ref_rt_peak`（选定主峰 RT 中位数）、`ref_skew`、`r2`、`mz_expected_rt` |
| `standard_mz_expected_rt.csv` | mz 级预期 RT | `mz`、`mz_expected_rt`、`n_channels_total`、`n_channels_used`、`used_compound_keys` |
| `r2_compare_compound.csv` | 化合物级 R² 对比 | `compound_key`、`r2_before`、`r2_after`、`improved` |
| `standard_mz_quant_max_area.csv` | mz 级定量表（按 mz×浓度取 area_proxy 最大行） | = standard_selected_peaks 全列 |
| `r2_compare_mz_max_area.csv` | mz 级 R² 前后对比 | `mz`、`n_points`、`r2_before`、`r2_after`、`slope_before`、`intercept_before`、`slope_after`、`intercept_after`、`removed_point_index_for_after` |

### 6.4 sample_mode 样品模式输出

生成脚本：`model/postprocessing/peak_refinement.py`（`run_sample_mode`），用 standard_refs 的参考 RT/偏度过滤小峰、计算复合/最终置信度。在 prediction_refined.csv 全列基础上新增（`--output_csv`）：

| 新增列名 | 含义 |
|---|---|
| `compound_key` | 化合物键（mz+q3） |
| `ref_rt_peak` / `ref_skew` | 标品参考 RT / 偏度 |
| `target_rt_used` / `target_rt_source` | 生效目标 RT / 来源（`standard_ref` / `main_rt_fallback`） |
| `main_refine_applied` / `main_interval_width` / `small_interval_width` | 精修标记与区间宽度 |
| `keep_small_by_standard` | 小峰是否按标品规则保留 |
| `main_conf_composite` / `small_conf_composite` | 主峰/小峰复合置信度 |
| `main_conf_final` / `small_conf_final` | 主峰/小峰最终置信度 |
| `noise_mean_outside_all_boxes` | 所有框（主+小）外噪声均值 |
| `snr_main_over_noise_mean` / `snr_small_over_noise_mean` | 主/小峰相对噪声均值的 SNR |
| `final_conf_best` | 最终置信度 |
| `need_manual_review` | 是否需人工复核（0/1） |
| `small_recover_trigger` | 小峰恢复触发 |
| `small_rt_gate_pass` / `small_skew_gate_pass` / `small_ai_gate_pass` | 小峰门控 |

---

## 7. 面积积分（阶段 5）

生成脚本：`model/postprocessing/area_integration.py`
用途：对 prediction_refined.csv 的 main/small/small2/small3 四个 RT 区间，在 xic_matrix.npy 上按统一积分算法重算面积，供下游 R²/定量使用。

### 7.1 prediction_refined_with_area.csv

- 位置：单样本 `<images_path>/prediction_refined_with_area.csv`；批量 `batch_output/<样本>/` 下
- 用途：精修结果 + 补算面积，是 `all.csv`、标品定量、面积对比实验的数据源。
- 列：prediction_refined.csv 全列 + 新增：

| 追加列名 | 含义 |
|---|---|
| `main_area` / `small_area` / `small2_area` / `small3_area` | 各区间梯形积分 ×60（基线校正 + 噪声门限削噪） |
| `main_retention_time` / `small_retention_time` / `small2_retention_time` / `small3_retention_time` | 各区间峰顶 RT |
| `main_intensity_max` / `small_intensity_max` / `small2_intensity_max` / `small3_intensity_max` | 各区间最大强度 |
| `main_point_counts` / `small_point_counts` / `small2_point_counts` / `small3_point_counts` | 各区间内强度>0 的最长连续点数 |
| 兼容列 `area`、`retention_time`、`intensity_max`、`point_counts`、`rt_min`、`rt_max` | = main 对应列（兼容下游） |
| `integration_method_used` | 积分方法（如 `snr`） |

### 7.2 prediction.csv（非 --from_refined 模式，旧版 newtest 流程）

- 用途：模型预测 + SNR 积分一步完成。列：`image / image_path / compound_name / mz / q3 / native_id / old_rt / box_x1 / box_y1 / box_x2 / box_y2 / score / rt_min / rt_max / retention_time / intensity_max / area / point_counts`。写盘前按 `(mz, q3)` 去重保留面积最大行。

---

## 8. 管线汇总输出

生成脚本：`model/inference/cli.py`（跨样本汇总）+ `model/tools/evaluation/inference_report.py`（all.csv）

### 8.1 all.csv（最终交付明细表）

- 位置：`<base_out>/all.csv`
- 用途：**每样品 × 每 ROI 一行**的合并明细，正式呈现 `sample/label/compound_ion/score/rt/area` 语义。数据源为各样本精修结果（补算面积后写回 prediction_refined_with_area.csv）。

| 列名 | 含义 |
|---|---|
| `sample` | 样本名（= 子目录名） |
| `image` | ROI 图像文件名 |
| `native_id` | 由 feature 表行号反查的 native_id（如 `6-涕灭威-1`） |
| `compound` | 化合物名（从 native_id 或 image 解析） |
| `ion` | 离子通道号 1/2（定量/定性离子标识） |
| `mz` | 母离子 m/z（Q1） |
| `q3` | 子离子 m/z |
| `main_rt_min` / `main_rt_max` | 主峰起始/结束 RT（分钟） |
| `main_rt_peak` | 主峰峰顶 RT（分钟）——"rt" 的正式列 |
| `main_height` | 主峰峰高 |
| `main_score_ai` | 主峰 AI 置信度——"score" 的正式列 |
| `main_snr` | 主峰 SNR |
| `main_area` | 主峰面积——"area" 的正式列（`detected` 判定依据） |
| `main_intensity_max` | 主峰最大强度 |
| `main_point_counts` | 主峰连续点数 |
| `small_area` / `small2_area` / `small3_area` | 次峰 1/2/3 面积 |
| `small_count` | 实际存在 RT 区间的次峰个数（0-3） |
| `has_secondary_gate` | 次峰门控是否启用（0/1） |
| `gate_ok_for_adjustment` | 门控是否允许边界调整（0/1） |
| `lr_repredict_applied` | 是否应用 LR 重预测（0/1） |
| `detected` | 检出标记：main_area 有限且 > 0 |

### 8.2 predictions_model_all.csv（模型输出根级汇总）

- 位置：`<base_out>/predictions_model/predictions_model_all.csv`
- 用途：把各样本 `model_prediction_*.csv` 原样合并，首列插入 `stem`。列 = `stem` + 4.1 全部列。

### 8.3 prediction_refined_all.csv（精修输出根级汇总）

- 位置：`<base_out>/prediction_refined/prediction_refined_all.csv`
- 用途：优先合并各样本 `prediction_refined_with_area.csv`（缺失时回退 `prediction_refined.csv`），首列插入 `stem`。列 = `stem` + 精修/面积列。

### 8.4 QC 汇总表（output/QC/<run>/）

| 文件 | 来源（样本级） | 说明 |
|---|---|---|
| `qc1_label_rt.csv` | — | 标注 RT 一致性台账（见 2.1） |
| `qc2_roi.csv` | 各样本 `pipeline_qc_excluded.csv` | 首列插 `stem` 后 concat（见 3.1） |
| `qc3_threshold.csv` | 各样本 `qc3_threshold_*.csv` | 首列插 `stem`（见 4.2） |
| `qc4_snr.csv` | 各样本 `qc4_snr_*.csv` | 首列插 `stem`（见 5.1） |
| `qc5_refined.csv` | 各样本 `qc5_refined_*.csv` | 首列插 `stem`（见 6.2） |
| `qc_summary.md` / `qc_alert.md` | 各防线统计 | QC 汇总 / 全防线人工预警报告（需人工复核清单） |

> 注：`output/QC/full_pipeline/` 下存在旧版命名 `qc_label_rt.csv`、`qc_roi_channels.csv`、`qc_prediction_threshold.csv`、`qc_snr_boxes.csv`、`qc_post_refinement.csv`，字段与上述 qc1~qc5 完全一致，仅文件名不同（早期 run 命名）。

### 8.5 阶段级 Markdown 报告（含表格）

生成脚本：`model/inference/cli.py`（pipeline 收尾时 `_write_predictions_model_summary` / `_write_prediction_refined_summary`）

| 文件 | 位置 | 用途 |
|---|---|---|
| `predictions_model_report.md` | `<base_out>/predictions_model/` | 模型输出阶段汇总：每样本图数/峰数/最高置信度/平均 SNR/平均面积 |
| `prediction_refined_report.md` | `<base_out>/prediction_refined/` | 精修输出阶段汇总（精修后各统计量） |

### 8.6 推理报告 inference_report_<实验名>.md

生成脚本：`model/tools/evaluation/inference_report.py`（`generate_for_pipeline`，cli.py pipeline 收尾调用；`--no_report` 关闭）
- 位置：`<base_out>/inference_report_<实验名>.md`（实验名由 `--exp_name` 指定，缺省回退：单 mzML→文件名、目录→目录名）
- 用途：可读推理报告——样本摘要 / 化合物×样品面积矩阵 / QC 汇总 / 管线与列说明。同名配套机器可读明细为 `all.csv`（见 8.1）。
- 同脚本还为每个样品补算并写回 `prediction_refined/<样品>/prediction_refined_with_area.csv`（见 7.1）。

### 8.7 计时日志 pipeline_timing.log / pipeline_timing_runs.jsonl

生成脚本：`model/inference/cli.py`（`--no_timing` 关闭；终端仍打印计时汇总）
- 位置：`<base_out>/pipeline_timing.log`（文本）、`<base_out>/pipeline_timing_runs.jsonl`（结构化 JSONL）
- 用途：记录 pipeline 各阶段（XIC 提取/预测/SNR/精修/报告）耗时，供性能分析。非 CSV，属配套日志。

---

## 9. 全谱扫描（massnova）

生成脚本：`model/inference/massnova.py`
用途：不依赖标注，对 mzML 全部 transition 做整谱峰识别 + 模型验证。位置：`<out_root>/<样本>/`。

### 9.1 massnova_peaks.csv（主结果表）

- 用途：**每个峰一行**的明细。

| 列名 | 含义 |
|---|---|
| `mzml_stem` | 输入 mzML 文件名 stem |
| `chrom_index` | mzML 中色谱（通道）索引 |
| `uid` | 通道 native_id |
| `compound_name` | 化合物名（此处取 uid 文本） |
| `q1` | 母离子 m/z |
| `peak_no` | 通道内峰序号（RT 升序 1..n） |
| `rt_min` / `rt_peak` / `rt_max` | 峰起始 / 峰顶 / 结束 RT（分钟） |
| `apex_intensity` | 峰顶强度 |
| `area` | 峰面积（trapz 积分） |
| `snr` | 峰级本地 SNR（门控用） |
| `n_points` | 峰内高于基线的连续点数 |
| `validated` | 是否通过模型验证（提供 --model 时；True=模型框 RT 优先） |
| `model_score` | 模型置信度（未验证为 NaN） |
| `boundary_source` | 边界来源：`signal`（信号外推）/ `model`（模型框 RT） |

### 9.2 scan_summary.csv（通道级汇总）

| 列名 | 含义 |
|---|---|
| `chrom_index` | 色谱索引 |
| `uid` | 通道 native_id |
| `compound_name` | 化合物名 |
| `q1` | 母离子 m/z |
| `n_peaks` | 该通道检出的峰数 |
| `rt_range_min` / `rt_range_max` | 该通道 RT 轴最小/最大值（分钟） |
| `max_intensity` | 该通道最大强度 |

### 9.3 scan_qc_excluded.csv（通道 QC 剔除记录，仅存在剔除时写盘）

| 列名 | 含义 |
|---|---|
| `chrom_index` | 色谱索引 |
| `uid` | 通道 native_id |
| `q1` | 母离子 m/z |
| `reason` | 剔除原因：`empty`（无数据点）、`tic_excluded`（无 Q1，多为 TIC）、`duplicate`（与已处理通道完全重复）、`too_few_points`（点数不足）、`low_max_intensity`（强度过低） |
| `n_points` | 色谱点数 |
| `max_intensity` | 最大强度（平滑前 raw） |

---

## 10. 评估工具

### 10.1 evaluate_baseline.py（`model/tools/evaluation/evaluate_baseline.py`）

**match_details.csv** —— 每个预测框/标注峰与人工标注的匹配判定明细（TP/FP/FN），用于 P/R/F1 溯源。

| 列名 | 含义 |
|---|---|
| `stem` | mzML 文件名（去扩展名） |
| `native_id` | ROI 的 native_id |
| `result` | `TP` / `FP` / `FN` |
| `gt_start` / `gt_end` | 人工标注峰起始/结束 RT（分钟；FP 无命中时取该图第一个标注峰） |
| `pred_start` / `pred_end` | 预测框起始/结束 RT（FN 时为 None） |
| `score` | 预测置信度（FN 时为 None） |
| `quant` | 是否进入定量配对（1/0） |

**area_pairs.csv** —— TP 及宽松配对中"预测面积 vs 人工面积"数据对，供面积 R² 与跨进样 RSD 计算。

| 列名 | 含义 |
|---|---|
| `stem` | mzML 文件名 |
| `native_id` | ROI 的 native_id |
| `pred_area` | 预测框面积 |
| `manual_area` | 人工标注面积 |
| `match` | 配对类型：`strict` / `loose` |

**evaluation_report.json** —— 一键评测汇总（非 CSV）：`{model, mzmls, labels, tolerance, quant_tolerance, metrics}`，metrics 含 P/R/F1、面积 R²、RT 偏差、RSD 等。

**qc_alert.md** —— 评测侧人工预警报告（与管线 QC 的 qc_alert.md 同源格式）。

### 10.2 refine_ablation.py（`model/tools/evaluation/refine_ablation.py`）

**detail.csv** —— "无精修（A）/有精修（B）"两种方法下每个检测框相对人工标注的命中明细。

| 列名 | 含义 |
|---|---|
| `sample` | 样品名 |
| `native_id` | ROI 的 native_id |
| `method` | `A_无精修` / `B_精修` |
| `tag` | 峰标签：`main` / `small` / `small2` / `small3` |
| `gt_start` / `gt_end` / `gt_area` | 命中的人工峰起止 RT 与面积（未命中为 NaN） |
| `pred_start` / `pred_end` / `pred_score` / `pred_area` | 预测框起止 RT、置信度、面积 |
| `hit` | 是否命中人工峰（1/0，起止偏差 ≤ ±0.1 min） |

**boundary_pairs.csv** —— 同一人工峰同时被 A 和 B main 命中时，起止边界与面积的成对明细（核心指标）。

| 列名 | 含义 |
|---|---|
| `sample` / `native_id` | 样品与通道 |
| `gt_start` / `gt_end` / `gt_area` | 人工峰起止 RT 与面积 |
| `A_start` / `A_end` / `A_area` | A 方法（无精修）框起止 RT 与面积 |
| `B_start` / `B_end` / `B_area` | B 方法（精修）main 峰起止 RT 与面积 |

### 10.3 visualize_compare.py（`model/tools/evaluation/visualize_compare.py`）

**compare_summary.csv** —— 双模型（v1/v2）预测与人工标注的逐通道复核汇总（`{name1}`/`{name2}` 由 --name1/--name2 决定）。

| 列名 | 含义 |
|---|---|
| `sample` | 样品目录名 |
| `native_id` | ROI 的 native_id |
| `gt_start` / `gt_end` / `gt_area` | 人工标注起止 RT 与面积（无标注为 None） |
| `{name1}_start` / `{name1}_end` / `{name1}_score` / `{name1}_verdict` | 模型1 最高分框起止 RT、置信度、判定（`TP`/`FN(漏检)`/`FP(偏Δs/Δe)`/`FP(区间无效)`/`无标注`） |
| `{name2}_start` / `{name2}_end` / `{name2}_score` / `{name2}_verdict` | 同模型2 |

### 10.4 inference_report.py（`model/tools/evaluation/inference_report.py`）

- 输出 `<output_dir>/all.csv`（跨样品合并明细，同 8.1）与各样品 `prediction_refined_with_area.csv`（同 7.1）。

---

## 11. 实验脚本

### 11.1 oulu/area_compare.py → `oulu_area_compare_details.csv`

对比欧陆标准品面积 vs AI 主峰面积（main_area）相对误差。

| 列名 | 含义 |
|---|---|
| `sample` | 样品号（1-6） |
| `pipeline_name` | pipeline 样品目录名 |
| `compound` | 化合物名（标准品 Component Name） |
| `std_area` | 标准品面积 |
| `ai_area` | AI 主峰面积（main_area） |
| `rel_error` | 相对误差 \|ai−std\|/std |

### 11.2 oulu/full_report.py → `oulu_full_comparison.csv`

欧陆标准 vs AI 主峰面积（`prediction_refined.csv` 的 `main_area`）全量对比。

| 列名 | 含义 |
|---|---|
| `sample` / `pipeline_name` / `compound` / `std_area` / `ai_main_area` | 同 12.1 |
| `rel_error` | 相对误差 |
| `error_band` | 误差分档：`(0%,1%]` / `(1%,2%]` / `(2%,5%]` / `(5%,10%]` / `(10%,50%]` / `(50%,100%]` / `>100%` |

### 11.3 jiangnan/area_compare.py → `area_comparison_details.csv`

对比江南大学农残仪器 CSV 与人工加标 OS.txt 面积。

| 列名 | 含义 |
|---|---|
| `compound` | 化合物名（标准化后小写） |
| `csv_area` | CSV（仪器）面积 |
| `os_area` | OS（人工加标）面积 |
| `rel_error` | 相对误差（一侧缺失为 None） |
| `pair` | 对比对标识（`<csv文件>_vs_<os_sample>`） |

### 11.4 jiangnan/area_ratio.py → `area_ratio_details.csv`

两浓度对（10/20ppb 或 20/50ppb）下 CSV 面积比 vs OS 面积比一致性。

| 列名 | 含义 |
|---|---|
| `compound` | 化合物名（标准化后） |
| `csv_low` / `csv_high` | 低/高浓度 CSV 面积 |
| `os_low` / `os_high` | 低/高浓度 OS 面积 |
| `R_csv` | csv_low / csv_high |
| `R_os` | os_low / os_high |
| `metric_diff` | \|R_csv − R_os\|（核心指标） |

### 11.5 jiangnan/compound_presence.py → `compound_presence_all.csv` / `compound_only_summary.csv` / `only_*.csv`

对比 CSV（仪器）与 OS（人工加标）两侧化合物集合差异。

| 文件 | 列名 | 含义 |
|---|---|---|
| `compound_presence_all.csv` | `compound` | 化合物名 |
| | `in_csv_low` / `in_csv_high` / `in_os_low` / `in_os_high` | 是否出现在低/高浓度 CSV / OS（布尔） |
| | `only_type` | only 分类：`none` / `only_csv_low` / `only_csv_high` / `only_os_low` / `only_os_high` / `only_csv_both` / `only_os_both` / `present_multiple` |
| `compound_only_summary.csv` | `only_type` / `count` | 各 only 类型计数 |
| `only_*.csv` | `compound` | 该类型独占的化合物名 |

---

## 12. mzML 工具

### 12.1 mzml/inspect.py

把 mzML 可读内容导出为 CSV（`<stem>_inspect/` 目录）。

| 文件 | 列名 | 含义 |
|---|---|---|
| `<stem>_chrom_summary.csv` | `chrom_index` | 色谱序号（0 起） |
| | `native_id` | 色谱 native id |
| | `q1` / `q3` | 前体/产物离子 m/z |
| | `n_points` | 数据点数 |
| | `rt_min_min` / `rt_max_min` | RT 最小/最大值（分钟，自动换算） |
| | `intensity_max` / `intensity_mean` | 最大/平均强度 |
| `<stem>_spectrum_summary.csv` | `spec_index` / `ms_level` / `n_peaks` / `rt_sec` | 谱图序号 / 质谱级数 / 峰数 / 保留时间（秒） |
| `<stem>_points/<safe_name>.csv` | `rt_sec` / `intensity` | 单色谱数据点（RT 秒 / 强度） |

### 12.2 mzml/chromatogram.py（export 子命令）

输出 `<stem>_<safe_name>_points.csv`：`rt_sec`（保留时间，秒）、`intensity`（强度）。供外部工具绘图或核对。

---

## 13. 常用字段字典

| 字段 | 含义 | 出现位置 |
|---|---|---|
| `stem` | 样本/源文件目录名（去扩展名） | 所有汇总表首列 |
| `sample` | 样本名 | all.csv、verify_snr_dual_matrix.csv、评估工具 |
| `image` | ROI 图像文件名 | 几乎所有明细表 |
| `native_id` | mzML 色谱通道标识（如 `6-涕灭威-1`） | feature.csv、all.csv、QC 表 |
| `compound` / `compound_name` | 化合物名 | all.csv / 明细表 |
| `ion` | 离子通道号 1/2 | all.csv |
| `compound_ion` | "化合物-离子"串 | verify_snr_dual_matrix.csv |
| `mz` / `q1` | 母离子 m/z（Q1） | 明细表 |
| `q3` | 子离子 m/z | 明细表 |
| `RT` / `old_rt` / `ref_rt_peak` / `target_rt_used` | 参考/预期保留时间（分钟） | feature.csv、精修表、标品表 |
| `rt_min` / `rt_max` / `rt_peak` / `peak_start` / `peak_end` | 峰起/止/顶 RT（分钟） | 明细表 |
| `main_rt_min` / `main_rt_max` / `main_rt_peak` | 主峰起/止/顶 RT | 精修表、all.csv |
| `height` / `intensity_max` / `apex_intensity` / `main_height` | 峰高（最大强度） | 明细表 |
| `score` / `main_score_ai` / `model_score` | 模型 AI 置信度（0~1） | 明细表 |
| `snr` / `main_snr` / `snr_outside_box` | 信噪比 | 明细表、QC 表 |
| `area` / `main_area` / `small*_area` | 峰面积 | 明细表、all.csv |
| `detected` / `is_detected` / `validated` | 检出标记（bool） | all.csv、massnova_peaks.csv |
| `point_counts` / `n_points` / `main_point_counts` | 峰内连续点数 | 明细表 |
| `integration_method_used` | 积分方法名 | 明细表 |

---

## 附：新增 CSV 时如何更新本文档

新增任何 CSV/表格（无论哪个脚本），请在本文件对应章节（或新增章节）追加：

```
### <文件名>

- 生成脚本：<脚本路径>（函数/行号）
- 位置：<默认输出路径>
- 用途：<为什么生成，被谁消费>
- 列：

| 列名 | 含义 |
|---|---|
| ... | ... |
```

同时建议在 [常用字段字典](#14-常用字段字典) 中补充新字段。
