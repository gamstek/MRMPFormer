# 推理产物重构方案（目录改名 / 表格列调整 / 可视化 / QC 命名 / 冗余分析）

> 日期：2026-08-27
> 范围：`model/inference/`（cli.py、predictor.py）、`model/postprocessing/`（snr_filter.py、peak_refinement.py）、`model/utils/predict_utils.py`、`model/preprocessing/`（xic_extraction.py、masked_roi_generator.py）、`model/tools/`（reprocess.py、artifacts.py 及各诊断/可视化/评估工具的路径同步）、README / User_Tutorials / CLAUDE.md 文档同步。
> 本文档共四部分：**§1 目标目录树与需求映射**（12 条需求 → 实施项总表，第 11/12 条为追加的 `xic_roi` 改名与 QC 五防线编号统一）；**§2 分步执行计划**（Step 1–9，含每步的改动点、实现要点、验证命令）；**§3 冗余分析**（对应需求第 8、10 条，**只分析、不直接实施**）；**§4 风险/兼容、验证清单与决策记录**。
> QC 命名约定（用户已确认，2026-08-27 定稿）：QC 表编号统一**与 README"五道防线"一致**——`qc1_label_rt`（标注 RT 一致性）、`qc2_roi`（ROI 通道级）、`qc3_threshold`（预测框级）、`qc4_snr`（SNR 框级）、`qc5_refined`（精修框级）；**带 `<样本名>` 者为样本内 QC 明细，不带者为整轮推理的全样本汇总**。README 防线表中"标注 RT 一致性（新）"的"（新）"去除（见 Step 4 / Step 8）。

---

## §1 目标目录树与需求映射

### 1.1 重构前后目录树对照

**现状**（`cli.py:1013-1023` 定义的 pipeline 布局）：

```
<output_dir>/                              # 默认 ../output/inference/full_pipeline
├── xic-roi-batch/<样本>/                  # ROI 阶段 ← 需求11：改名 xic_roi
├── batch_predictions/<样本>/              # ← 需求1：改名 predictions_model
│   ├── prediction.csv                     # ← 需求3：改名 model_prediction_<样本>.csv + 列调整
│   ├── qc_prediction_threshold.csv        # ← 需求4：改名 qc3_threshold_<样本>.csv（防线3）
│   └── predicted_plots/                   # ← 需求2：--plot 默认改走 XIC 曲线图 model_plots/；predicted_plots 保留为 --plot_style roi 可选项
├── snr_filtered/<样本>/SNR_box_3/         # ← 需求5+6：改名 prediction_refined，去掉 SNR_box 层
│   ├── box_outside_snr_report.csv         # ← 需求9：改名 qc4_snr_<样本>.csv（防线4）
│   ├── prediction_snr.csv
│   ├── feature.csv / roi_windows.csv / xic_matrix.npy
│   ├── prediction_refined.csv / prediction_refined_with_area.csv
│   ├── snr_kept/ / snr_dropped/           # ROI 红框图（snr_filter 默认 save_jpeg=True）
│   └── refined_plots/                     # ← 需求7：增强 query+置信度标注
├── inference_report.md / inference_report_all.csv   # cli.py:1304-1316 自动生成
└── pipeline_timing.log / pipeline_timing_runs.jsonl
../output/QC/<实验名>/
├── qc_label_rt.csv                        # ← 改名 qc1_label_rt.csv（防线1；README 去"（新）"）
├── qc_roi_channels.csv                    # ← 改名 qc2_roi.csv（防线2）
├── qc_prediction_threshold.csv            # ← 改名 qc3_threshold.csv（防线3）
├── qc_snr_boxes.csv                       # ← 改名 qc4_snr.csv（防线4）
├── qc_post_refinement.csv                 # ← 改名 qc5_refined.csv（防线5；新增样本级 qc5_refined_<样本名>.csv）
└── qc_summary.md / qc_alert.md
```

**目标**：

```
<output_dir>/                              # 默认 ../output/inference/<实验模式>_<实验名>（如 pipeline_demo1，需求13）
├── xic_roi/<样本>/…                                              # ← 需求11：原 xic-roi-batch
├── predictions_model/                                            # ← 原 batch_predictions
│   ├── predictions_model_all.csv                                 # 需求1：模型输出整体汇总（全样本，含 stem 列，放阶段文件夹内）
│   ├── predictions_model_report.md                               # 需求1：模型输出结果报告
│   └── <样本>/
│       ├── model_prediction_<样本名>.csv                         # 需求3：列见 §1.2
│       ├── qc3_threshold_<样本名>.csv                            # 需求4：预测阈值样本内 QC（防线3，列不变）
│       ├── model_plots/                                          # 需求2：--plot 默认生成，XIC 曲线图
│       │   └── <image_stem>_model.png
│       └── [predicted_plots/]                                    # 可选：--plot_style roi 时的 ROI 画框图（功能保留）
├── prediction_refined/                                           # ← 原 snr_filtered
│   ├── prediction_refined_all.csv                                # 需求5：修正后整体输出（全样本，含 stem 列，放阶段文件夹内）
│   ├── prediction_refined_report.md                              # 需求5：修正后输出报告
│   └── <样本>/                                                   # 需求6：不再有 SNR_box_<thr>/ 一层
│       ├── qc4_snr_<样本名>.csv                                  # 需求9：SNR 样本内 QC（防线4，原 box_outside_snr_report.csv）
│       ├── qc5_refined_<样本名>.csv                              # 精修门控样本内 QC（防线5，门控列子集）
│       ├── prediction_snr.csv                                    # （冗余分析见 §3，本次不动）
│       ├── feature.csv / roi_windows.csv / xic_matrix.npy        # （冗余分析见 §3，本次不动）
│       ├── prediction_refined.csv / prediction_refined_with_area.csv
│       ├── snr_kept/ / snr_dropped/                              # 可选（--save_snr_jpeg）
│       └── refined_plots/                                        # 需求7：增强标注后的 XIC 图
├── inference_report_<实验名>.md                                  # 需求13：全局推理报告（原 inference_report.md）
├── all.csv                                                       # 需求13：全局合并明细（原 inference_report_all.csv）
└── pipeline_timing.log / pipeline_timing_runs.jsonl              # 不变
../output/QC/<实验模式>_<实验名>/          # run_name = base_out.name，自动跟随新目录命名
├── qc1_label_rt.csv              # 防线1：标注 RT 一致性（原 qc_label_rt.csv）
├── qc_alert.md                   # 防线1 人工预警报告
├── qc2_roi.csv                   # 防线2：ROI 通道级剔除汇总（原 qc_roi_channels.csv）
├── qc3_threshold.csv             # 防线3：预测阈值全样本汇总（原 qc_prediction_threshold.csv）
├── qc4_snr.csv                   # 防线4：SNR 全样本汇总（原 qc_snr_boxes.csv；使用阈值记录于 qc_summary.md 第 4 节）
├── qc5_refined.csv               # 防线5：精修门控全样本汇总（原 qc_post_refinement.csv）
└── qc_summary.md
```

命名澄清：目录 `prediction_refined/` 与样本内文件 `prediction_refined.csv` 同名不同层级（README 目录树需注明），保留现状以减少读方改动（`cli.py:652`、`reprocess.py` 等按文件名定位）。

### 1.2 model_prediction_<样本名>.csv 目标列（需求 3）

| 列名 | 来源 | 说明 |
|---|---|---|
| `compound_name` | 原 prediction.csv | 不变 |
| `mz` | 原 prediction.csv | **即 Q1（母离子 m/z）**，保留；q3 删除 |
| `old_rt` | 原 prediction.csv | label 给出的标称 RT，保留 |
| `box_x1` `box_y1` `box_x2` `box_y2` | 原 prediction.csv | 像素框，保留 |
| `score` | 原 prediction.csv | 置信度 |
| `peak_start` | 原 `rt_min` | **改名** |
| `peak_end` | 原 `rt_max` | **改名** |
| `retention_time` | 原 prediction.csv | 模型预测峰顶 RT |
| `intensity_max` `area` `point_counts` `snr` `noise_std` `baseline_slope` `peak_width_ratio` `dynamic_range` `integration_method_used` | 原 prediction.csv | 不变 |
| `peak_index` | **新增（最右列）** | 同一 ROI 图/XIC 内按 `retention_time` 升序编号 1..n（一张图可检出多峰） |
| ~~`image_path`~~ ~~`q3`~~ | — | **删除**（`image` 列按用户反馈**保留**，继续作跨表定位键） |

`image` 列保留（用户确认）：它是 SNR 阶段、绘图、QC 汇总共用的主键，保留后跨表对齐零行为变更；`peak_index` 直接按 `image` 分组计算，落在最右列。

### 1.3 全部需求 → 实施项映射

| 需求 | 内容 | 落点 |
|---|---|---|
| 1 | batch_predictions→predictions_model；根级汇总 csv + 报告 md | Step 1 + Step 7 |
| 2 | --plot 默认生成 model_plots/（XIC 曲线，多 query 阴影+置信度信息框）；`--plot_style roi` 仍可生成 ROI 画框图（predicted_plots），**功能保留不删** | Step 6 |
| 3 | prediction.csv→model_prediction_<样本>.csv，删 image_path/q3 两列，2 列改名，加 peak_index | Step 3 |
| 4 | qc_prediction_threshold.csv→qc3_threshold_<样本>.csv（+QC 根 qc3_threshold.csv） | Step 4 |
| 5 | snr_filtered→prediction_refined；根级汇总 csv + 报告 md | Step 2 + Step 7 |
| 6 | 去掉 SNR_box_<thr>/ 一层 | Step 2 |
| 7 | refined_plots 同样标注所有 query 与置信度 | Step 6（共享绘图核心） |
| 8 | 样本内 6 个 csv 去冗余分析（**只分析**） | §3.1–§3.2 |
| 9 | box_outside_snr_report.csv→qc4_snr_<样本>.csv + QC 根 qc4_snr.csv | Step 5 |
| 10 | xic-roi-batch 中间表与 snr 阶段重复的分析（**只分析**） | §3.3 |
| 11 | xic-roi-batch 目录改名 xic_roi | Step 1b |
| 12 | QC 五防线编号统一：qc1_label_rt / qc2_roi / qc3_threshold / qc4_snr / qc5_refined；README 去"（新）" | Step 4 / Step 8 |
| 13 | 全局报告改名迁移：`inference_report_<实验名>.md` + `all.csv` @ `../output/inference/<实验模式>_<实验名>/`；输出目录命名改 `{mode}_{exp_name}`；**删除旧接口，无兼容** | Step 7 |

---

## §2 分步执行计划

> 每步独立可验证、可提交。行号以当前代码为准，实施时以锚点文本为准。

### Step 1 目录改名：batch_predictions → predictions_model

**写方（3 处）**

| 文件 | 位置 | 旧 → 新 |
|---|---|---|
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1019) | `pred_root = base_out / "batch_predictions"` | `"predictions_model"` |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1352) | roi2inference 模式默认 `../output/inference/batch_predictions` | `../output/inference/predictions_model` |
| [predictor.py](file:///d:/yinlibo/MRMPFormer/model/inference/predictor.py#L827-L830) | `--batch_output` 默认值及 help | 同上 |

**读方（同步改名或走 artifacts 兼容层，清单见 §4.1）**

| 文件 | 位置 | 处理 |
|---|---|---|
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L628) | QC 汇总 glob `batch_predictions/*/qc_prediction_threshold.csv` | 随 Step 4 一并改（新 glob + 新表名） |
| [reprocess.py](file:///d:/yinlibo/MRMPFormer/model/tools/batch/reprocess.py#L106) | `batch_root = result_root / "batch_predictions"` | 改名 + 旧名回退 |
| [masked_roi_generator.py](file:///d:/yinlibo/MRMPFormer/model/preprocessing/masked_roi_generator.py#L183-L186) | `--batch_predictions` 参数默认值 | 改名；参数名保留旧名或加 alias，help 注明 |
| [two_round_detection.py](file:///d:/yinlibo/MRMPFormer/model/inference/two_round_detection.py#L842) | `--batch_predictions` 默认值 | 同上 |
| [area_integration.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/area_integration.py#L684) | `--batch_output` 默认值 | 同上 |
| [plot_gt_vs_pred.py](file:///d:/yinlibo/MRMPFormer/model/tools/visualization/plot_gt_vs_pred.py#L155-L170) | `pred_root` 默认 `<pipeline_dir>/batch_predictions` | 改名 + 旧名回退 |
| [artifacts.py](file:///d:/yinlibo/MRMPFormer/model/tools/_shared/artifacts.py#L76-L85) | 已有旧新目录名兼容机制 | 扩展映射表：`batch_predictions→predictions_model`（旧目录优先级低于新目录） |
| README/CLAUDE.md/User_Tutorials | 全局搜索 `batch_predictions` | 文档同步（Step 8） |

**实现要点**
- `tools/_shared/artifacts.py` 的兼容映射保证旧产物目录仍可被诊断工具读取（先找新名，找不到回退旧名）。
- `masked_roi_generator` / `two_round_detection` 的 CLI 参数名 `--batch_predictions` 保留（避免脚本断裂），仅改默认值并在 help 注明新目录名。

**验证**
```powershell
# 跑一个双样本 pipeline（小数据），确认新目录生成且无 batch_predictions 残留
python -m inference.cli --mode pipeline --exp_name demo1 --labels <labels.xlsx> --mzml <dir> `
  --plot                          # 输出根默认 ../output/inference/pipeline_demo1/
Get-ChildItem ../output/inference/pipeline_demo1         # 应见 predictions_model/
```

### Step 1b 目录改名：xic-roi-batch → xic_roi（追加需求）

**写方（3 处）**

| 文件 | 位置 | 旧 → 新 |
|---|---|---|
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1019) | `roi_root = base_out / "xic-roi-batch"` | `"xic_roi"` |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1332) | roi 模式默认 `../output/inference/xic-roi-batch` | `../output/inference/xic_roi` |
| [xic_extraction.py](file:///d:/yinlibo/MRMPFormer/model/preprocessing/xic_extraction.py#L931-L932) | `--output_dir` 默认值 + help | 同上（help 中示例一并更新） |

**读方（全部命中点已 grep 核实）**

| 文件 | 位置 | 处理 |
|---|---|---|
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L620) | QC 汇总 glob `xic-roi-batch/*/pipeline_qc_excluded.csv` | 改 `xic_roi/...` |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L750) | `--output_dir` help 示例 | 同步 |
| [reprocess.py](file:///d:/yinlibo/MRMPFormer/model/tools/batch/reprocess.py#L107-L113)、[L272-L274](file:///d:/yinlibo/MRMPFormer/model/tools/batch/reprocess.py#L272-L274) | `result_root / "xic-roi-batch"` 两处 + 报错文案 | 改新名，旧名回退 |
| [verify_rt_axis.py](file:///d:/yinlibo/MRMPFormer/model/tools/diagnostics/verify_rt_axis.py#L89) | `result_root / "xic-roi-batch" / sample` | 改新名 + 回退 |
| [check_chrom_snr_alignment.py](file:///d:/yinlibo/MRMPFormer/model/tools/diagnostics/check_chrom_snr_alignment.py#L114-L167) | roi_windows 定位 + help | 同上 |
| [artifacts.py](file:///d:/yinlibo/MRMPFormer/model/tools/_shared/artifacts.py#L117-L146) | xic_matrix / roi_windows 候选路径列表 | 候选列表首位加 `xic_roi`，保留旧名候选兜底（旧产物可读） |
| [plot_gt_vs_pred.py](file:///d:/yinlibo/MRMPFormer/model/tools/visualization/plot_gt_vs_pred.py#L16-L172) | docstring、`roi_root` 默认、报错文案 | 新名优先回退旧名 |
| [plot_prediction_rt.py](file:///d:/yinlibo/MRMPFormer/model/tools/visualization/plot_prediction_rt.py#L107) | 按相对位置拼 `parent.parent.parent / "xic-roi-batch"` | 改新名 + 回退 |
| [evaluate_baseline.py](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/evaluate_baseline.py#L194) | feature.csv 定位 | 改新名 + 回退 |
| [refine_ablation.py](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/refine_ablation.py#L236-L246) | `pipe / "xic-roi-batch"` 两处 | 同上 |
| [inference_report.py](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/inference_report.py#L208-L368) | docstring、`roi_root` 默认、CLI help | 同上 |
| [masked_roi_generator.py](file:///d:/yinlibo/MRMPFormer/model/preprocessing/masked_roi_generator.py#L13-L191) | docstring 示例、`--images_root` 默认值 | 默认值改新名；docstring 同步 |
| [two_round_detection.py](file:///d:/yinlibo/MRMPFormer/model/inference/two_round_detection.py#L13-L843) | docstring、`--images_root` 默认值 | 同上 |
| [area_integration.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/area_integration.py#L7-L689)、[predictor.py](file:///d:/yinlibo/MRMPFormer/model/inference/predictor.py#L824) | 仅 help/docstring 示例 | 文案同步 |
| check_box_rt_mapping.py / regenerate_xic.py 等纯示例路径 | 文档字符串 | 文案同步 |

**实现要点**
- **诊断工具适配为硬性要求**（用户强调）：所有依赖该目录的诊断/可视化/评估工具必须**以 `xic_roi/` 新名为一等路径**原生支持新产物，不允许只依赖回退逻辑；统一收敛到 `tools/_shared/artifacts.py` 的定位助手实现（候选列表 xic_roi 在前、xic-roi-batch 仅作遗留数据兜底），并在 Step 9-③ 于新产物上逐一跑通作为验收项。
- 样本 stem 内含下划线与目录名无冲突（`xic_roi` 是根目录名，不影响 `<stem>` 解析）。

**验证**（可与 Step 1 合并执行）
```powershell
Get-ChildItem ../output/inference/pipeline_demo1        # 应见 xic_roi/
python -m tools.evaluation.inference_report --output_dir <新 pipeline_demo1> --exp_name demo1   # 新结构直读不报错
```

### Step 2 目录改名 snr_filtered → prediction_refined，去掉 SNR_box 层（需求 5、6）

**改动点**

| 文件 | 位置 | 旧 → 新 / 做法 |
|---|---|---|
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1020) | `snr_root = base_out / "snr_filtered"` | `"prediction_refined"` |
| [snr_filter.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/snr_filter.py#L397-L403) | `_snr_run_dir_name()` 返回 `SNR_box_<thr>` | 增加开关：pipeline 调用时直接写入 `output_dir` 本身（即样本目录），不再建子层 |
| [snr_filter.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/snr_filter.py#L702) | `--output_dir` help："其下创建 SNR_box_<阈值>/" | 更新说明：默认不建子层；保留 `--nested_run_dir` 兼容旧行为（独立运行时可指定） |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1167-L1178) | 定位 `SNR_box_<thr>` + `refined_root_dir` | `refined_root_dir = snr_root / stem`（去掉拼名逻辑） |
| [reprocess.py](file:///d:/yinlibo/MRMPFormer/model/tools/batch/reprocess.py#L401) | `--snr_subdir` 默认 `"SNR_box_3"` | 默认改为空（直接进样本目录），保留参数兼容旧目录重跑 |
| [reprocess.py](file:///d:/yinlibo/MRMPFormer/model/tools/batch/reprocess.py#L67-L70) | `SNR_box_%...` 目录名拼接 | 仅在 `--snr_subdir` 显式给出时使用 |
| [verify_rt_axis.py](file:///d:/yinlibo/MRMPFormer/model/tools/diagnostics/verify_rt_axis.py#L103-L104) | `snr_dir = .../snr_filtered/<样本>/SNR_box_*` | 改为 `prediction_refined/<样本>`，旧路径回退 |
| [check_chrom_snr_alignment.py / trace_refined_case.py / export_case_evidence.py] | 各自 `snr_filtered/SNR_box_*` 拼路径处 | 改为 `prediction_refined/<样本>` 一等路径（去层后直达），旧结构仅回退；grep `SNR_box` 全库收尾 |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L639) | QC glob `snr_filtered/*/SNR_box_*/box_outside_snr_report.csv` | `prediction_refined/*/qc4_snr_*.csv`（随 Step 5） |

**实现要点**
- `snr_filter.run()` 增加 `nested_run_dir: bool = False` 参数：False 时 `run_dir = output_dir`，True 时保持旧行为。pipeline（[cli.py:1153-1163](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1153-L1163)）传 False。
- **SNR 阈值信息不放样本目录层**（用户已确认）：去掉 SNR_box 子层后，阈值溯源由 QC 报告承担——pipeline 把本次 `--snr_min` 传入 `_collect_qc_tables`，在 `../output/QC/<实验名>/qc_summary.md` 第 4 节显式记录（如 "使用 SNR 阈值: 3.0"）；可选在 `qc4_snr_<样本>.csv` 追加常量列 `min_snr_used` 便于单样本独立追溯。不在样本目录留 meta 文件或任何子层。

**验证**
```powershell
Get-ChildItem ../output/inference/pipeline_demo1/prediction_refined -Recurse -Directory | Select-Object FullName
# 不应再出现 SNR_box_* 目录；prediction_snr.csv 等应直接位于 <样本>/ 下
python -m tools.batch.reprocess --stage post --snr_filtered_dir <...>/prediction_refined --dry-run
```

### Step 3 model_prediction_<样本名>.csv：改名 + 列调整（需求 3）

**改动点**

| 文件 | 位置 | 做法 |
|---|---|---|
| [predictor.py](file:///d:/yinlibo/MRMPFormer/model/inference/predictor.py#L750-L765) | `pred_basename = "prediction[_<method>].csv"`；`pred_out = output_base / subdir.name / pred_basename` | 批量分支改为 `model_prediction_{subdir.name}.csv`（非 linear 方法仍带后缀：`model_prediction_{subdir}_{method}.csv`） |
| [predictor.py](file:///d:/yinlibo/MRMPFormer/model/inference/predictor.py#L358-L381) | `rows.append({...})` 列构造 | ① 删 `'image_path'/'q3'` 两键（`image` 保留）；② `'rt_min'→'peak_start'`、`'rt_max'→'peak_end'`；③ 追加 `'peak_index'`（见下） |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1097-L1101) | `pred_basename` 定义（pipeline 内 snr 输入定位） | 同步新名；[cli.py:1138](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1138) `pred_csv` 拼接同步 |
| [snr_filter.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/snr_filter.py#L582-L652) | `rec = {k: row[k] for k in df_pred.columns ...}`、`out_row` 复制 prediction 列 | 列名驱动复制，自动跟随；确认无硬编码 `rt_min/rt_max`（grep 收尾：`box_outside` 计算、`save_roi_jpeg_with_box` 内的窗口取值改用新列名） |
| [peak_refinement.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py) | 读 prediction.csv 的 `rt_min/rt_max/old_rt` 处 | grep `rt_min|rt_max` 全库，凡读模型预测表的改新列名（`roi_windows.csv` 的 `rt_lo/rt_hi` **不改**，属于 ROI 表） |

**peak_index 计算规则**（predictor 内，写出前）：
```python
df = pd.DataFrame(rows)
df["peak_index"] = (
    df.groupby("image")["retention_time"]      # 按 image 分组：同一张 ROI 图内的多个峰
      .rank(method="first").astype(int)        # 图内 RT 升序编号，1..n
)
df.drop(columns=["image_path", "q3"], inplace=True)   # image 列保留落盘
```
注意：`peak_index` 排在 `integration_method_used` 之后、置于最右列。

**风险与对策**
- `image` 列保留后，主键链路（predictor → snr_filter → 绘图/QC）零行为变更。删除的两列中，`image_path` 是绝对路径（换机即失效，本就该清理）、`q3` 与 feature.csv 重复（mz 已保留作为 Q1）。
- 外围读方收尾：grep 引用 prediction 表 `image_path` / `q3` 列的脚本（add_compare_error.py、standard_curves.py、two_round_detection.py、masked_roi_generator.py 等），改为缺失容忍（`row.get()`）或从 feature 表取值。
- 单图多 query：`predict_utils.predict`（[predict_utils.py:38-84](file:///d:/yinlibo/MRMPFormer/model/utils/predict_utils.py#L38-L84)）`keep` 后同一图可产生多框，rows 逐框 append，天然支持多行/图——`peak_index` 正是为该场景引入。

**验证**
```powershell
# 列检查
python -c "import pandas as pd; df=pd.read_csv(r'.../predictions_model/S1/model_prediction_S1.csv'); print(df.columns.tolist()); assert 'image' in df.columns; assert 'image_path' not in df.columns; assert 'q3' not in df.columns; assert df.columns[-1]=='peak_index'"
# 多峰图：peak_index 应从 1 连续编号
```

### Step 4 QC 命名统一（防线 1–3）：qc1_label_rt / qc2_roi / qc3_threshold_<样本名>.csv

**防线 3：预测阈值样本内 QC（需求 4）**

| 文件 | 位置 | 做法 |
|---|---|---|
| [predictor.py](file:///d:/yinlibo/MRMPFormer/model/inference/predictor.py#L573-L582) | `qc_pred_path = .../qc_prediction_threshold.csv` | 改 `qc3_threshold_{样本名}.csv`（样本名=输出目录名；单图模式为 `prediction_output` 父目录名）；批量分支在 [predictor.py:760-L765](file:///d:/yinlibo/MRMPFormer/model/inference/predictor.py#L760-L765) 落盘处同步 |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L626-L635) | QC 汇总 glob + 输出名 `qc_prediction_threshold.csv` | glob `predictions_model/*/qc3_threshold_*.csv`，输出 `qc3_threshold.csv`（stem 列已由 `_merge_qc_csvs` 添加） |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L688-L691) | `qc_summary.md` 第 3 节标题 | 更新文件名引用 |

列不变（`image/n_queries/n_kept/n_dropped/max_confidence`，[predict_utils.py:75-84](file:///d:/yinlibo/MRMPFormer/model/utils/predict_utils.py#L75-L84)）；含 `image` 列，可兼作逐图对照表。

**防线 1 / 防线 2 重命名（需求 12）**

| 文件 | 位置 | 旧 → 新 |
|---|---|---|
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1037-L1040) | 写入 `qc_label_rt.csv` / `qc_alert.md` 处 | `qc_label_rt.csv` → **`qc1_label_rt.csv`**（`qc_alert.md` 名称保留） |
| [label_qc.py](file:///d:/yinlibo/MRMPFormer/model/preprocessing/label_qc.py) | docstring 中产物名 | 同步 |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L618-L624) | 汇总输出 `qc_roi_channels.csv` | → **`qc2_roi.csv`** |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L684-L686) | `qc_summary.md` 第 2 节标题引用 | 同步 |
| 其他读方 | grep `qc_label_rt|qc_roi_channels` 全库收尾（诊断/评估工具、README、User_Tutorials） | 改新名 + 可选旧名回退 |
| [README.md](file:///d:/yinlibo/MRMPFormer/README.md#L78) | 防线表中"标注 RT 一致性**（新）**" | 去掉"（新）"；产物路径列更新为新命名 |

**样本内 / 汇总命名规则**（本步起全管线统一）：带 `<样本名>` = 样本内 QC 明细；不带 = 整轮推理全样本汇总（QC 根目录）。

### Step 5 QC 命名统一（防线 4–5）：qc4_snr_<样本名>.csv + qc5_refined_<样本名>.csv

**防线 4：SNR 样本内 QC（需求 9）**

| 文件 | 位置 | 做法 |
|---|---|---|
| [snr_filter.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/snr_filter.py#L596-L598) | `rep_path = run_dir / "box_outside_snr_report.csv"` | 改 `qc4_snr_{样本名}.csv`（样本名=run_dir 名，即样本目录；独立 CLI 模式下取 `--output_dir` 目录名或显式 `--sample_name`） |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L637-L646) | QC 汇总 glob `snr_filtered/*/SNR_box_*/box_outside_snr_report.csv` → 输出 `qc_snr_boxes.csv` | glob `prediction_refined/*/qc4_snr_*.csv`，输出 **`qc4_snr.csv`**；[qc_summary.md 第 4 节标题](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L693-L697) 同步（并按 Step 2 记录使用阈值） |
| [reprocess.py](file:///d:/yinlibo/MRMPFormer/model/tools/batch/reprocess.py) / 诊断工具 | grep `box_outside_snr_report` | 同步或旧名回退 |

**防线 5：精修门控——新增样本级 QC（需求 12）**

现状：精修门控列只存在于根级 `qc_post_refinement.csv`（聚合自 prediction_refined.csv 的门控列子集），样本内无独立 QC 表。本次补齐：

| 文件 | 位置 | 做法 |
|---|---|---|
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1248-L1252) 附近（每样本 post 结束后） | 新增写盘 | 把 `_QC_POST_GATE_COLS` 门控列子集逻辑（现内联于 [cli.py:651-665](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L651-L665)）抽成辅助函数，逐样本落盘 `prediction_refined/<样本>/qc5_refined_<样本名>.csv` |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L648-L670) | 根级聚合改为直接合并各样本 qc5_refined_*.csv | 输出 **`qc5_refined.csv`**（原 `qc_post_refinement.csv`），保留 stem 列与 need_manual_review 计数 |
| [peak_refinement.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py) post_newtest 独立模式 | 可选 | 提供同样的门控子集输出（`--emit_qc5`），保证脱离 pipeline 重跑时也能产出样本级 QC；不作为本次硬性要求 |
| 其他读方 | grep `qc_post_refinement` 全库收尾 | 改新名 + 可选旧名回退 |

语义说明：`qc4_snr_*.csv` 是"每框 SNR 判定证据表"（全量框含未通过），与 `prediction_snr.csv`（仅保留框）的列级重复关系见 §3.1——**本次只改名不去列**，去冗余待 §3 方案评审后另行实施。

### Step 6 共享绘图核心：XIC 曲线 + 多 query 阴影 + 信息框（需求 2、7）

**前置确认（用户问题：当前 snr 输出的图是否真是 XIC 曲线？——是，已核实）**

snr 阶段实际有两类图，**均为真实 XIC 曲线绘制（RT × 强度数据），不是在 ROI 像素 jpeg 上贴框**：

| 图 | 生成处 | 内容 |
|---|---|---|
| `snr_kept/`、`snr_dropped/*.jpeg` | `save_roi_jpeg_with_box` [snr_filter.py:261-394](file:///d:/yinlibo/MRMPFormer/model/postprocessing/snr_filter.py#L261-L394) | `ax.plot(rt_sec/60, intensity, blue, lw=1.5)`（L292）；apex±1 min 窗口；prediction 像素框经 `box_x_to_rt_minutes`+`_pixel_y_to_intensity` 映射回 RT/强度坐标画红框 + score 白字红底标签；`#c0392b` 虚线=积分RT；左上角 fig 级信息框。**风格上隐藏刻度与边框**（`set_xticks([])` L389-392，figsize 4×3@100dpi = 400×300 px），以复刻模型输入 ROI 图观感 |
| `refined_plots/*_refined.png` | `_plot_refined_predictions` [peak_refinement.py:1990-2088](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L1990-L2088) | 同为 XIC 曲线但带完整坐标轴（xlabel="Retention Time (min)" / ylabel="Intensity"）、多色区间阴影、轴内信息框 |

**决策**：需求 2 已指定 model_plots "类似于 refined_plots"，故 `model_plots` 与 `refined_plots` 统一采用下述共享绘图核心的**带轴分析版**样式；`snr_kept/dropped` 的无刻度紧凑版保持现状不动（其用途是和 ROI 输入图并排人工比对）。三张图同源同数据、仅版式两档。

两处需求同构（`model_plots` 与 `refined_plots`），抽一个共享绘图模块，避免两份平行实现漂移：

**新增 `model/utils/plot_xic_peaks.py`**（唯一新文件，predictor 与 postprocessing 均可 import，不依赖 tools）：

```python
def plot_xic_with_queries(
    ax,                         # 调用方创建的 Axes
    rt: np.ndarray,             # 分钟
    intensity: np.ndarray,      # 原始强度（函数内按 sigma 平滑，非负裁剪）
    queries: list[dict],        # 每元素: {rt_lo, rt_hi, rt_peak, score, height, snr, n_points, label}
    q1: float,                  # 母离子 m/z
    sigma: float = 1.0,
    title: str = "",
    show_roi_window: tuple[float, float] | None = None,
) -> None
```

**视觉规格（与现有 refined_plots 对齐并扩展）**

| 元素 | 规格 | 依据 |
|---|---|---|
| 曲线 | 蓝色实线 lw=1.5，`gaussian_filter1d` 平滑后非负 | 现 [peak_refinement.py:2031-2053](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L2031-L2053) |
| 横轴/纵轴 | RT (min) / Intensity；秒制自动 /60（nanmax>200 判定） | 现 [peak_refinement.py:2122-2124](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L2122-L2124) |
| query 阴影 | `axvspan`，逐 query 不同颜色；**第 1 个沿用 `#2ecc71` alpha=0.24（淡绿，即现 Main interval）**，其后循环 `#f39c12`(0.22)/`#8e44ad`(0.18)/`#16a085`(0.16)/`#2980b9`(0.14)/`#c0392b`(0.12)，label 写 `query#k (score=0.xx)` | 现 [peak_refinement.py:2058-2071](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L2058-L2071) 色板扩展 |
| 置信度 | **不写在阴影上**；进信息框 + legend label | 需求 2 |
| 信息框 | 左上角 `ax.text(0.02, 0.98, ..., transform=ax.transAxes, va="top", bbox=dict(round, white, alpha=0.92))`；**多 query 时逐 query 一段**，字段：`RT(保留时间) / Q1(母离子m/z) / 峰高 / SNR / 峰区间扫描点数 / 置信度` | 现 `_annotate_refined_plot_axes` [peak_refinement.py:1898-1987](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L1898-L1987) 扩展 |
| 扫描点数 | 峰区间 `[rt_lo, rt_hi]` 内强度>0 的最长连续点数（**不是整张 ROI 图**） | 现 `_scan_points_in_interval` [peak_refinement.py:80-99](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L80-L99) |
| 图例 | upper right，fontsize=8 | 现状 |

信息框多 query 排版示例（每 query 一段，段间空行）：
```
query#1  RT=3.4212 min  Q1=152.0567  峰高=1.23e5  SNR=18.6  点数=14  置信度=0.97
query#2  RT=5.1084 min  Q1=152.0567  峰高=8.9e3   SNR=4.2   点数=7   置信度=0.83
```

**接入点 A：model_plots（predictor，需求 2）——XIC 为默认，ROI 画框降级为可选项**

新增 CLI 参数 `--plot_style {xic,roi}`（默认 `xic`；pipeline/roi2inference/单图模式通用，cli 透传 predictor）：`--plot` 开启绘图后按 style 分流。

| 文件 | 位置 | 做法 |
|---|---|---|
| [predictor.py](file:///d:/yinlibo/MRMPFormer/model/inference/predictor.py#L387-L483) | `_plot_predictions_with_baseline`（ROI 画框 + 基线叠加） | **保留不删**，仅在 `plot_style="roi"` 时调用（保持现输出目录名 `predicted_plots/` 与文件名 `<image_stem>_pred.png`）；新增函数 `_plot_model_xic` 在默认 `plot_style="xic"` 时调用：数据用内存中的 `xic_list / roi_windows / results(boxes,scores) / rows`，区间直接用 rows 的 `peak_start/peak_end`（已算好），经共享绘图核心出图 |
| [predictor.py](file:///d:/yinlibo/MRMPFormer/model/inference/predictor.py#L560-L569) | `build_predictor(plot=False)` → 延后绘制 | 积分后按 `args.plot_style` 分流到 `_plot_model_xic`（→ `model_plots/<image_stem>_model.png`）或既有 ROI 画框路径（→ `predicted_plots/`） |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1110-L1116)、[L1358](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1358) | `plot_dir="predicted_plots"` / Namespace 组装 | 增加 `plot_style=args.plot_style` 字段；plot_dir 按 style 取 `model_plots`（xic，默认）或 `predicted_plots`（roi） |
| [predict_utils.py](file:///d:/yinlibo/MRMPFormer/model/utils/predict_utils.py#L108-L162) | `plot_results`（ROI 红框） | **保留**：仍被 fullscan、单图调试及 `plot_style=roi` 的独立调用使用 |

原方案"由 `_plot_model_xic` 替代 ROI 画框"作废（用户已确认功能不删除）。

**接入点 B：refined_plots 增强（peak_refinement，需求 7）**

| 文件 | 位置 | 做法 |
|---|---|---|
| [peak_refinement.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L1990-L2088) | `_plot_refined_predictions` | 内部改为构造 `queries` 列表（main 区间 + small/small2/small3 区间，各自 `rt_lo/rt_hi/rt_peak/height/snr/点数/score`）后调共享 `plot_xic_with_queries`；置信度取 `main_score_ai / small_conf_* / final_conf_best`（列见 [prediction_refined 列清单](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L2883-L2926)） |
| [peak_refinement.py](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L1898-L1987) | `_annotate_refined_plot_axes` | 保留标称 RT 竖线（`#c0392b` 虚线）与主峰 RT 竖线逻辑，作为共享函数的可选叠加项；信息框文本切换为共享函数生成的多 query 版 |

**验证**
```powershell
# --plot 跑通后人工抽查两类图
Get-ChildItem ../output/inference/pipeline_demo1/predictions_model/*/model_plots
Get-ChildItem ../output/inference/pipeline_demo1/prediction_refined/*/refined_plots
# 检查点：① 多 query 不同颜色阴影；② 阴影上无置信度文字；③ 信息框含 RT/Q1/峰高/SNR/区间点数/置信度；④ Q1 来自 mz 列
# ROI 画框回归：--plot_style roi 时 predicted_plots/ 正常生成（红框+score+基线叠加与改版前一致）
```

### Step 7 根级汇总与全局报告（需求 1、5、13）

**7a. 输出目录命名：`{实验模式}_{实验名}`（需求 13）**

| 文件 | 位置 | 做法 |
|---|---|---|
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py) argparse | 新增 `--exp_name` 参数 | 缺省回退链：单个 mzML → 文件 stem；`--mzml`/`--batch_dir` 为目录 → 目录名；再兜底 UTC 时间戳。pipeline / roi2inference / fullscan 模式通用 |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1014) | `base_out = Path(args.output_dir) if args.output_dir else Path("../output/inference/full_pipeline")` | 默认改为 `../output/inference/{args.mode}_{exp_name}`（显式传 `--output_dir` 时尊重用户值） |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1016-L1017) | `run_name = base_out.name`、`qc_root = ../output/QC/<run_name>` | 无需改动，自动变为 `../output/QC/{mode}_{exp_name}` |
| [predictor.py](file:///d:/yinlibo/MRMPFormer/model/inference/predictor.py#L827-L830)、[xic_extraction.py](file:///d:/yinlibo/MRMPFormer/model/preprocessing/xic_extraction.py#L931-L932) 等 | 各模式独立默认路径 | 同步 `{mode}_{exp_name}` 风格（Step 1/1b 已列，命名规则此处统一） |

**7b. 阶段级汇总（需求 1、5）**

复用 `_merge_qc_csvs` 模式（glob → concat → stem 列 → utf-8-sig），在 pipeline 收尾新增 `_write_predictions_model_summary(base_out)` 与 `_write_prediction_refined_summary(base_out)`：

- **predictions_model_all.csv**：glob `predictions_model/*/model_prediction_*.csv`，合并 + `stem` 列。
- **predictions_model_report.md**：样本数 / ROI 图数 / 检出框数 / 阈值丢弃数（读 `qc3_threshold_*.csv`）/ 每样本一行统计表（样本、图数、检出峰数、最高置信度、平均 SNR）。
- **prediction_refined_all.csv**：glob `prediction_refined/*/prediction_refined_with_area.csv`（含面积版本为准），合并 + `stem`。
- **prediction_refined_report.md**：每样本一行（保留框数 / SNR 通过数（读 `qc4_snr_*.csv`）/ 主峰修正数 / 需人工复核数（读 `qc5_refined_<样本名>.csv` 门控列）/ 面积合计）。

**7c. 全局报告改名迁移（需求 13）：删除旧接口，直接新命名**

改造 [tools/evaluation/inference_report.py](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/inference_report.py)，**不留旧名兼容、旧写盘逻辑删除**：

| 文件 | 位置 | 旧 → 新 |
|---|---|---|
| inference_report.py docstring | L7-12 示例 | 更新为新文件名与新目录示例 |
| [inference_report.py](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/inference_report.py#L306-L334) `generate_for_pipeline` | 签名增加 `exp_name` 入参（cli 调用处传入）；落盘 `inference_report.md` → **`inference_report_{exp_name}.md`**；`inference_report_all.csv` → **`all.csv`** | 返回 dict 的 report_md 键同步 |
| [inference_report.py](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/inference_report.py#L313-L314) | 内部默认 `snr_filtered` / `xic-roi-batch` | `prediction_refined` / `xic_roi`（走 artifacts 定位助手） |
| [inference_report.py](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/inference_report.py#L215-L216)、[L228](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/inference_report.py#L228)、[L284](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/inference_report.py#L284) | 阶段表/明细说明中的旧文件名文案 | 改 `all.csv` |
| [inference_report.py](file:///d:/yinlibo/MRMPFormer/model/tools/evaluation/inference_report.py#L365-L368) 独立 CLI | `--output_dir` 默认 full_pipeline、help 中旧名 | 新增 `--exp_name`（同 7a 回退链）；`--output_dir` 缺省按 `{mode}_{exp_name}` 拼装；docstring 同步 |
| [cli.py](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1304-L1316) | `generate_for_pipeline(...)` 调用 | 传入 `exp_name=args.exp_name, mode=args.mode`；打印路径更新 |
| 全库收尾 | grep `inference_report` | 凡引用固定名 `inference_report.md` / `inference_report_all.csv` 的脚本/文档直接改新名，**无回退分支** |

定位规则：全局报告固定位于 `../output/inference/<实验模式>_<实验名>/` 目录根（即 base_out 根），md 与 csv 同层。

### Step 8 文档与命名规范同步

| 文件 | 内容 |
|---|---|
| [README.md](file:///d:/yinlibo/MRMPFormer/README.md#L322-L347) | 输出目录树替换为新结构（根目录 `{mode}_{exp_name}`、报告 `inference_report_{exp_name}.md`+`all.csv`）；QC 防线表（L74-112）：①"标注 RT 一致性（新）"去"（新）"；②产物路径列统一更新为五防线新命名（qc1_label_rt / qc2_roi / qc3_threshold / qc4_snr / qc5_refined），并补"带 `<样本名>` = 样本内 QC、不带 = 全样本汇总"规则说明 |
| User_Tutorials.md / CLAUDE.md | grep `batch_predictions|xic-roi-batch|snr_filtered|SNR_box|predicted_plots|box_outside_snr_report|qc_prediction_threshold|qc_snr_boxes|qc_label_rt|qc_roi_channels|qc_post_refinement|inference_report` 全部更新 |
| [docs/plan_products.md](file:///d:/yinlibo/MRMPFormer/docs/plan_products.md) | 追加一行指向本文档的链接（§2.2 决策状态更新：SNR 目录三件套冗余分析已扩展至本文档 §3） |

### Step 9 回归验证（全链路）

```powershell
# 1) 全链路（含绘图）
python -m inference.cli --mode pipeline --exp_name demo1 --labels <labels.xlsx> --mzml <样本目录> `
  --plot --save_snr_jpeg          # 输出根默认 ../output/inference/pipeline_demo1/

# 2) 断言（脚本或人工核对）
#    a. predictions_model/ 下有 all.csv + report.md + 各样本文件夹
#    b. 样本内: model_prediction_<样本>.csv / qc3_threshold_<样本>.csv / model_plots/
#    c. prediction_refined/ 下有 all.csv + report.md；样本内无 SNR_box_* 层，直接是 qc4_snr_<样本>.csv / qc5_refined_<样本>.csv 等
#    d. QC 根 ../output/QC/pipeline_demo1/: qc1_label_rt.csv / qc2_roi.csv / qc3_threshold.csv / qc4_snr.csv / qc5_refined.csv 存在且行数正确
#    e. prediction 列: 保留 image；无 image_path/q3；有 peak_start/peak_end/peak_index
#    f. 根下 ROI 目录为 xic_roi/（无 xic-roi-batch 残留），QC 汇总 qc2_roi.csv 仍正常生成
#    g. 根目录有 inference_report_demo1.md + all.csv；全库无 inference_report.md/inference_report_all.csv 写盘残留
# 3) 诊断/评估/批量工具在新产物上一一跑通（硬性验收）
python -m tools.diagnostics.verify_rt_axis --result_root <新 pipeline_demo1> --sample <样本> ...   # 直达 prediction_refined/<样本>/
python -m tools.diagnostics.check_chrom_snr_alignment --result_dir <新 pipeline_demo1>
python -m tools.visualization.plot_prediction_rt --prediction_csv <新 model_prediction_<样本>.csv> ...
python -m tools.evaluation.inference_report --output_dir <新 pipeline_demo1> --exp_name demo1   # 直读新布局与新报告名
python -m tools.batch.reprocess --stage post --result_root <新 pipeline_demo1> --dry-run   # 缺省直达样本目录，无需 --snr_subdir
# 4) 旧目录兼容（历史数据回退抽查）
#    注：全局报告不参与回退——inference_report 旧接口已删除，旧 full_pipeline 目录如需报告须按新工具手工重跑
python -m tools.diagnostics.verify_rt_axis --result_root <旧 full_pipeline> ...   # 走 artifacts 回退
python -m tools.batch.reprocess --stage snr-post --snr_filtered_dir <旧目录> --snr_subdir SNR_box_3 --dry-run
# 5) 独立重跑
python -m postprocessing.snr_filter --mzml ... --prediction_csv <新名csv> --output_dir <样本目录>
python -m postprocessing.peak_refinement post_newtest --results_dir <样本目录> --xic_dir <roi目录> --plot
```

---

## §3 冗余分析（需求 8、10 —— 只分析，不实施）

### 3.1 prediction_refined/<样本>/ 内 6 张表的角色与重复

| # | 文件（重构后） | 角色 | 行粒度 | 关键内容 |
|---|---|---|---|---|
| T1 | `qc4_snr_<样本>.csv`（原 box_outside_snr_report） | SNR 判定证据表 | **全量框（含未通过）** | prediction 全列 + `snr_outside_box/noise_rms_outside_box/signal_outside_mean/noise_mean_outside_box/n_noise_points/rt_window_lo_hi_used/passed_snr_threshold/passed_min_chrom_points/passed_min_chrom_max_intensity/pred_row_index` |
| T2 | `prediction_snr.csv` | SNR 通过框 → 下游精修输入 | 通过框 | prediction 全列 + `snr_outside_box`；`compound_name` 重编号为紧凑 1..N |
| T3 | `feature.csv` | post_newtest 的通道元数据 | 通过框 | `Compound Name(=T2 重编号)/mz/q3/RT(=apex)` |
| T4 | `roi_windows.csv` | post 绘图/窗口定位 | 通过框 | `image(=snr_kept/<fname>)/rt_lo/rt_hi(apex±1min)` |
| T5 | `prediction_refined.csv` | 框修正结果 | 通过框 | `image/compound_name/mz/q3` + `main_*`（rt/peak/height/score_ai/snr/skew）+ `small_*`（含 small2/small3）+ 门控列 |
| T6 | `prediction_refined_with_area.csv` | 最终定量输出 | 通过框 | T5 全列 + 面积列 |

**列级重复矩阵**（≈ 完全重复 / 部分重复）：

| | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| T1 | — | T2 ⊂ T1（T2=T1 的 passed 子集，且 ~24 列逐列同值） | mz/q3/编号 3 列重复 | — | mz/q3/编号 3 列重复 | 同 T5 |
| T2 | | — | mz/q3/编号 3 列重复 | image 键重复 | mz/q3/编号 + 行一一对应（精修前后各一份） | 同 T5 |
| T3 | | | — | — | `Compound Name`↔`compound_name`、mz、q3 | 同 T5 |
| T4 | | | | — | image 键 | image 键 |
| T5 | | | | | — | T6 = T5 + 1..2 列（**行级 100% 重复**） |
| T6 | | | | | | — |

**冗余结论（建议，未实施）**
1. **T6 并入 T5**：面积作为 T5 的列（或 `--with_area` 开关控制是否多写面积列），消除一份整表复制。代价：下游读 T6 的地方（`inference_report`、`standard_curves`、实验脚本）改读 T5。
2. **T2 瘦身为索引表**：T2 与 T1 的重复最大（T2 是 T1 的行子集+全列复制）。可让 T2 只保留 `pred_row_index / image / compound_name(新编号) / snr_outside_box`，其余列经 `pred_row_index` 回查 T1。代价：post_newtest、peak_refinement、绘图读 T2 的 ~20 处取值需 join；**与"自包含跑 post"的设计取向冲突**（plan_products.md §2.2 已有保留决策），故列为低优先。
3. **T3 是 T2 的 3 列投影**（mz/q3/编号 + apex RT）。可由 T2 + apex 计算替代，但 post_newtest 以 feature.csv 为通道对齐契约（[peak_refinement `_resolve_xic_matrix_row`](file:///d:/yinlibo/MRMPFormer/model/postprocessing/peak_refinement.py#L290)），改动面大，建议保留、仅补 `native_id` 透传（现 SNR 版 feature.csv 丢失通道身份，见 §3.3）。
4. **T4 与 T1**：T4 的 `rt_window_lo/hi_used` 列与 T1 同名列信息重复（T1 记录用窗、T4 记重算窗，语义不同但易混淆）——若实施 2，建议 T4 列改名 `plot_rt_lo/plot_rt_hi` 消歧。
5. T1 本身两张身份（QC 证据 + T2 的母表）。用户已将其定名 `qc4_snr_*`（QC 身份固化，防线4），与 2 的"瘦身 T2"方案天然配套。

### 3.2 与 QC 根目录的第三份重复

- `T1 → ../output/QC/<实验名>/qc4_snr.csv`（+stem 列）：同一数据本地明细 + 全局汇总两份，属分层设计（每样本可独立重跑），建议保留。
- `T5 门控列子集 → qc5_refined.csv`（原 qc_post_refinement.csv）：仅取门控列，非全量复制，保留。

### 3.3 与 ROI 阶段中间表的跨阶段重复（需求 10）

> 注：本节"xic-roi-batch 版"按重构前称呼书写；重构后该目录为 `xic_roi/<样本>/`（见 Step 1b）。

xic_roi/<样本>/（原 xic-roi-batch）产物（[xic_extraction.py](file:///d:/yinlibo/MRMPFormer/model/preprocessing/xic_extraction.py#L578-L633)）：`feature.csv / roi_windows.csv / xic_matrix.npy / pipeline_qc_excluded.csv / *.jpeg`。

| 数据 | xic-roi-batch 版 | prediction_refined 版 | 重复性质 |
|---|---|---|---|
| feature.csv | 全通道、全局编号、`RT`=标注窗中心、含 `native_id` | 通过框子集、紧凑重编号、`RT`=平滑后 apex、**无 native_id** | **同名不同义**（最易踩坑）：编号体系不同（snr_filter.py:644-650 注释已警示）、RT 语义不同 |
| roi_windows.csv | `image`=裸 jpeg 名、窗=标注中心±1min 实际裁剪 | `image`=`snr_kept/<fname>` 前缀、窗=apex±1min 重算 | 同名不同义；artifacts.py:76-85 已按 basename 兼容 |
| xic_matrix.npy | 全通道、原始网格 | 保留行子集、重插值到统一 linspace 网格 | 数值可能不一致（插值），非简单复制 |
| pipeline_qc_excluded.csv | 本地明细 | —（汇总至 QC/qc2_roi.csv） | 明细+汇总分层，保留 |

第三处复制：[masked_roi_generator.py:303-307](file:///d:/yinlibo/MRMPFormer/model/preprocessing/masked_roi_generator.py#L303-L307) 又把 feature/roi_windows 原样拷贝到 `batch_predictions_masked/`（训练造数据用），属独立用途，建议保留。

**跨阶段冗余结论（建议，未实施）**
1. pipeline 模式下 post_newtest 已能直接读 roi 阶段数据（[cli.py:1187](file:///d:/yinlibo/MRMPFormer/model/inference/cli.py#L1187) 已传 `--xic_dir`），SNR 目录内 T3/T4/xic_matrix 三件套**仅"脱离 pipeline 单独重跑 snr_filter→post"时必需**。可加 `--no_copy_inputs` 开关：pipeline 内跑时不落这三件，直接引用 roi 目录 → 每样本省 3 文件与一次矩阵重插值。
2. 若保留三件套（自包含路线），最低限度应：① feature.csv 补 `native_id` 透传（找回通道身份）；② `RT` 列改名 `apex_rt` 或在 README 注明与 roi 版语义差异；③ 重编号列改用独立列名（如 `channel_kept_idx`），避免与 roi 版 `Compound Name` 编号混淆。
3. `pred_row_index`（T1）是衔接 prediction.csv 原行序的桥，删 `image` 列后价值上升，建议在任何瘦身方案中保留。

### 3.4 实施与否的决策入口

§3.1–§3.3 全部为分析结论，**不在本次重构中实施**；若后续采纳，按 §6 决策表逐项立独立任务（每项都涉及多读方，需单独回归）。

---

## §4 风险、兼容与决策

### 4.1 全库读方清单（改名影响面，grep 复核用）

| 关键字 | 主要命中 | 策略 |
|---|---|---|
| `batch_predictions` | cli.py / predictor.py / reprocess.py / masked_roi_generator.py / two_round_detection.py / area_integration.py / plot_gt_vs_pred.py / benchmark/runner.py / postprocessing/evaluation/* / artifacts.py | 写方+读方全部同步；artifacts.py 兼容映射兜底 |
| `xic-roi-batch` | cli.py ×4 / xic_extraction.py / reprocess.py ×2 / verify_rt_axis.py / check_chrom_snr_alignment.py / plot_gt_vs_pred.py / plot_prediction_rt.py / evaluate_baseline.py / refine_ablation.py / inference_report.py / masked_roi_generator.py / two_round_detection.py / area_integration.py / artifacts.py | 写方+读方全部同步；artifacts.py 候选列表加 `xic_roi` 首位、旧名兜底 |
| `snr_filtered` | cli.py / snr_filter.py / reprocess.py / verify_rt_axis.py / check_chrom_snr_alignment.py / trace_refined_case.py / export_case_evidence.py / 文档 | 同上 |
| `SNR_box` | snr_filter.py / cli.py / reprocess.py / verify_rt_axis.py / check_box_rt_mapping.py | 保留函数供独立模式，pipeline 路径去层 |
| `box_outside_snr_report` | snr_filter.py / cli.py / 诊断工具 | 改名 + 旧名回退 |
| `qc_prediction_threshold` / `qc_snr_boxes` | predictor.py / predict_utils.py / cli.py | 改名（qc3_threshold / qc4_snr；样本级+根级同步） |
| `qc_label_rt` / `qc_roi_channels` / `qc_post_refinement` | cli.py（写方）/ 诊断评估工具（读方）/ label_qc.py docstring | 改名 qc1_label_rt / qc2_roi / qc5_refined；qc5 新增样本级 `qc5_refined_<样本名>.csv` 落盘后聚合 |
| `rt_min` / `rt_max` | 全库（注意与 `rt_lo/rt_hi`、refined 表的 `main_rt_min` 等区分） | **仅模型预测表语境**改 peak_start/peak_end；ROI 表 `rt_lo/rt_hi` 与 refined 表 `main_rt_min/main_rt_max/small_rt_*` 不动 |
| `predicted_plots` | predictor.py / cli.py / peak_refinement 注释 | 功能保留：默认 `--plot_style xic` → `model_plots`；显式 `--plot_style roi` 时仍输出 `predicted_plots`（目录名/文件名/画法不变） |
| `inference_report.md` / `inference_report_all.csv`（固定名） | inference_report.py 写盘/文案、cli.py 调用、README/User_Tutorials 示例 | 改 `inference_report_{exp_name}.md` / `all.csv`；输出根改 `{mode}_{exp_name}`；**旧接口删除，无兼容分支** |

### 4.2 主要风险

1. **删 `image` 列的连锁**（Step 3）：最大的行为变更。需在 snr_filter、绘图、QC 生成三处确认均不依赖该列读盘（当前 snr_filter 按行+索引处理，兼容；`save_roi_jpeg_with_box` 文件名函数需复核）。
2. **同名不同义目录**（`prediction_refined/` 目录 vs `prediction_refined.csv` 文件）：README 目录树必须显式画出两层，防误读。
3. **诊断工具双布局适配**（用户硬性要求）：artifacts.py 改造与新路径切换先行合入，验收主项为「诊断工具在新产物上原生跑通」；旧目录回退仅为历史数据的兼容路径，验证矩阵须覆盖新产物直读与旧产物回退两组。
4. **报告多份并存**：阶段级 `predictions_model_report.md` / `prediction_refined_report.md` 与全局 `inference_report_{exp_name}.md` 内容有部分重叠，按 D3 已收口——阶段报告管"目录内快览"，全局报告管"跨样品汇总+QC+补算面积"，职责固定，不做进一步合并。

### 4.3 验证清单（汇总）

- [ ] Step 1：新目录生成，无 batch_predictions 残留；旧目录可被 artifacts 兼容读取
- [ ] Step 1b/2：无 xic-roi-batch / SNR_box_* 残留；reprocess --dry-run 双模式（新/旧）均可定位
- [ ] Step 3：列断言（保留 image；无 image_path/q3；peak_start/peak_end/peak_index 就位；同图内 peak_index 自 1 连续）
- [ ] Step 4/5：QC 根 qc1_label_rt.csv / qc2_roi.csv / qc3_threshold.csv / qc4_snr.csv / qc5_refined.csv 齐全、行数正确；qc_summary.md 第 4 节记录实际使用的 SNR 阈值
- [ ] Step 6：model_plots 与 refined_plots 均含多 query 阴影 + 信息框六要素；阴影无置信度文字；`--plot_style roi` 的 predicted_plots 与改版前行为一致（回归）
- [ ] Step 7：四份根级产物（2 csv + 2 md）生成且 stem 列齐全
- [ ] Step 8：README 防线表去"（新）"并更新产物路径为新命名；CLAUDE.md/User_Tutorials 无旧名残留
- [ ] 诊断工具（verify_rt_axis、check_chrom_snr_alignment、plot_prediction_rt、plot_gt_vs_pred、evaluate_baseline、refine_ablation、inference_report）在新产物结构上逐一跑通
- [ ] Step 9：全链路 + 诊断工具新产物遍历 + 旧目录回退抽查 + 独立重跑（snr_filter / post_newtest / reprocess）不回归

### 4.4 决策记录

| # | 项 | 决策 | 状态 |
|---|---|---|---|
| D1 | QC 表编号体系 | **统一按 README 五道防线**：qc1_label_rt / qc2_roi / qc3_threshold / qc4_snr / qc5_refined；带 `<样本名>` = 样本内 QC，不带 = 全样本汇总；README 去"（新）" | [x]（用户已确认，2026-08-27 定稿） |
| D2 | predictor --plot 默认图型与 ROI 画框去留 | **默认生成 model_plots（XIC 曲线）**；ROI 画框功能保留为 `--plot_style roi` 可选项（predicted_plots，行为不变），不删除 | [x]（用户已确认） |
| D3 | inference_report.md/all.csv 与新两份报告的关系 | 全局报告改名为 `inference_report_{exp_name}.md` + `all.csv`（位于 `../output/inference/{mode}_{exp_name}/` 根），新增 `--exp_name` 参数；**旧固定名接口直接删除，无兼容**；阶段级两份汇总保留各司其职 | [x]（用户已确认） |
| D4 | SNR 目录内 T3/T4/xic_matrix 三件套 | 本次保留（自包含路线延续），冗余处置走 §3.4 独立任务 | [ ] |
| D5 | T2/T6 瘦身（§3.1 结论 1/2） | 不在本次实施 | [ ] |
| D6 | SNR 阈值溯源（去掉 SNR_box 目录层后） | 不留 meta 文件/子层；写入 `qc_summary.md` 第 4 节（`_collect_qc_tables` 增加阈值入参），`min_snr_used` 常量列可选 | [x]（用户已确认） |
| D7 | model_prediction 是否删 `image` 列 | **保留** `image`（用户确认），仅删 `image_path`/`q3`；`peak_index` 按 `image` 分组、落在最右列 | [x]（用户已确认） |
| D8 | 诊断工具兼容策略 | 新产物为一等路径（硬性验收项，Step 9-③ 逐一跑通）；旧产物仅走 artifacts 回退兜底 | [x]（用户已确认） |
