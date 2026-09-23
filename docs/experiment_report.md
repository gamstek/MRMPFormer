# 实验报告（Experiment Report）

> 记录模型实验的假设、证据、结论与影响。每篇实验独立成节，按时间倒序排列。
> 数据与命令均可复现，结论均给出证据出处。

---

## 实验日志 004：massnova 模型框双向基线精修、停阈值护栏、截断峰 SNR 守卫与全链路分模块计时（2026-09-23）

- **日期**：2026-09-23
- **基线**：昨日最后一次提交 `9fdef78`（2026-09-22 20:23「暂时提交」；原哈希 cf6727c，大文件迁入 Git LFS 改写历史后为 9fdef78）；本节改动已随本条 feat(model) 提交入库，`git diff 9fdef78 HEAD` 可完整复现
- **规模**：主仓 10 文件修改（+641/−138）＋ 8 个新增文件（合计 1513 行）＋ CW 子模块内 1 文件（+70/−1，子模块内未提交）
- **状态**：代码完成；新增 4 个单测文件 19 用例全部通过（0.011 s，见 §6）

### 1. 背景与动机（本日解决的四个问题）

| # | 问题 | 根因 | 对策 |
|---|---|---|---|
| ① | P4（已知问题表）：模型框直接作为最终边界，对非「单峰居中」形态（多头簇/双驼峰/偏心峰）框回归偏窄、偏移（莠去津-2 只框半个峰） | 训练分布为 2min 窗单峰居中 | 模型框从「最终边界」降级为「精修种子」：外推 + 内收双向基线校正（`refine_model_boundaries`） |
| ② | 半峰截断：test2 chrom052 甲羧除草醚-2（RT≈16.0 候选）积分面积仅完整峰的一半 | 右侧 ~1 min 处存在更高邻居峰，`stable_tail_mean` 尾窗（apex±1min）把邻居峰翼当本侧基线，估出停阈值达 apex 的 1.05~1.11 倍 → 内审在 apex 后一步即截停 | `_boundary_stop_levels` 加护栏：估计值 ≥ apex 时按 apex×0.2 封顶 |
| ③ | 截断峰 SNR 低估甚至 nan 误杀 | 峰尾被运行末端/窗口截断时，框外扇区全是陡峭下降尾，峰-峰噪声被抬高 | `compute_local_snr`/`compute_snr_outside_box` 加 `tail_reject_scale=3.0` 守卫 + 全迹安静点兜底 |
| ④ | 无分模块计时；torch(.pth) 路径逐图前向（batch_size 不生效） | 工程缺口 | massnova 五阶段计时进终端与报告；roi/roi2inference 补计时汇总；`predict` 支持批量前向 |

### 2. 修改总览

| 文件 | 类型 | 规模 | 主题 |
|---|---|---|---|
| `model/inference/massnova.py` | 修改 | +407/−83 | 模型框精修核心、停阈值护栏、五阶段计时、报告章节、参数接入 |
| `model/inference/two_round_detection.py` | 修改 | +11/−2 | `adjust_first_round_interval` 新增 `edge_threshold_scale` |
| `model/inference/cli.py` | 修改 | +34/−2 | 2 个新 CLI 参数；pipeline/roi/roi2inference 计时汇总 |
| `model/inference/massnova_runtime.py` | 修改 | +1 | 嵌入式默认配置补 `scan_model_boundary_baseline_ratio` |
| `model/utils/predict_utils.py` | 修改 | +174/−104 | torch 批量前向 + `_build_result` 公共抽取 |
| `model/utils/xic_peak_utils.py` | 修改 | +82/−28 | 截断峰噪声守卫 + 全迹安静点兜底 |
| `model/tools/evaluation/inference_report.py` | 修改 | +58/−5 | 推理报告新增「运行用时（各模块）」章节 |
| `model/configs/inference_pipeline.json` | 修改 | +1/−1 | `pipeline_min_max_intensity` 1000→3000 |
| `model/configs/massnova.json` | 修改 | +5/−5 | batch_size / threshold / scan_min_peak_ratio / scan_min_snr |
| `model/tests/test_*.py` ×4 | 新增 | 264 行 | 回归测试，见 §5 |
| `model/tools/evaluation/{run,evaluate,compare}_massnova_sim.py` | 新增 | 873 行 | 模拟集三档评测工具链 |
| `model/tools/diagnostics/trace_channel_queries.py` | 新增 | 376 行 | 逐层逐 query 诊断工具 |
| `CW/CW/Centwave`（子模块内） | 修改 | +70/−1 | `estimate_peak_bounds` 补全 + `np.trapz` 兼容 |

### 3. 逐文件逐处修改明细

#### 3.1 `model/inference/massnova.py`（核心，+407/−83）

**A. 模型框精修核心：新增 4 个函数（约 L382-586）**

| 函数 | 行为 |
|---|---|
| `_boundary_stop_levels(rt, y, apex_idx, edge_noise_stop_mode, edge_max_span_min, edge_noise_percentile=25.0, edge_threshold_scale=1.0, max_stop_apex_ratio=0.2)` | 复算与 `adjust_first_round_interval` 同口径的左右停阈值（三分支：`low_percentile`→`one_sided_low_noise_baseline` 单侧低分位；`roi_bottom_decile_mean`→全迹低 10% 均值双侧同值；默认 `stable_tail_mean`→`one_sided_edge_stop_threshold_stable_tail_mean` 峰侧稳定尾噪声），乘 `edge_threshold_scale`（非有限或 ≤0 回退 1.0）。**护栏**：估计值 ≥ apex 强度时封顶为 `apex_y × max_stop_apex_ratio`（默认 0.2）——邻居峰翼抬基线时防止阈值失去基线语义导致半峰截断 |
| `_first_stable_baseline_from_apex(rt, y, apex_idx, boundary_rt, stop_level, baseline_floor, side, lookahead, mean_scale)` | 内审游走：从 apex 向一侧边界找首个「稳定」基线穿越——单点 ≤ stop_level 不足够，还需沿方向 lookahead 个点均值 ≤ `baseline_floor + mean_scale×(stop_level−baseline_floor)`；方向/参数非法或找不到时返回原边界 |
| `_audit_model_interval_inward(...)` | 内收审核：对已外推的模型区间两侧独立游走，只收不放（`audited_lo = max(lo, natural_left)`、`audited_hi = min(hi, natural_right)`）；停阈值取 `baseline + baseline_proximity_ratio×(apex−baseline)`（默认 ratio=0.01，即基线上方 1% 动态高度带）；结果必须仍含 apex、区间有效且 ≥ `min_points`（取 `max(5, min_peak_span_points)`）个采样点，否则维持原区间 |
| `refine_model_boundaries(rt, y, candidates, scan_params)` | 对每个模型框执行「外推→内收」：先 `adjust_first_round_interval`（`min_secondary_ratio=0.05`、`boundary_peer_thr_scale=2.0`，透传 edge 模式/跨度/缩放与 posterior 参数），再 `_audit_model_interval_inward`；跑两遍（pass2 以 pass1 结果互为 peer 防撞），逐候选返回 `(rt_min, rt_max)` |

**B. `validate_with_model`（模型前置验证）**
- docstring：「命中 → 模型框 RT 即最终边界」→「命中 → 模型框 RT 作为后续精修种子」；补 `onnx_batch_size` 说明（ONNX 与 torch(.pth) 两条路径共用；`onnx_use_gpu` 仅对 ONNX 生效，torch 由 `utils.torch_device` 自动选 CUDA）；
- torch 路径批推理：`build_predictor(...)` 调用新增 `batch_size=max(1, int(onnx_batch_size or 1))`——候选窗口拼 batch 前向，摊薄逐图 kernel/预处理固定开销；
- 命中分支新增两行：`p["model_rt_min"] = float(left)`、`p["model_rt_max"] = float(right)`——保留模型原始框；后续精修只改 `rt_min/rt_max`，原始框供 `prediction_model/` 出图与追溯。

**C. `finalize_channel_peaks`（mzML 推理与嵌入式 C DLL 共用的最终峰处理核心）**
- docstring 更新：validated 模型框与信号兜底框**均**做边界精修；信号类质量门控仍只作用于兜底峰；
- 模型命中分支整体重写：原实现直接把候选 `rt_min/rt_max` 抄入 peaks；现改为——收集 `model_cands`（复制候选 dict，置 `boundary_source="model"`、`peak_score=model_score`）→ `_dedup_identical_peak_bounds` 去重（重叠候选窗观察到同一框时保「一框一事件」）→ `refine_model_boundaries` 精修 → 输出 peak 新增 `model_rt_min`/`model_rt_max` 两列（种子缺省回退候选区间），`rt_min/rt_max` 取精修结果；
- 信号兜底分支：`refine_all_boundaries(...)` 调用新增 `edge_threshold_scale=sp["edge_threshold_scale"]` 透传。

**D. `plot_massnova_stage_windows`（2min 窗口图）**
- 新形参 `model_seed=False`（docstring：True 时用 `model_rt_min/max` 模型原始框画图，用于 `prediction_model/`）；
- 新增内嵌 `_bounds(pk)`：model_seed=True 且模型框有限有效时返回原始框，否则返回最终精修边界；
- 窗口相交判定与 queries 构造的 `rt_lo`/`rt_hi` 均改走 `_bounds(pk)`。

**E. `_MASSNOVA_WINDOW_STAGES`**：三元组扩为四元组 `(文件夹, boundary_source, 标题前缀, 是否用模型原始框)`——`("prediction_model", "model", "Model prediction", True)`，signal/refined 为 False。

**F. `write_outputs`**：docstring 同步（prediction_model = 模型原始框，未精修）；阶段循环解包四元组并传 `model_seed=use_model_seed`。

**G. `write_massnova_report`**
- 新形参 `stage_seconds=None`（docstring：各阶段累计耗时，用于「运行用时（各模块）」章节）；
- 新增「## 4. 运行用时（各模块）」：总耗时行（分母优先 `total_seconds`）+ `| 模块 | 耗时 (s) | 占比 |` 表；无计时数据时输出提示；
- 原第 4/5 章（CSV 关键列 / 已知问题）顺延为第 5/6 章；CSV 关键列两行措辞更新（模型框与信号兜底框均经过稳定基线精修；`boundary_source=model`=以模型框为种子的信号精修边界）；
- 已知问题 **P4 状态 ⚠️ 未根治 → ✅**：推理端已增加模型框双向基线校正（框内边界仍高于基线时外推，覆盖多余基线时从 apex 向外审核并内收）；训练分布问题仍需后续重训；
- 对策参数清单：`P1/P2` → `P1/P2/P4`，追加 `scan_model_boundary_baseline_ratio`（默认 0.01）。

**H. `_scan_params_from_args`**：新增两键——`edge_threshold_scale = float(_g("scan_edge_threshold_scale", 0.8))`、`model_boundary_baseline_ratio = float(_g("scan_model_boundary_baseline_ratio", 0.01))`。

**I. `run_massnova_on_mzml`**：五阶段 `time.perf_counter()` 计时——`1_XIC抽取(Phase0)`、`2_候选枚举(Phase1)`、`3_模型验证(Phase3b)`、`4_边界精修与门控(Phase2/3a/4)`、`5_输出与绘图(Phase4)`；返回 info dict 新增 `stage_seconds`；docstring 流程描述同步（模型命中框执行双向稳定基线精修）。

**J. `main`**：跨样本聚合 `stage_seconds`（同名阶段累加）；有值时调 `_print_pipeline_timing_summary(mode_label="massnova", ...)` 打终端汇总，并传入 `write_massnova_report(..., stage_seconds=...)`。

**K. `build_parser`**：`--smooth_sigma` 默认 0.8→**0**（关闭平滑、强度保持原始值；`massnova.json` 中该键仍为 0.8，仅直连入口 `python -m inference.massnova` 的默认值变化）；新增 `--scan_edge_threshold_scale`（默认 0.8）与 `--scan_model_boundary_baseline_ratio`（默认 0.01）。

**L. 其他**：新增 `import time`；`utils.xic_peak_utils` 导入扩为 `compute_local_snr, one_sided_edge_stop_threshold_stable_tail_mean, one_sided_low_noise_baseline, roi_full_low_decile_mean_intensity`；模块头 Phase3b/Phase2/3a/Phase4 流程描述同步。

#### 3.2 `model/inference/two_round_detection.py`（+11/−2）

`adjust_first_round_interval`：
- 签名新增形参 `edge_threshold_scale: float = 1.0`（位于 `edge_noise_stop_mode` 之后）+ docstring 一行（<1 阈值更低、外推更远即框更宽；1.0=原行为）；
- 阈值计算改写：`edge_scale = float(edge_threshold_scale)`，非有限或 ≤0 回退 1.0；`y_threshold_left = float(baseline_left) × edge_scale`、`y_threshold_right = float(baseline_right) × edge_scale`（原为直接取 baseline 值）。

pipeline 链路不传该参数（默认 1.0，行为不变）；massnova 整谱模式默认 0.8。

#### 3.3 `model/inference/cli.py`（+34/−2）

- L1166-1171：新增 `--scan_edge_threshold_scale`（float，默认 0.8，help：边界截停阈值缩放系数，<1 阈值更低、外推更远即框更宽，1.0=原行为）与 `--scan_model_boundary_baseline_ratio`（float，默认 0.01，help：模型框内收审核的基线附近带宽，相对 apex-baseline 动态高度）；
- pipeline 模式（L1578-1606）：`_print_pipeline_timing_summary(...)` 返回值原来被丢弃，现接为 `timing_record` 并传入 `generate_for_pipeline(..., timing_record=timing_record)`——推理报告带上本次运行分模块耗时；
- roi 模式（L1622-1645）：`t_roi = time.perf_counter()` 包住 XIC 提取循环；结束后打印 `[推理完成]` 横幅（模式/样本数/总耗时/输出目录）+ `_print_pipeline_timing_summary(stage_seconds={"1_ROI生成(xic_extraction)": total_sec})`；
- roi2inference 模式（L1667-1685）：同款 `t_pred` 计时 + 完成横幅 + 汇总（阶段名 `1_模型预测(predictor)`；样本数按 `--batch_dir` 子目录计数）。

#### 3.4 `model/utils/predict_utils.py`（+174/−104）

- 新函数 `_build_result(img_path, probas, pred_boxes_item, size, threshold, return_all, qc_stats)`：单图输出→结果 dict 的公共整理（keep 掩码计算、qc_stats 逐图统计追加、boxes/scores 反归一化；无检测且非 return_all 返回 None）——**逐图路径与批处理路径共用，保证两条路径结果完全一致**；保留 DETR 取类关键注释（logits 布局 `[类别0, no-object]`，取 `[..., :1]` 类别 0，禁用 `[:-1]`/`[1:]`）；
- `predict(...)` 新形参 `batch_size=1`，docstring 说明（>1 时拼 batch 前向摊薄固定开销；批内尺寸不一致自动退回逐图，结果与 batch_size=1 完全一致）：
  - `bs<=1`：原逐图路径（行为不变），结果整理换用 `_build_result`，verbose 调试输出保留；
  - `bs>1`：按 chunk 预处理收集 `tensors/sizes`，尺寸集合唯一时 `torch.stack` 批前向，`probas_all = pred_logits.softmax(-1)[:, :, :1]`，逐图 `_build_result` 收尾；尺寸不一致时逐图退回；
- `build_predictor(...)` 新形参 `batch_size=1`，透传给 `predict`。

#### 3.5 `model/utils/xic_peak_utils.py`（+82/−28）

- 新函数 `_global_quiet_noise_pp(intensity_row, low_frac=0.4, pp_lo=5.0, pp_hi=95.0)`：全迹安静点（强度 ≤ 40 分位）峰-峰噪声 + 中位基线；点数不足或退化（pp≤0）返回 `(nan, nan)`；
- `compute_snr_outside_box(..., tail_reject_scale=3.0)`：
  - 入口统一 `np.asarray` 化；`low_ref = roi_full_low_decile_mean_intensity(intensity)`；
  - 每侧框外区（≥2 点）加守卫：`median(侧) > 3 × max(low_ref, 1e-9)`（仍处峰尾，常见于运行末端截断）→ 该侧不进 `all_noise`、不产 `noise_pp_*`；
  - 双侧均不可用：先全迹安静点兜底（`signal = peak_max − base_glob`，≤0 返回 nan），再退段内估计 `_compute_snr_peak_to_peak`；
  - baseline 缺失分支同样先试全迹兜底，失败才退段内；
- `compute_local_snr(..., tail_reject_scale=3.0)`：docstring 增补截断峰守卫说明；同款守卫作用于左右「安静区段」（邻居区段/回退扇区）；**双侧均不可用（截断/全占满）由返回 nan 改为全迹安静点兜底**（signal ≤0 才 nan）——避免 SNR nan 导致门控误杀。

#### 3.6 `model/tools/evaluation/inference_report.py`（+58/−5）

- 新函数 `load_timing_record(out_root)`：逐行读 `pipeline_timing_runs.jsonl`，返回最后一条记录（= 本次运行）；文件不存在/解析失败返回 None；
- 新函数 `render_timing_section(record)`：渲染「## 5. 运行用时（各模块）」——总耗时 + 记录时间 + `| 模块 | 耗时 (s) | 占比 |` 表（total≤0 时以 stage_seconds 求和为分母）；无 stage_seconds 时提示「pipeline 模式默认采集，`--no_timing` 只影响日志落盘」；
- `render_report(...)` 新形参 `timing_record=None`：QC 章节后插入上述章节；原第 5/6 章（管线各阶段说明 / 输出文件与关键列）顺延为第 6/7 章；
- `generate_for_pipeline(..., timing_record=None)`：调用方未显式提供时自动 `load_timing_record(out_root)` 兜底。

#### 3.7 `model/inference/massnova_runtime.py`（+1）

`DEFAULT_RUNTIME_CONFIG` 新增 `"scan_model_boundary_baseline_ratio": 0.01`——嵌入式 DLL 桥（与 `finalize_channel_peaks` 共用精修核心）的缺省参数对齐 CLI 默认。

#### 3.8 配置文件

| 文件 | 键 | 旧值 | 新值 | 说明 |
|---|---|---|---|---|
| `model/configs/inference_pipeline.json` | `pipeline_min_max_intensity` | 1000.0 | **3000.0** | QC：平滑后整条 XIC 最大强度低于此值的通道不生成 ROI、不参与预测 |
| `model/configs/massnova.json` | `batch_size` | 128 | **32** | 候选窗口批大小（注释更新：ONNX 与 torch .pth 路径共用） |
| `model/configs/massnova.json` | `threshold` | 0.5 | **0.6** | 模型检测框保留阈值 |
| `model/configs/massnova.json` | `scan_min_peak_ratio` | 0.04 | **0.05** | 峰高门 = baseline + r·dynamic |
| `model/configs/massnova.json` | `scan_min_snr` | 10.0 | **15.0** | 峰级本地 SNR 门（模型前置架构下仅作用于信号兜底路径） |

（`scan_edge_threshold_scale`/`scan_model_boundary_baseline_ratio` 未写入 json，走代码默认 0.8/0.01。）

#### 3.9 `CW/CW/Centwave` 子模块（内部未提交，+70/−1）

`CentreWave/centrewave/validation.py`：
- `compute_peak_metrics` 面积积分：`np.trapezoid` → `np.trapz`（numpy < 2.0 兼容，数值口径不变）；
- 新增 `estimate_peak_bounds(t, x, t_peak, sigma=None, n_sigma=2.0, min_side_points=5)`（72 行）：CentreWave 缺失的「峰起止时间」接口补全——峰区（t_peak±3σ，σ 未知仅剔除峰顶单点）外取中位基线 + `n_sigma × robust_std` 为阈值，从峰顶向两侧找首个强度回落点作 start/end；某侧未回落或窗口过窄时回退 `t_peak ± 3σ`；完全失败返回 `(t_peak, t_peak)`。

### 4. 行为影响

1. massnova 模型命中峰的边界从「模型框原值」变为「模型框种子 + 双向基线精修」——`massnova_peaks.csv` 的 `rt_min/rt_max` 及在其上重算的 `area/snr/n_points` 会变化；`boundary_source="model"` 语义改为「以模型框为种子的信号精修边界」；新增 `model_rt_min`/`model_rt_max` 列可追溯原始框；
2. `prediction_model/` 图画模型原始框（未精修），`prediction_refined/` 画最终精修结果；
3. 兜底路径因停阈值缩放（默认 0.8）与 apex 护栏，边界整体略宽、不再半峰截断；
4. 截断峰不再因 SNR nan 被门控误杀；
5. pipeline 链路 `adjust_first_round_interval` 默认参数不变（scale=1.0），行为完全兼容（单测钉死）；
6. 推理报告（pipeline 与 massnova）均新增「运行用时（各模块）」章节；roi/roi2inference 模式补完成横幅与计时汇总；
7. `output/inference/massnova_*/` 下 `massnova_peaks.csv` 新增两列，读方（如 all.csv 汇总）向后兼容不受影响。

### 5. 新增单测（4 文件 / 19 用例）

| 文件（行数） | 覆盖点 |
|---|---|
| `model/tests/test_edge_threshold_scale.py`（69） | 不传新参数 = 显式 1.0（pipeline 兼容钉子）；scale<1 右边界更外；0.5 比 0.8 更外（单调性）；0/负/NaN 非法值回退 1.0；`_boundary_stop_levels` 线性缩放、非法值保持原水平 |
| `model/tests/test_massnova_model_boundary_refinement.py`（45） | 框内边界高于基线 → 外扩到稳定基线（±0.011 min 容差）；框罩多余基线 → 内收；一侧扩一侧收可同时发生；已贴基线 → 稳定不变 |
| `model/tests/test_massnova_stop_level_apex_guard.py`（92） | 复现 test2 chrom052 甲羧除草醚-2 双峰形态（低峰 + 1 min 外更高邻居）：污染侧停阈值被压到 apex 显著比例以下；右边界越过 apex 一段距离而非 apex 后一步截停；孤立峰护栏零影响；输出 peak 保留 `model_rt_*` 且 ≠ 精修后最终边界 |
| `model/tests/test_truncated_noise_reference.py`（58） | 运行末端截断峰只用健康侧（修复后 SNR>100，旧逻辑 `tail_reject_scale=1e9` 模拟 <30）；双侧截断回退全迹安静点不返 nan；内部峰保持有限 SNR；框占满全迹用全局兜底；陡峭右尾侧被剔除 |

### 6. 验证命令与结果（2026-09-23 实测）

```powershell
cd D:\yinlibo\MRMPFormer\model
D:\miniconda\envs\gamstekpeaking\python.exe -m unittest `
  tests.test_edge_threshold_scale tests.test_massnova_model_boundary_refinement `
  tests.test_massnova_stop_level_apex_guard tests.test_truncated_noise_reference -v
```

结果：**Ran 19 tests in 0.011s — OK（19/19 通过）**。

### 7. 新增工具（4 文件 / 1249 行，未纳入单测，属评测/诊断脚本）

| 文件（行数） | 用途 |
|---|---|
| `model/tools/evaluation/run_massnova_sim.py`（182） | 在 MRM-XIC 模拟集（easy/medium/hard，仅 `xic_data/*.json` 无 mzML）上运行 massnova：临时替换 Phase0 为 `_json_extract_full_xics`（uid=`compound_id\|ion_type` 对齐 label.csv 主键），其余相位（候选枚举→模型前置→兜底精修→门控→输出报告）完全复用 `inference.massnova`，与真实 mzML 同口径；`--max_channels` 冒烟、`--limit` 限量 |
| `model/tools/evaluation/evaluate_massnova_sim.py`（469） | 对 all.csv 做精度评测：真值 = `label/label.csv` + `generation_manifest.csv`（难度/噪声子类型）；同通道内 apex RT 最近贪心一对一匹配（`--tol` min）；产出 report.md、metrics_*.csv（Precision/Recall/F1、RT 偏差、边界 IoU、面积 log-log 相关、噪声通道误报等）、matched/fp/fn 明细、figures/*.png（中文字体适配） |
| `model/tools/evaluation/compare_massnova_sim.py`（222） | 汇总三档（test_easy/medium/hard）`metrics_headline.csv`：核心指标跨数据集对比表 + 可视化（列名中英映射） |
| `model/tools/diagnostics/trace_channel_queries.py`（376） | 单通道 case 诊断：复现 Phase3b 同一输入（同款 transform/切窗），逐 decoder 层（L1/L2/L3）分类分（共享 class_embed 逐层各自 softmax）与 FDR 精化边界（base/L1/L2/L3）、逐 query 分数/RT 区间/框宽/中心偏移、复算 massnova 贪心分配（判断 query2/query3 是否产出重复/虚假框），并与部署 ONNX 同图 L3 分数/框做一致性校验 |

### 8. 产物与位置

| 产物 | 路径 |
|---|---|
| 全部改动 | 已提交（feat(model)，大文件经 Git LFS）；基线 `9fdef78`，`git diff 9fdef78 HEAD` 复现 |
| 本报告 | `docs/experiment_report.md`（本节，实验日志 004） |
| 模拟集评测输出（按工具约定） | `output/inference/massnova_<dataset>/`（run_massnova_sim.py 产出，评测明细在其 `report/` 子目录） |

---

## 实验日志 002：test1 标准试卷、四模型横评与 mrmpformerv2 的诞生（多源联合训练）

- **日期**：2026-08-22
- **涉及模型**：quanformer.pth（基线）、quanformerv2.pth（v2）、quanformerv3.pth（v3）、mrmpformerv1.pth（v1，traindata3 训练）、mrmpformerv2.pth（v2，多源联合训练，本次新增）
- **状态**：全部完成；四模型权重已入库 `model/checkpoint/`；联合评估报告见 `docs/joint_evaluation_4models.md`

### 1. 背景与动机

此前（日志 001 与后续评测）反复出现"同一模型换数据集排名反转"的现象，各模型能力缺乏统一、公平的度量。本次引入 **test1 标准试卷**（`data/mzml/test1/`，11 mzML：8 农残混标 + 3 空白，96 个 GT 峰，`data/label/test1.xlsx`）——该数据此前从未参与任何模型训练，作为外域基准；同时把多源联合训练作为根治跨域不稳定的主要手段。

### 2. 过程与关键事件

#### 2.1 test1 数据准备与评估链路修复

- mzML 按内部 `<sample name>` 重命名（81~91 保留：空白1/-2/-3 + 农残混标/-2~-8，其余 80 个删除）；
- 评估链路两处修复（不修则结果全错）：
  - `evaluate_baseline._parse_gt_peaks`：跳过 `peak_label=0`（空白负样本），避免 `"0"` 被解析成 RT=0 的伪 GT 峰；
  - `evaluate_baseline` GT 匹配：兼容 compound 列已带通道后缀的标注（test1 的「6-涕灭威-1」），否则 `label_key` 二次拼接后缀导致 GT 全空；
  - `visualize_compare`：多峰格式（peak_start1-3）解析修复，修复前 GT 全 None（TP/FN/FP 全 0 假象）。

#### 2.2 四模型在 test1 标准试卷的首轮横评（score≥0.5，±0.1 min）

| 模型 | TP/FP/FN | F1 | 空白样 FP |
|---|---|---|---|
| baseline | 1/111/95 | 0.010 | 17 |
| v2（shiyaoyuan 微调） | 0/0/96（零检出，max score 0.01~0.06） | 0.000 | 0 |
| v3（merged 微调） | 4/105/92 | 0.039 | 14 |
| mrmpformerv1 | 86/4/10 | **0.925** | 2 |

放宽容差到 ±0.3 后 baseline/v3 跳升至 0.914/0.927——证明它们"检得到峰"但边界系统性偏差 ~0.11 min（恰好卡在 ±0.1 之外）；±0.1 口径的 F1 实质度量**边界约定匹配度**。

#### 2.3 对照实验：test1 微调基线能否追平 v1？

用 test1 的 7 个样品微调基线（留出 4 个：农残混标-7/8 + 空白1-2/3，24 峰），留出集对比：

| 模型 | F1@0.1 | RT 起/止 | 空白 FP |
|---|---|---|---|
| 微调基线 | 0.836 | 0.009/0.019 | 7 |
| mrmpformerv1 | **0.894** | 0.016/0.032 | 1 |

**结论**：边界约定可被数据学到（偏差 0.11→0.009），但空白样臆造（7 FP）未被解决；v1 仍领先 0.06 F1——架构优势（空白抑制）真实存在，非纯约定巧合。

#### 2.4 架构审查发现与优化执行（P0-1/P0-2/P0-4/P2-8）

架构审查（含训练日志诊断）发现：FDR 三层精化 IoU 增益仅 +0.005、FDR 损失有效权重 ≈8.8 过度主导、右边界 MAE 系统性偏大 23%、以及一个真实 bug——**`build_matcher` 未透传 `iou_type`，匈牙利匹配恒用 GIoU，与 PW-CIoU 损失口径错位**。已执行：

| 项 | 内容 | 状态 |
|---|---|---|
| P0-1 | matcher 透传 `iou_type=ciou`（`models/shared/matcher.py`，22 个单测通过） | ✅ 已提交 git（a14c438） |
| P0-2 | `fdr_loss_coef` 2.0→1.0（有效 8.8→4.4） | ✅ 载于 `configs/mrmpformer_v1_multisrc.json` |
| P0-4 | 阈值校准扫描：0.4~0.7 平台 F1≥0.92，推荐 0.5 | ✅ |
| P2-8 | 多源数据集 **multisrc**：traindata3×98 + shiyaoyuan_1 + traindata2×5 + test1×7 = 17796 图/13929 框；留出 QC/QC5 + shiyaoyuan_2 + 农残混标-7/8 + 空白1-2/3 作 val | ✅ |

**mrmpformerv2** = v1 热启动 + 上述全部改动，AMP 混合精度训练 30 epochs（61 分钟）。训练加速三项（--amp/--tf32/--cudnn_benchmark，engine.py + train.py）亦在本次加入。

### 3. 最终四模型联合评估（holdout4 公平口径，24 峰，四模型均未训练）

| 模型 | F1@0.1 | F1@0.3 | RT 起/止@0.1 | R²@0.3 | RSD | 空白 FP |
|---|---|---|---|---|---|---|
| baseline | 0.000 | 0.800 | — | 0.98004 | 1.27% | 12 |
| quanformerv3 | 0.035 | 0.842 | 0.019/0.085 | 0.98108 | 1.09% | 9 |
| mrmpformerv1 | 0.894 | 0.936 | 0.016/0.032 | 0.98968 | 1.31% | 1 |
| **mrmpformerv2** | **0.913** | **0.957** | **0.007/0.012** | **0.99052** | 1.33% | **0** |

跨域（shiyaoyuan）稳定性：v2 F1@0.1 0.442 > v1 0.380，RT 偏差（0.016/0.035）反超两个域内模型；±0.5 下四模型 0.98~1.00。定量 R² 全员 ≥0.98，瓶颈在检测配对而非积分。

### 4. 结论

1. **mrmpformerv2 综合最优**：边界偏差较 v1 减半（止边界 -63%）、空白样零臆造、跨域边界一致性最佳、置信度平台宽（0.4~0.7 稳定）；
2. v1→v2 增量归因：matcher 修复 + 损失再平衡 → 边界精度；多源空白负样本 → 假阳性抑制；召回未变（剩余 3 个 FN 为同一批难例，需 P1 级架构改动）；
3. **±0.1 严格 F1 ≈ 边界约定匹配度**：跨数据集排名反转（test1 上 v 系碾压 / shiyaoyuan 上 baseline 主场 0.59）是约定差异所致；根治靠多源联合训练，架构优化不能替代数据多样性；
4. 空白负样本的数量与多样性是假阳性抑制的第一要素（baseline 每张空白图臆造 6 峰，会直接导致质控违规）。

### 5. 产物与位置

| 产物 | 路径 |
|---|---|
| 四模型权重 | `model/checkpoint/{quanformer,quanformerv2,quanformerv3,mrmpformerv1,mrmpformerv2}.pth` |
| 多源数据集 | `data/coco/multisrc/`（train 17796 图 / val 429 图） |
| 训练配置 | `model/configs/mrmpformer_v1_multisrc.json`（含 P0-1/P0-2 统一设定） |
| 联合评估原始数据 | `output/evaluation/joint4_results.json`、`test1_*`、`holdout4_*`、`shiyaoyuan_*` |
| 四模型完整评估报告 | `docs/joint_evaluation_4models.md` |
| 代码修复 | `models/shared/matcher.py`（已提交）、`tools/evaluation/evaluate_baseline.py`、`tools/evaluation/visualize_compare.py`、`train.py`+`framework/engine.py`（AMP，未提交） |

---

## 实验日志 003：quanformer「基线」真实出身考据 —— 外部下载权重与 v2 域坍缩根因（仅分析）

- **日期**：2026-08-22
- **涉及模型**：quanformer.pth（下载权重）、quanformerv2.pth（在其上单样品微调）
- **状态**：分析完成；无代码/训练改动（应用户要求仅分析）
- **更正声明**：日志 001/002 及 `docs/joint_evaluation_4models.md` 中将 quanformer.pth 称为"基线（本项目从零训练）"不准确——其为本节考据的外部下载权重，相关表述以本节为准。

### 1. 关键证据：checkpoint 内嵌的原始训练参数

读取 `checkpoint/quanformer.pth` 的 `args` 字段（外部训练遗留，铁证）：

```
resume='D:\workspace\autopeakV3\detr-r50-e632da11.pth'   ← DETR 官方 COCO 预训练起点
coco_path='D:\workspace\train-dataset\peak-all'          ← 外部色谱峰数据集 peak-all
output_dir='D:\workspace\autopeakV3\output\peakdetr\peak-ciou-all113-res'  ← 外部项目 autopeakV3（"all113"≈113 样品）
epochs=50（保存于 epoch 29），lr=1e-4
```

真实训练链：**DETR COCO 预训练 →（外部 autopeakV3 项目）peak-all 数据集微调 30+ epochs → 下载为本项目"quanformer 基线"**。`configs/quanformer_baseline.json` 从未实际训练过（`coco_path: data/test/coco` 为文档遗留）。架构参数（resnet50/1+1 层/3 query）与本项目 QuanFormer 恰好一致，故可直接加载。

### 2. baseline 表现差的根因（test1 F1@0.1=0.010，空白臆造 17 峰）

**"数据饥饿从零训练"论不成立**（推翻 002 前的旧分析）——基线见过大规模外部峰数据。真实原因是**外来约定与本项目评估协议的系统性错位**：

1. **框约定错位（主因）**：peak-all 的标注约定为紧贴 apex 的窄框（~0.23 min），与 shiyaoyuan/test1 人工宽积分边界（~0.47 min）系统性差 ~0.11 min——恰好卡在 ±0.1 之外、±0.3 之内。这解释了 F1@0.1=0.010 → F1@0.3=0.914 的跳变：**不是检不到，是口径不对**（在 shiyaoyuan 域内同样只有 0.59@0.1，宽容差 0.95~0.98，四模型中除 v3 外最好）。
2. **空白盲区**：peak-all 为"peak"数据集，大概率不含空白进样负样本。模型带着"每图必有峰"先验，在 test1 的 3 张空白图上照常 0.99 触发（36 通道臆造 17 峰，质控场景致命）。
3. **域外高置信触发（0.967~0.999）恰是成熟检测器的正常泛化**：检测头对"apex 状亮斑"概念稳定，跨域照常工作——模型本体能力在线。

### 3. v2 零检出的根因（test1 全 132 图无一过 0.5，max score 0.01~0.06）

v2 = 上述外部权重 + shiyaoyuan test_1 **单样品 61 图**微调（lr 1e-5 × 10 epochs）。双重病理：

1. **灾难性遗忘**：在成熟宽域检测器上做窄域微调，分类头决策边界从"peak-all 宽峰形态"被拉向"shiyaoyuan 宽积分形态"——旧泛化能力被部分覆写。基线域外 0.99 触发 → v2 域外 0.06 以下：微调磨掉的不只是"不认识 test1"，连 peak-all 赋予的宽泛触发也一并丢失。
2. **窄域概念重定义**：正样本从 apex 斑块换成人工宽积分边界，概念精确化必然绑定训练域（域内 F1 0.008→0.455 的代价）。
3. **对照封死归因**：v3 = 同一基线 + 同样 10 epochs，唯一区别是数据多样性（merged 多样品 + 负样本）→ test1 正常触发 105 框。**"窄数据微调宽模型"= 遗忘快于学习**；v2 的崩溃不是微调原罪，而是数据多样性低于基线原有知识广度时的必然结果。

### 4. 对项目的启示

1. `quanformer.pth` 不宜再当作"本项目基线"参与模型对比——它是"外部约定的探测器"，公平的 quanformer 对照应是用 multisrc 多源数据微调该权重（即 `quanformer_test1_ft` 实验思路，F1 已达 0.836/0.935）；
2. 任何下载/外部权重的第一步：读取 `checkpoint['args']` 考据出身，再决定其在实验矩阵中的定位；
3. 窄域微调成熟模型的正确姿势：小学习率 + 少 epoch + 保留多样本（或直接混入通用数据回放），否则等于用 61 张图覆写上万图的知识。

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
| 2026-08-22 | test1 标准试卷导入（11 mzML 重命名）；四模型首轮横评 + 评估链路三处修复；test1 微调基线对照实验；架构审查定位 matcher iou_type bug；执行 P0-1/P0-2/P0-4/P2-8 → **mrmpformerv2**（multisrc 多源联合训练）；四模型联合评估完成，v2 holdout F1@0.1=0.913 全面最优（详见实验日志 002） |
| 2026-08-22 | 考据 `checkpoint/quanformer.pth` 内嵌 args：基线实为外部下载权重（DETR COCO → autopeakV3/peak-all 微调），非本项目自训；v2 零检出根因确认为窄域微调引发的灾难性遗忘 + 域坍缩（详见实验日志 003） |
| 2026-09-23 | massnova 模型框降级为精修种子（外推+内收双向基线校正，P4 推理端对策）；停阈值 apex 护栏修复甲羧除草醚-2 半峰截断；截断峰 SNR 守卫 + 全迹安静点兜底；torch(.pth) 批量前向；pipeline/roi/roi2inference/massnova 全模式分模块计时进终端与报告；模拟集三档（easy/medium/hard）评测工具链与逐层逐 query 诊断工具；19 个新增单测全部通过（详见实验日志 004） |

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
| `roi2inference` | 已有 ROI 目录批量预测 | `--batch_dir`（每子目录一套 ROI） | `<out>/<子目录>/prediction.csv` |
| `pipeline`（默认） | 完整管线①~④（单文件或批量） | 同 `roi` | `base_out/` 四阶段产物 |

> `<key>` = mzML 文件名 stem；目录递归下不同子目录同名 stem 自动改为路径展平（如 `子目录A__样品1`）避免覆盖。

### 各模式详细输出

**① `roi`**：`extract_xic_with_pyopenms` 逐 mzML 读色谱 → `<out>/<key>/`（仅 ROI 生成，无需 `--model`，不支持 `--plot`）：
- 每通道 ROI jpeg（命名 `N_mz{母离子}_q3{子离子}.jpeg`）
- `feature.csv`、`roi_windows.csv`、`xic_matrix.npy`
- 被 QC 剔除的通道记录 `pipeline_qc_excluded.csv`
- 想看预测框标注：`roi` 生成 ROI 后，再对该目录跑 `roi2inference --plot`（模型仅加载一次）

**② `roi2inference`**：复用 `predictor.main()` 批量模式，逐子目录：
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

---

## 附录 I：训练期 COCO 评估输出解读（2026-08-22）

> 每次 epoch 结束 `evaluate()` 会打印 12 行标准 COCO 检测指标（pycocotools，`framework/datasets/coco_eval.py`）。本文档说明每一行是什么、为什么设计成这 12 行、以及本项目应如何解读。

### I.1 一次真实输出（epoch 末 bbox 评测）

```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.262
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.537
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.226
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.124
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.382
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.407
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.604
```

### I.2 12 行指标的构成逻辑

12 行 = **AP 6 行 + AR 6 行**，由 3 个维度组合：

| 维度 | 取值 | 含义 |
|---|---|---|
| IoU 阈值 | 0.50 / 0.75 / **0.50:0.95** | 判定"框算对"的严格程度；0.50:0.95 为 0.05 步长 10 档的平均 |
| 目标面积 | all / small / medium / large | 按 GT 框面积分档（small ≤ 1024 px²，medium ≤ 9216 px²，large > 9216 px²） |
| 检测预算 | maxDets = 1 / 10 / 100 | 每张图最多保留的候选框数（仅 AR 有） |

### I.3 逐行解读（结合真实数值）

| 行 | 数值 | 解读 |
|---|---|---|
| AP @0.50:0.95 all | 0.262 | **COCO 官方主指标**，10 个 IoU 阈值上的平均精度，唯一横向可比数字 |
| AP @0.50 all | 0.537 | 宽松 IoU（PASCAL VOC 口径），"框大概对上"即算对 |
| AP @0.75 all | 0.226 | 严格 IoU，**对边界贴合极度敏感** |
| AP @0.50:0.95 small | -1.000 | 无 small 档 GT → 无法计算（见 I.5） |
| AP @0.50:0.95 medium | 0.124 | 中等面积目标精度明显低于 large → 中等宽度峰边界更难学准 |
| AP @0.50:0.95 large | 0.274 | 大目标精度最高 |
| AR @maxDets=1 | 0.382 | 每图只允许 1 个框时的召回——**贴近本项目部署口径**（每通道最终只保留面积最大框） |
| AR @maxDets=10 | 0.589 | 放宽到 10 个框的召回上限 |
| AR @maxDets=100 | 0.589 | 与 @10 相同 → 每图候选数有硬上限，加预算无收益 |
| AR small / medium / large | -1 / 0.407 / 0.604 | 同面积分档的召回 |

### I.4 为什么需要这些指标（设计动机）

1. **IoU 分档**：单一 IoU 阈值只测一种严格程度，容易被标注噪声带偏。0.50:0.95 平均兼顾"有没有检测到"与"框得准不准"，跨数据集可比；0.5 保留与 PASCAL 时代结果的衔接；0.75 单独暴露定位质量。
   - 本项目特殊性：bbox 就是人工积分边界（`peak_start/peak_end`），**AP50 与 AP75 的差距 ≈ 边界贴合度 ≈ 定量口径的 RT 起止偏差来源**。
2. **面积分档**：防止"只检测到大目标"被平均掩盖。小/中/大三档阈值来自 COCO 原始设计；分档数值可指导按峰宽调参（如 medium 更差 → 窄峰/中等宽度峰需要更强的边界精化）。
3. **AR + maxDets 分档**：AP 衡量的是"按分数排序后的最终输出"精度；AR 衡量**不考虑精度时的召回上限**（模型到底能找回多少 GT）。maxDets 回答"给多少候选预算才能吃满召回"。
4. **-1 的约定**：pycocotools 对"该格无 GT 样本"返回 -1 表示**未定义**（不是 0——0 意味着"一个没找对"），读表时应跳过。

### I.5 本项目特别注意点

- **small 恒为 -1**：已实测 val 标注 1410 个 GT 框中 small 档为 0（全部为 medium 112 + large 1298），属常态而非异常；
- **AR@10 = AR@100**：`num_queries=3` 使每张 ROI 图最多输出 3 个框，检测预算上限不生效；
- **AR@1 是部署口径**：推理端每 (mz,q3) 通道最终只保留面积最大的一行，AR@1 与生产指标的相关性高于 AP；
- **AP75 低 ≠ 检测失败**：若定量（面积 R²、RT 偏差）优秀而 AP75 低，说明框宽约定与人工边界存在系统性偏差（参见实验日志 001 的教训：口径不同结果天差地别）。
