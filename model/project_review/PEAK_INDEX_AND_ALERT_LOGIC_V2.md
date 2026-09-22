# 峰索引、置信度与预警逻辑 V2

## 1. 文档范围

本文记录当前 `MRMPFormer + MassNova 信号算法` 推理链路中峰事件的生成、编号、AI 与 MassNova 软件传统峰的对齐、置信度计算和人工复核预警规则。

- 推理主流程：`inference/massnova.py`
- 峰自身得分：`utils/peak_scores.py`
- AI/传统峰对齐与预警：`postprocessing/comparison_confidence_v2.py`
- 推理配置：`configs/massnova.json`
- 置信度配置：`configs/comparison_confidence_v2.json`
- 输出版本：`comparison_confidence_v2_1`

本次没有修改 `D:\yinlibo\MRMPFormer\cpp` 中的软件接口代码。当前先稳定 Python 侧字段和规则，软件接口应在字段定稿后单独适配。

## 2. 必须区分的四类量

| 字段 | 适用峰 | 含义 | 是否为概率 |
|---|---|---|---|
| `score_ai` / `model_score` | 模型边界峰 | MRMPFormer 检测框原始得分 | 模型输出分数，尚未做概率校准 |
| `score_signal` / `signal_score` | 信号边界峰 | 根据最终峰区间的 SNR 与有效点数计算的规则分 | 否 |
| `score_peak` / `peak_score` | 每个保留的 AI 峰 | 按峰来源选取 `score_ai` 或 `score_signal` 的统一峰自身分 | 否；用于统一接口 |
| `comparison_confidence` | 正式匹配峰；有最近传统参考的额外 AI 峰 | 峰自身分、RT 差、面积差、匹配质量及匹配状态组成的复合置信度 | 否；当前为待标定规则分 |
| `final_confidence` | 所有可比较输出事件 | 正式匹配峰取正式比较分；额外 AI 峰取最近参考比较分；传统漏检取 0 | 否 |
| `review_status` | 所有输出事件 | 人工复核工作流状态 | 不参与数值计算 |

`score_source` 明确记录统一峰分来自哪里：

- `model`：`score_peak = score_ai`
- `signal_rule`：`score_peak = score_signal`
- `missing`：两种分数都无法得到，属于异常数据

这几个字段不能混叫“置信度”。推理图显示峰自身分 `score_peak`；AI 与传统算法校验表中的 `comparison_confidence` 包含 RT 和面积信息。`comparison_reference_type` 用来区分正式一对一匹配和最近传统诊断参考。

## 3. 推理阶段的峰生成与保留

### 3.1 通道内处理顺序

每个 mzML 色谱通道独立执行以下步骤：

1. Phase 1 用当前生产候选器 `SciPy find_peaks + prominence` 枚举候选峰顶。
2. 每个候选峰顶截取约 ±1 min 的窗口，交给 MRMPFormer 推理。
3. 模型框包含候选 apex 且 `score_ai >= threshold` 时，该峰采用模型边界，`boundary_source=model`。
4. 没有被模型命中的候选进入信号兜底：边界精修、SNR、有效点数和面积门控通过后保留，`boundary_source=signal`。
5. 模型峰和信号峰合并后执行重叠去重。
6. 在最终边界上统一重算面积、SNR、有效点数和峰自身分。
7. 所有最终峰写入 CSV，并进入整通道图和逐峰窗口图。

“所有预测峰都参与置信度”指第 5 步之后的全部最终峰。信号门控之前被判定为噪声的原始候选不属于最终预测峰，不会进入置信度表。

### 3.2 模型阈值

当前 `configs/massnova.json` 中模型阈值由旧生产试验值 `0.8` 下调为暂定值 `0.6`。作用是让更多中等模型分数的候选保留模型边界并进入复核链路。

阈值只决定候选是否采用模型边界。低于阈值的候选仍可经信号兜底成为最终峰，所以不会仅因模型分数低于 0.6 就自动消失。

`0.6` 目前是工程试验值，需要用有人工结论的数据绘制可靠性曲线，并结合漏报率、误报率和复核工作量重新标定。

### 3.3 信号峰得分

信号峰没有 MRMPFormer 原始分数，因此新增透明、可复算的规则分：

```text
conf_snr    = SNR / (SNR + 10)
conf_points = clip(n_points / 10, 0, 1)

score_signal = conf_snr^0.8 × conf_points^0.2
```

设计理由：

- SNR 是主要证据，权重为 0.8；用饱和函数避免超高 SNR 无限放大。
- 有效峰点数是辅助证据，权重为 0.2；点数不足会降低分数。
- 几何平均要求两个证据都不能太差，同时不会把规则分伪装成模型概率。
- 该分数只描述这个信号峰自身是否清晰，不使用传统峰 RT 或面积。

默认参数下的示例：

| SNR | 有效点数 | `score_signal` |
|---:|---:|---:|
| 10 | 5 | 0.5000 |
| 10 | 10 | 0.5743 |
| 20 | 10 | 0.7230 |
| 40 | 10 | 0.8365 |

CSV 同时保留 `signal_score_snr_component` 和 `signal_score_points_component`，以后修改公式时可以回放和审计。

## 4. 峰索引逻辑

### 4.1 推理峰索引

一个峰当前由以下层级定位：

```text
mzml_stem（样本）
  └─ chrom_index + uid（MRM transition/通道）
       └─ peak_no（该通道内按 rt_min 升序排列，从 1 开始）
```

- `mzml_stem`：mzML 文件名去掉扩展名，例如 `test3_1`。
- `chrom_index`：该 mzML 中提取出来的色谱通道序号。
- `uid`：通道/化合物标识，例如 `flonicamid-1`。
- `peak_no`：最终去重后，在同一个样本、同一个 `chrom_index` 内按 `rt_min` 从早到晚重新编号。
- `ai_row_id`：置信度程序读取 `all.csv` 后，按文件行顺序添加的全表临时行号。

重叠去重判据为：两个相邻候选区间重叠，并且 apex 的 RT 距离不超过 `scan_dup_apex_tol=0.2 min`。同一重复组中优先保留模型边界峰；同来源时保留峰顶强度更高的峰。

`peak_no` 和 `ai_row_id` 都不是跨版本永久主键。改变阈值、候选器、边界或去重参数后，前面新增或删除一个峰，后续编号就可能改变。当前回查一个峰至少应保存：

```text
(mzml_stem, chrom_index, uid, peak_no, rt_peak, schema_version)
```

软件长期落库时建议新增不可变 `peak_event_id`，由一次推理运行 ID、样本、通道、RT 和峰序号生成；不要只用 `peak_no` 关联历史人工复核记录。

### 4.2 图片索引

- 整通道图：`model_plots/<sample>/chromNNN_<uid>_model.png`
- 信号峰逐峰图：`prediction_signal/<sample>/chromNNN_peakNNN_<uid>.png`
- 模型峰逐峰图：`prediction_model/<sample>/chromNNN_peakNNN_<uid>.png`
- 全部最终峰逐峰图：`prediction_refined/<sample>/chromNNN_peakNNN_<uid>.png`

`prediction_refined` 不按来源过滤，因此每个最终模型峰和信号峰都作为 seed 生成一张图。窗口中若还有相邻峰，也会一起标注。图例和左上信息框显示统一 `peak_score` 及 `score_source`，信号峰不再显示 `--`。

### 4.3 传统峰索引

`trad_row_id` 是读取传统算法 Excel 后按原表行序添加的临时行号。传统峰与 AI 峰匹配前不会用 `peak_no` 直接对号，因为两边峰数量和排序可能不同。

## 5. AI 峰与传统峰的对齐

### 5.1 分组边界

只在完全相同的 `(sample_id, uid)` 分组内匹配。不同样本或不同 transition 永远不会互相匹配。

### 5.2 候选配对门

一对峰满足下列任意条件才可进入匹配矩阵：

```text
|AI apex RT - 传统 apex RT| <= 0.5 min
或
AI 区间与传统区间存在重叠
```

### 5.3 配对代价

```text
RT项     = min(|ΔRT| / 0.5, 2)
边界项   = 1 - interval_IOU
峰分项   = 1 - score_peak

match_cost = 0.70×RT项 + 0.25×边界项 + 0.05×峰分项
```

峰自身分只占 5%，只用于 RT 与边界相近时的轻量择优，不能让一个高分但 RT 很远的峰抢走配对。

每个 `(sample_id, uid)` 分组使用 Hungarian 算法求全局一对一最小总代价。一个 AI 峰最多匹配一个传统峰，一个传统峰也最多匹配一个 AI 峰。

若某传统峰的最佳与次佳候选代价差小于 `0.10`，状态为 `MATCHED_AMBIGUOUS`，直接要求人工复核。其余成功配对为 `MATCHED_UNIQUE`。

一对一分配完成后，每个 `AI_ONLY_EXTRA` 在同一 `(sample_id, uid)` 内选择 apex RT 最近的传统峰作为诊断参考，并计算：

```text
delta_rt_abs = |AI峰RT - 最近传统峰RT|
area_diff_abs = |AI峰面积 - 最近传统峰面积|
area_diff_pct = area_diff_abs / ((AI面积 + 最近传统面积)/2) × 100%
```

最近参考不会改变 `AI_ONLY_EXTRA` 状态，多个额外峰可以引用同一传统峰进行诊断，但不会被登记为多个正式匹配峰。

对齐后保留全部输入事件：

| `match_status` | 含义 |
|---|---|
| `MATCHED_UNIQUE` | 唯一匹配 |
| `MATCHED_AMBIGUOUS` | 匹配存在歧义 |
| `AI_ONLY_EXTRA` | 同样本同通道中未匹配的 AI 峰 |
| `NO_TRAD_SAMPLE` | 该 AI 样本在传统结果中完全不存在 |
| `TRAD_ONLY_AI_MISS` | 传统峰没有匹配到任何 AI 峰 |
| `REFERENCE_MISSING` | 传统行存在，但关键 RT 缺失，无法比较 |

## 6. 面积差与面积置信度

按新定义计算对称面积差百分比：

```text
area_diff_abs = |area_ai - area_trad|
area_mean     = (area_ai + area_trad) / 2
area_diff_pct = area_diff_abs / area_mean × 100%
```

输出中 `area_diff_pct` 与 `conf_area_diff_pct` 都保存这个原始百分比。两者当前值相同，后者用于明确它是面积置信度模块的输入。

因为复合置信度的所有 `conf_*` 分量都采用“越大越可信”的 0～1 方向，原始百分比再转换为：

```text
conf_area = clip(1 - area_diff_pct / 200, 0, 1)
```

对非负面积而言，对称差最大为 200%。示例：

| AI 面积 / 传统面积 | `area_diff_pct` | `conf_area` |
|---:|---:|---:|
| 1.0 | 0% | 1.0000 |
| 1.5 | 40% | 0.8000 |
| 2.0 | 66.67% | 0.6667 |
| 4.0 | 120% | 0.4000 |
| 一边为 0、另一边大于 0 | 200% | 0.0000 |

两个面积都为 0 时定义差异为 0%。面积缺失或为负数时 `conf_area=0` 并记录 `AREA_MISSING`。

## 7. 匹配峰的复合置信度

只有成功对齐的峰对才计算完整的比较置信度：

```text
conf_rt    = 2^(-|ΔRT| / 0.1)
conf_match = exp(-match_cost)

comparison_confidence = conf_match
                      × conf_peak^0.35
                      × conf_rt^0.40
                      × conf_area^0.25
```

`AI_ONLY_EXTRA` 使用最近传统峰计算相同分量，再乘未匹配状态系数：

```text
非竞争性额外峰：conf_status = 0.75
竞争性额外峰：  conf_status = 0.50

comparison_confidence_extra
= conf_status × conf_match_nearest
× conf_peak^0.35 × conf_rt_nearest^0.40 × conf_area_nearest^0.25
```

- `conf_peak` 是统一峰自身分：模型峰取 `score_ai`，信号峰取 `score_signal`。
- `conf_rt` 在 RT 差为 0.1 min 时减半。
- `conf_area` 使用第 6 节的新面积公式。
- 几何加权会让任何一个明显差的证据拉低结果。
- `final_rule_risk_score = 100 × (1 - final_confidence)`，数值越大风险越高。

`confidence_scope` 说明 `final_confidence` 的可解释范围：

| 范围 | `final_confidence` 的取值 |
|---|---|
| `PAIRED_COMPARISON` | 完整 `comparison_confidence` |
| `NEAREST_REFERENCE_COMPARISON` | 额外 AI 峰与最近传统峰的诊断比较分 |
| `NO_AI_PEAK` | 0 |
| `UNAVAILABLE` | 空值 |

## 8. 预警规则与优先级

峰级 `alert_level` 按以下优先级判定：

1. `TRAD_ONLY_AI_MISS`：`RED`。
2. `REFERENCE_MISSING` 或 `NO_TRAD_SAMPLE`：`UNAVAILABLE`。
3. `AI_ONLY_EXTRA`：竞争性额外峰为 `RED`；非竞争性额外峰按最近参考比较分判定，`>=0.60` 为 `YELLOW`，否则为 `RED`。额外峰不会自动放行为绿色。
4. 匹配歧义或统一峰分缺失：`RED`。
5. `|ΔRT| >= 0.30 min`：`RED`，原因 `RT_HARD_LIMIT`。
6. `area_diff_pct >= 120%`：`RED`，原因 `AREA_HARD_LIMIT`。该值等价于一边面积约为另一边 4 倍。
7. 传统峰自身质量为红色：当前先降为 `YELLOW`，避免把低质量传统结果直接当成 AI 错误。
8. 其余匹配峰按复合置信度：
   - `>= 0.80`：`GREEN`
   - `0.60～0.80`：`YELLOW`
   - `< 0.60`：`RED`

低于 0.8 与低于 0.6 的匹配峰分别记录 `COMPOSITE_BELOW_GREEN` 和 `COMPOSITE_BELOW_YELLOW`，方便解释单纯由复合分阈值产生的预警。

当前人工复核开关为：

```text
review_required = alert_level in {YELLOW, RED, UNAVAILABLE}
```

- `INFO`：保留并展示，默认不强制复核。
- `GREEN`：默认不复核。
- `YELLOW`：需要复核。
- `RED`：需要优先复核。
- `UNAVAILABLE`：缺少完整比较条件，需要补数据或人工判断。

`review_status` 初始为 `PENDING` 或 `NOT_REQUIRED`；`review_decision`、`reviewer`、`reviewed_at` 和 `review_comment` 留给后续人工工作流填写，算法不会自动冒充人工结论。

## 9. 样本级预警

样本级汇总只用有传统参考的事件计算：

```text
match_coverage = 已匹配传统峰数 / 可用传统峰数
matched_confidence_p10 = 已匹配峰 comparison_confidence 的第10百分位
sample_confidence = match_coverage × matched_confidence_p10
```

`sample_alert_level` 取该样本全部事件中的最严重级别，额外 AI 峰的预警会传递到样本级。`all_event_comparison_confidence_p10` 汇总正式匹配和最近参考比较事件的第 10 百分位。

## 10. 当前验证结果

### 10.1 阈值 0.6 单样本端到端检查

输入 `test3_1.mzML`，使用 `mrmpformer_special_v2.pth`、模型阈值 0.6、信号 SNR 门 10：

| 指标 | 结果 |
|---|---:|
| 最终峰 | 949 |
| 模型边界峰 | 169 |
| 信号边界峰 | 780 |
| `peak_score` 缺失 | 0 |
| 最低模型分 | 0.60046 |
| 模型分低于旧阈值 0.8 的峰 | 143 |

这说明下调阈值真实地保留了 0.6～0.8 的模型峰，同时所有信号峰也获得了规则分。

### 10.2 旧 test3 推理结果的 V2 兼容回放

旧 `massnova_test3/all.csv` 是阈值 0.8 的历史推理结果。V2 在不重新推理的情况下为旧信号峰补算规则分并完成对齐：

| 项目 | 数量 |
|---|---:|
| 输出事件行 | 51,700 |
| 已匹配峰对 | 38,193 |
| AI 独有且同样本有传统结果 | 9,804 |
| AI 样本无传统结果 | 2,472 |
| 传统峰被 AI 漏掉 | 1,186 |
| 传统参考关键字段缺失 | 45 |
| GREEN / INFO / YELLOW / RED / UNAVAILABLE | 34,498 / 0 / 3,275 / 11,407 / 2,520 |
| 需要复核的事件 | 17,202 |

9,804 个 `AI_ONLY_EXTRA` 中有 9,801 个获得最近传统参考和比较分，另外 3 个因传统关键 RT 缺失为不可用。当前参数下 9,801 个可比较额外峰全部为红色，说明历史推理结果的额外峰规模会显著增加人工复核量，后续需要用人工结论标定候选门控和预警阈值。

样本级最严重状态已包含额外峰预警：58 个历史 test3 样本中 56 个为 RED，2 个为 UNAVAILABLE。

该回放只能验证新置信度逻辑，不能体现阈值从 0.8 降到 0.6 后 test3 全集的最终峰数量。完整效果必须重新运行 58 个 mzML。

## 11. 输出与复现命令

V2 对齐与置信度：

```powershell
cd D:\yinlibo\MRMPFormer\model
conda run -n gamstekpeaking python -m postprocessing.comparison_confidence_v2 `
  --traditional ..\data\MASSNOVA_Tra_Test3.xlsx `
  --ai ..\output\inference\massnova_test3\all.csv `
  --output_dir project_review\confidence_v2_test3_from_existing `
  --config configs\comparison_confidence_v2.json
```

使用阈值 0.6 重新推理 test3 时，`configs/massnova.json` 已设置 `batch_dir=../data/mzml/test3`、`plot=true`。运行后会给每个最终峰生成逐峰图；test3 峰数很多，图片数量与磁盘占用也会明显增加。

```powershell
cd D:\yinlibo\MRMPFormer\model
conda run -n gamstekpeaking python -m inference.cli `
  --config configs\massnova.json `
  --mode massnova `
  --exp_name test3_confidence_v2
```

## 12. 后续标定时不能跳过的事项

1. 用人工复核结论分别校准模型峰和信号峰，检查同一个 `score_peak` 是否代表接近的正确率。
2. 在独立数据上选择模型阈值、绿黄红阈值和硬限制，不能用 test3 同时调参又报告最终性能。
3. 分开统计错误类型：漏峰、额外峰、RT 偏差、左右边界偏差和面积偏差。
4. 软件端展示时同时显示 `score_source` 与 `confidence_scope`，避免把信号规则分或 AI 自身分误读成完整比较置信度。
5. 人工复核记录必须绑定稳定事件 ID 和算法版本，避免重跑后 `peak_no` 变化导致历史结论错位。
