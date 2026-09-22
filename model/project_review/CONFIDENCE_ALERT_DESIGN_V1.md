# MRMPFormer 与 MassNova 传统算法匹配、置信度及预警设计 v1

## 1. 文档目的与边界

本文记录 `comparison_confidence_v1` 的业务定义、输入数据、峰匹配、分数计算、预警规则、输出字段、test3 验证结果和后续标定计划。目的是让每一次预警都有可以回溯的原始量与计算路径。

本版本是**离线原型**，没有接入或修改仓库根目录 `cpp/` 下的软件接口。只有在离线规则完成标定、接口字段冻结并经过单独评审后，才进入 C++/JSON 接口实现。

本项目中的名称固定如下：

- **AI 峰识别链路**：候选峰算法 + MRMPFormer + 边界/面积后处理。
- **传统算法**：MassNova 软件当前已有的峰识别和积分算法。
- **代码 `massnova` 模式**：Python 中整条 XIC 无标签推理模式，不等于 MassNova 软件传统算法。
- **`score_ai`**：MRMPFormer 峰类相对背景类的 Softmax 分数。
- **`quality_trad_rule`**：根据传统输出字段计算的规则质量分，不是模型分数。
- **`comparison_confidence`**：两套结果成功匹配后计算的复合置信度。
- **`final_rule_risk_score`**：面向预警排序的风险分，值越高风险越高。
- **`review_status`**：人工复核工作流状态，不是算法分数。

## 2. 为什么必须先匹配再计算置信度

同一样本、同一离子通道中，AI 链路可能输出多个峰，而传统算法通常输出一个目标峰。如果直接比较任意两个 RT 或面积，会出现以下错误：

1. 把邻近杂峰的面积与传统目标峰面积比较；
2. 同一个 AI 峰被分配给两个传统峰；
3. 面积相近但 RT 完全错误的峰被强行匹配；
4. 匹配失败被隐藏成一个普通的低分。

所以处理顺序固定为：

```text
样本与通道精确分组
→ 构造允许匹配的峰对
→ 一对一最优分配
→ 判断唯一/歧义/漏检/额外峰
→ 仅对成功匹配峰计算 RT、面积和 AI 子分数
→ 规则生成预警和人工复核状态
```

## 3. 输入数据

### 3.1 传统算法输入

当前验证文件：`data/MASSNOVA_Tra_Test3.xlsx`，`Sheet1` 共 39,424 行。

核心字段映射：

| 业务字段 | Excel列 |
|---|---|
| 样本ID | `Unnamed: 0` |
| 样本名 | `样品名` |
| 完整离子通道ID | `化合物` |
| 离子类型 | `离子类型` |
| 峰顶RT | `保留时间` |
| 预期RT | `预期保留时间` |
| 起点RT | `峰起始时间`中括号前的数值 |
| 终点RT | `峰结束时间`中括号前的数值 |
| 面积 | `分析物峰面积` |
| 峰高 | `分析物峰高度` |
| SNR | `信噪比` |
| 离子比率 | `离子比率` |

test3 中有 45 行缺少 RT、边界和面积，332 行缺少传统 SNR。缺失值必须保留为缺失，禁止用 0 替代。

### 3.2 AI 输入

当前验证文件：`output/inference/massnova_test3/all.csv`，共 50,469 个最终峰。

关键字段包括：

- `mzml_stem`：样本ID；
- `uid`：完整离子通道ID；
- `rt_min/rt_peak/rt_max`：AI 峰边界与峰顶；
- `area`：AI 积分面积；
- `model_score`：MRMPFormer 原始 `score_ai`；
- `boundary_source`：`model` 或 `signal`；
- `validated`：是否由模型分数超过推理阈值确认。

当前 test3 使用原有 SciPy 候选模式的结果。CentreWave 直接替换实验召回较差，暂不作为本版置信度输入。

## 4. 分组主键

匹配只能发生在：

```text
sample_id 完全相同 AND uid 完全相同
```

`uid` 中的 `-1/-2` 后缀必须保留，因为它们代表不同离子通道。禁止跨样本、跨化合物、跨定量/定性通道按最近 RT 匹配。

本版只做精确 ID 匹配，不做模糊名称匹配。生产接口应提供稳定的 `sample_id` 和 `compound/transition_id`，避免显示名称变化破坏匹配。

## 5. 峰事件一对一匹配

### 5.1 候选硬门控

传统峰 `T` 和 AI 峰 `A` 至少满足一项才允许进入匹配矩阵：

```text
abs(A.rt_peak - T.rt_peak) <= 0.50 min
或
两峰的 [rt_min, rt_max] 区间有交集
```

`0.50 min` 是 v1 起始值，不是已经验证的生产阈值。允许区间相交用于兼容宽峰：宽峰峰顶可能相差较大，但两个算法实际上覆盖同一色谱事件。

### 5.2 区间 IoU

一维峰区间交并比：

```text
IoU = intersection_length / union_length
```

完全重合为 1，无交集为 0。传统边界缺失时，匹配代价中的边界项使用中性惩罚 0.5，并保留缺失状态。

### 5.3 匹配代价

对每个允许峰对计算：

```text
rt_part       = min(delta_rt_abs / 0.50, 2)
boundary_part = 1 - interval_iou
ai_part       = 1 - score_ai

match_cost = 0.70 * rt_part
           + 0.25 * boundary_part
           + 0.05 * ai_part
```

没有 `score_ai` 的信号兜底峰，`ai_part=1`。AI 分数只占 5%，用途是在位置、边界接近的候选之间决胜，不能用一个高 AI 分数覆盖明显错误的 RT。

**面积不参与匹配。** 面积只有在峰已经匹配后才参与置信度，避免面积相近的错误色谱事件被配在一起。

### 5.4 一对一分配

每个样本和通道内使用匈牙利算法求总代价最小的一对一分配：

- 每个传统峰最多匹配一个 AI 峰；
- 每个 AI 峰最多匹配一个传统峰；
- 代价大于 1.0 的分配拒绝；
- 实现兼容未来传统算法在一个通道输出多个峰的情况。

### 5.5 匹配歧义

记录最佳和第二候选代价：

```text
match_margin = second_best_cost - match_cost
```

当 `match_margin < 0.10` 时标记 `MATCHED_AMBIGUOUS`。匹配结果仍保存，但必须复核。

### 5.6 未选中的 AI 峰

未选中峰继续保留，不能静默删除，但分两类处理：

- **竞争峰**：其代价不高于已选峰代价加 0.10，可能与目标峰混淆，黄色预警；
- **远端额外峰**：没有形成实际竞争，只记为 `INFO`，不污染样本置信度。

这一区分很关键。test3 中共有 9,804 个未选 AI 峰，其中 5,726 个在宽松匹配门内，但真正达到近似竞争条件的只有 39 个。若把全部未选峰都判红，会产生不可用的预警量。

## 6. 匹配状态

| 状态 | 含义 | 处理 |
|---|---|---|
| `MATCHED_UNIQUE` | 找到稳定的一对一匹配 | 计算复合置信度 |
| `MATCHED_AMBIGUOUS` | 第一、第二候选过近 | 计算分数并强制红色复核 |
| `TRAD_ONLY_AI_MISS` | 传统有峰，AI 无可接受匹配 | 红色复核 |
| `AI_ONLY_EXTRA` | AI 峰未被选择 | 竞争峰黄警，远端峰仅 INFO |
| `REFERENCE_MISSING` | 传统核心字段缺失 | 置信度不可用，进入复核 |
| `NO_TRAD_SAMPLE` | AI 样本无对应传统样本 | 置信度不可用，进入复核 |

未匹配和缺失状态本身就是结果，不能强行配峰后再输出一个普通低分。

## 7. AI 分数

```text
conf_ai = clip(score_ai, 0, 1)
```

`score_ai` 是模型峰类与背景类 logits 的 Softmax 结果，不包含传统 RT、面积或明确的 SNR 数值。

信号兜底峰没有模型分数。v1 数值计算临时使用：

```text
conf_ai = 0.25
```

同时添加 `SCORE_AI_MISSING` 并强制红色复核。这个 0.25 只用于风险排序，不能显示成模型置信度，原始 `score_ai` 仍保持空值。

当前 AI 输出只保留了模型阈值 0.8 以上的确认峰，所以有效 `score_ai` 主要集中在 0.8～0.99，存在选择偏差。后续做概率校准时应保存阈值前的原始 query 分数。

## 8. RT 一致性分数

```text
delta_rt_abs = abs(rt_ai - rt_trad)
conf_rt = 2 ^ (-delta_rt_abs / 0.10)
```

解释：

| RT差 | `conf_rt` |
|---:|---:|
| 0 | 1.000 |
| 0.05 min | 0.707 |
| 0.10 min | 0.500 |
| 0.20 min | 0.250 |
| 0.30 min | 0.125 |

当差值达到 0.30 min 时触发 `RT_HARD_LIMIT` 红色预警。

## 9. 面积一致性分数

业务要求的绝对面积差原样输出：

```text
area_diff_abs = abs(area_ai - area_trad)
```

由于面积跨化合物和浓度可相差多个数量级，绝对差不能直接共用一个全局阈值，因此同时输出：

```text
area_diff_rel  = abs(area_ai - area_trad) / max(area_ai, area_trad, eps)
area_log_ratio = abs(log((area_ai + eps) / (area_trad + eps)))
area_fold      = exp(area_log_ratio)
```

v1 面积分数：

```text
conf_area = 2 ^ (-abs(log2(area_ai / area_trad)))
```

等价于按倍数差衰减：

| 面积倍数差 | `conf_area` |
|---:|---:|
| 1倍 | 1.000 |
| 1.5倍 | 0.667 |
| 2倍 | 0.500 |
| 4倍 | 0.250 |

面积达到 4 倍差触发 `AREA_HARD_LIMIT` 红色预警。

两种算法的基线扣除和积分细节必须核对。如果积分口径不同，面积差反映的是“算法差异”，不必然代表 AI 错误；这正是后续必须按人工结果标定的原因。

## 10. 匹配质量分

```text
conf_match = exp(-match_cost)
```

歧义匹配额外乘 0.60。该分数反映匹配关系本身是否稳定，与 `conf_rt`、`conf_area` 和 `conf_ai` 分列保存。

## 11. 综合置信度

v1 使用可解释的加权几何平均：

```text
comparison_confidence
    = conf_match
    * conf_ai   ^ 0.35
    * conf_rt   ^ 0.40
    * conf_area ^ 0.25
```

RT 权重最高，因为同一色谱事件首先应在保留时间上相符；AI 分数其次；面积受积分口径和浓度影响较大，初始权重最低。

几何平均的作用是：任何一个关键因素很差都会明显拉低总分，避免其他高分完全补偿严重异常。

这些权重是可审计的起始参数，不是已标定结论。

## 12. 传统峰质量怎么处理

传统 Excel 没有统一的原生峰置信度，因此禁止把 SNR 改名为 `score_trad`。v1 单独计算 `quality_trad_rule`：

### 12.1 传统预期 RT 分

```text
trad_expected_rt_diff_abs = abs(trad_rt_peak - trad_rt_expected)
conf_trad_expected_rt = 2 ^ (-trad_expected_rt_diff_abs / 0.10)
```

### 12.2 传统 SNR 分

- `SNR <= 3`：0；
- `SNR >= 10`：1；
- 3～10：在 `log(1+SNR)` 空间线性插值；
- SNR 缺失：保持空值，使用剩余可用成分。

### 12.3 传统规则质量

```text
quality_trad_rule
    = conf_trad_expected_rt ^ 0.60
    * conf_trad_snr         ^ 0.40
```

只有一个成分可用时，权重重新归一化；两个成分都不可用时为 `UNAVAILABLE`。

该分数暂时不进入 `comparison_confidence`。如果传统参照质量为红色，而匹配结果原本为绿色，只提升为黄色并附加 `TRAD_REFERENCE_LOW_QUALITY`，表示需要判断偏差来自 AI 还是传统结果。

离子比率当前没有可靠的目标比率和允许偏差，先原样保存，不进入规则分。得到方法学限值后再增加。

## 13. 最终规则分和人工复核状态

```text
final_rule_risk_score = 100 * (1 - comparison_confidence)
```

数值含义：

- `comparison_confidence` 越高越可信；
- `final_rule_risk_score` 越高越需要优先复核。

基础颜色阈值：

- `GREEN`：置信度不低于 0.80；
- `YELLOW`：0.60～0.80；
- `RED`：低于 0.60。

硬规则可以覆盖数值颜色：

- 传统峰漏检：红；
- 匹配歧义：红；
- `score_ai` 缺失：红；
- RT差不低于 0.30 min：红；
- 面积达到4倍差：红；
- 传统参考质量红且其他结果绿：提升到黄；
- 参考字段/传统样本缺失：`UNAVAILABLE`；
- 远端未选AI峰：`INFO`。

人工状态独立保存：

- `PENDING`：等待人工复核；
- `NOT_REQUIRED`：当前规则不要求复核；
- `review_decision/reviewer/reviewed_at/review_comment`：预留给复核系统填写。

算法再次运行时不得用新算法状态覆盖已经完成的人工结论。生产实现应将算法运行记录和人工复核记录分表或使用不可变版本ID关联。

## 14. 样本级置信度

样本级统计只使用具有传统参照的行，不把远端额外AI峰加入分数。

```text
match_coverage = 成功匹配传统峰数 / 有效传统峰数
confidence_p10 = 匹配峰 comparison_confidence 的10%分位数
sample_confidence = match_coverage * confidence_p10
```

不用普通平均值，是因为704个通道的平均分很容易掩盖少数严重漏检；也不直接取最小值，因为一个低信号背景通道会让所有样本长期为零。10%分位数是保守起点，仍需根据实际人工复核负担标定。

样本汇总同时输出绿/黄/红数量、AI漏检数、额外峰数、歧义数和待复核数。界面应优先显示“需要复核的化合物列表与原因”，不能只显示一个样本总分。

## 15. 输出文件

### `peak_confidence.csv`

每个传统事件和保留的AI额外峰一行，包含：

- 原始传统字段与AI字段；
- `match_status`；
- `match_cost/second_best_cost/match_margin`；
- `ai_candidates_total/ai_candidates_eligible`；
- `unmatched_ai_peak_count/competitive_ai_peak_count`；
- RT、边界、面积的所有原始差值；
- `score_ai`；
- `quality_trad_rule`及子分数；
- `conf_match/conf_ai/conf_rt/conf_area`；
- `comparison_confidence/final_rule_risk_score`；
- `alert_level/alert_reasons`；
- 人工复核字段。

### `sample_alert_summary.csv`

每个样本一行，包含样本置信度、匹配覆盖率、置信度分位数、各等级数量以及漏检/额外/歧义统计。

### `confidence_run.json`

保存算法版本、输入绝对路径和本次运行的完整参数快照。

### `confidence_report.md`

保存本次运行的数据量、匹配状态和预警数量，便于快速审计。

## 16. test3 v1 离线结果

输入：原 SciPy 候选模式的 `massnova_test3/all.csv`。

匹配状态：

| 状态 | 数量 |
|---|---:|
| `MATCHED_UNIQUE` | 38,155 |
| `MATCHED_AMBIGUOUS` | 38 |
| `TRAD_ONLY_AI_MISS` | 1,186 |
| `REFERENCE_MISSING` | 45 |
| `AI_ONLY_EXTRA` | 9,804 |
| `NO_TRAD_SAMPLE` | 2,472 |

`NO_TRAD_SAMPLE` 来自 AI 中存在 `test3_57/58`，而当前传统 Excel 只覆盖 `test3_1～56`。

预警/信息状态：

| 等级 | 数量 |
|---|---:|
| `GREEN` | 32,622 |
| `YELLOW` | 1,549 |
| `RED` | 5,247 |
| `INFO` | 9,765 |
| `UNAVAILABLE` | 2,517 |

传统参照规则质量：

| 等级 | 数量 |
|---|---:|
| `GREEN` | 35,873 |
| `YELLOW` | 1,236 |
| `RED` | 2,270 |
| `UNAVAILABLE` | 45 |

这些数字是规则体检结果，不是模型准确率。当前56个有传统参照的样本都至少存在一个红色通道，说明阈值不能直接作为生产报警配置，也说明需要在化合物/离子对层面分析红色原因与真实人工错误。

## 17. 标定计划

生产前至少需要建立一套独立人工复核集，每个“样本 × 化合物 × 离子通道”记录：

1. 传统峰是否正确；
2. AI 峰是否正确；
3. 两峰是否代表同一事件；
4. AI 起点/终点是否可接受；
5. 传统起点/终点是否可接受；
6. 哪套面积更接近人工积分；
7. 是否必须人工复核；
8. 特殊峰类型：宽峰、肩峰、开叉峰、拖尾峰、小峰、噪声峰。

然后依次标定：

1. 匹配 RT 门限和匹配代价，先保证同峰匹配召回；
2. 歧义 margin，控制错误配峰；
3. `score_ai` 的温度缩放或等距回归校准；
4. RT 与面积衰减尺度；
5. 按化合物、浓度和仪器批次检查面积偏差；
6. 绿/黄/红阈值，使漏报风险和人工复核量达到业务目标；
7. 在从未用于调参的验证集上冻结并报告结果。

标定指标至少包含：预警召回率、预警精确率、漏警率、每样本复核数量、传统错峰发现率、AI错峰发现率和匹配错误率。

## 18. 当前限制和后续工作

1. 当前阈值与权重是透明初值，没有人工真值标定；
2. `score_ai` 未校准，且当前输出受0.8推理阈值截断；
3. 信号兜底峰没有模型分数，当前必须复核；
4. 两套算法的积分基线口径尚未逐项确认；
5. 传统离子比率缺少期望值和方法学允许误差；
6. 目前匹配按完整通道完成，定量/定性离子对的联合判断尚未加入；
7. 样本级阈值尚未按可接受复核工作量标定；
8. 当前实现是 Python 离线工具，没有改动 C++ 接口和 JSON 协议。

下一阶段应先人工抽查 test3 的红色、黄色和绿色样本，确认匹配是否正确，再调整参数。确认算法定义后，再单独设计接口字段和兼容版本。

### 18.1 test3 逐峰对比图集

逐峰人工抽查图位于：

`model/project_review/confidence_v1_test3/peak_plots/`

打开其中的 `index.html` 可以浏览传统峰与 AI 组合算法峰的 XIC 叠加图。图中绿色表示传统算法峰区间，蓝色表示 MRMPFormer＋信号算法峰区间。当前按八类预警原因各抽取12张、绿色对照16张、传统字段缺失6张，共118张，绘图失败0张。

对应绘图程序：

`model/postprocessing/plot_confidence_peak_gallery.py`

## 19. 代码与运行方式

实现：`model/postprocessing/comparison_confidence.py`

配置：`model/configs/comparison_confidence_v1.json`

测试：`model/tests/test_comparison_confidence.py`

预警统计与图表：`model/project_review/confidence_v1_test3/visualizations/`

图表生成脚本：`model/postprocessing/plot_confidence_alerts.py`

离线运行：

```powershell
cd D:\yinlibo\MRMPFormer\model
python -m postprocessing.comparison_confidence `
  --traditional ..\data\MASSNOVA_Tra_Test3.xlsx `
  --ai ..\output\inference\massnova_test3\all.csv `
  --output_dir project_review\confidence_v1_test3 `
  --config configs\comparison_confidence_v1.json
```

生成预警统计和图表：

```powershell
python -m postprocessing.plot_confidence_alerts `
  --peak_csv project_review\confidence_v1_test3\peak_confidence.csv `
  --sample_csv project_review\confidence_v1_test3\sample_alert_summary.csv `
  --output_dir project_review\confidence_v1_test3\visualizations
```

核心算法和接口协议应使用独立版本号。当前版本固定为：

```text
comparison_confidence_v1
```
