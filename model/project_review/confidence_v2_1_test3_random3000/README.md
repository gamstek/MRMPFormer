# test3 随机峰置信度与预警图集

- 输入置信度表：`D:\yinlibo\MRMPFormer\model\project_review\confidence_v2_test3_from_existing\peak_confidence.csv`
- mzML 目录：`D:\yinlibo\MRMPFormer\data\mzml\test3`
- 样本范围：`test3_3` ～ `test3_58`
- 固定随机种子：`20260915`
- 请求抽样数：3000
- 实际抽样数：3000
- 成功出图：3000
- 失败：0
- 耗时：361.3 秒

- 推理阈值说明：历史 massnova_test3/all.csv 使用模型阈值 0.8；本轮未重新推理

## 预警等级

| value | count | proportion |
|---|---:|---:|
| `GREEN` | 2123 | 70.77% |
| `RED` | 548 | 18.27% |
| `YELLOW` | 166 | 5.53% |
| `UNAVAILABLE` | 163 | 5.43% |

## 匹配状态

| value | count | proportion |
|---|---:|---:|
| `MATCHED_UNIQUE` | 2299 | 76.63% |
| `AI_ONLY_EXTRA` | 538 | 17.93% |
| `NO_TRAD_SAMPLE` | 163 | 5.43% |

## 峰分来源

| value | count | proportion |
|---|---:|---:|
| `model` | 2157 | 71.90% |
| `signal_rule` | 843 | 28.10% |

## 图片说明

- 蓝色：MRMPFormer + 信号算法最终 AI 峰。
- 绿色：正式一对一匹配的传统峰。
- 橙色：`AI_ONLY_EXTRA` 使用的最近传统诊断参考，不是正式匹配。
- 紫色：已经正式匹配该最近传统峰的另一个 AI 峰。
- 图中显示峰自身分、分数来源、RT 绝对差、面积绝对差、面积差百分比、比较置信度、最终置信度、风险分、预警等级和原因。
- `NO_TRAD_SAMPLE` 没有传统参考，比较置信度和最终置信度显示为 `--`，预警为 `UNAVAILABLE`。

## 结果解读

- 默认无需复核的 GREEN：2123 个，占 70.77%。
- 需要复核的 YELLOW、RED、UNAVAILABLE：877 个，占 29.23%。
- 正式匹配峰：2299 个，占 76.63%；其中 176 个需要复核，占正式匹配峰的 7.66%。
- 额外 AI 峰：538 个，占 17.93%。
- 无传统样本参考：163 个，占 5.43%。

## 主要预警原因

| reason | count |
|---|---:|
| `COMPOSITE_BELOW_YELLOW` | 559 |
| `AI_ONLY_EXTRA` | 538 |
| `COMPARISON_REFERENCE_NEAREST` | 538 |
| `RT_HARD_LIMIT` | 426 |
| `AREA_HARD_LIMIT` | 302 |
| `NO_TRAD_SAMPLE` | 163 |
| `COMPOSITE_BELOW_GREEN` | 139 |
| `TRAD_REFERENCE_LOW_QUALITY` | 42 |

## 数据边界

本图集用于检查置信度、最近传统参考和预警绘图逻辑。若输入峰表来自旧阈值推理，本图集中的峰数量和预警比例不代表新阈值重新推理后的结果。
