# test3 随机峰置信度与预警图集

- 输入置信度表：`D:\yinlibo\MRMPFormer\model\project_review\confidence_v2_test3_from_existing\peak_confidence.csv`
- mzML 目录：`D:\yinlibo\MRMPFormer\data\mzml\test3`
- 样本范围：`test3_3` ～ `test3_58`
- 固定随机种子：`20260915`
- 请求抽样数：5
- 实际抽样数：5
- 成功出图：5
- 失败：0
- 耗时：2.0 秒

## 预警等级

| value | count |
|---|---:|
| `GREEN` | 4 |
| `UNAVAILABLE` | 1 |

## 匹配状态

| value | count |
|---|---:|
| `MATCHED_UNIQUE` | 4 |
| `NO_TRAD_SAMPLE` | 1 |

## 峰分来源

| value | count |
|---|---:|
| `model` | 3 |
| `signal_rule` | 2 |

## 图片说明

- 蓝色：MRMPFormer + 信号算法最终 AI 峰。
- 绿色：正式一对一匹配的传统峰。
- 橙色：`AI_ONLY_EXTRA` 使用的最近传统诊断参考，不是正式匹配。
- 图中显示峰自身分、分数来源、RT 绝对差、面积绝对差、面积差百分比、比较置信度、最终置信度、风险分、预警等级和原因。
- `NO_TRAD_SAMPLE` 没有传统参考，比较置信度和最终置信度显示为 `--`，预警为 `UNAVAILABLE`。
