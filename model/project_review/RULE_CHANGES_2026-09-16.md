# 峰去重与传统预期 RT 容差规则变更（2026-09-16）

## 1. 变更范围

本次变更只修改 Python 推理与置信度计算逻辑，没有修改 `cpp` 软件接口目录。

涉及文件：

- `inference/massnova.py`
- `inference/cli.py`
- `configs/massnova.json`
- `postprocessing/comparison_confidence_v2.py`

## 2. 相邻峰去重规则

旧规则只要求：

```text
峰区间存在任意重叠
且 |RT_apex_1 - RT_apex_2| <= 0.20 min
```

该规则会将“边界只有少量接触、峰间存在深谷”的两个真实峰误判成重复峰。

新规则只有同时满足以下条件才去重：

```text
|RT_apex_1 - RT_apex_2| <= scan_dup_apex_tol
overlap_width / min(width_1, width_2) >= scan_dup_min_overlap_fraction
baseline_corrected_valley / baseline_corrected_smaller_apex
    >= scan_dup_shallow_valley_min_ratio
```

当前参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `scan_dup_apex_tol` | 0.20 min | 峰顶距离上限 |
| `scan_dup_min_overlap_fraction` | 0.25 | 重叠至少占较窄峰区间 25% |
| `scan_dup_shallow_valley_min_ratio` | 0.70 | 谷底达到较小峰峰高 70% 才视为没有深谷 |

只要重叠不明显，或者峰间存在深谷，两个候选峰都会保留。满足全部去重条件时，仍按“模型峰优先于信号峰，同来源保留峰顶更高者”选择代表峰。

### napropamide-2 回归案例

`test3_3 / chrom388 / napropamide-2` 原结果错误删除了 `RT=7.9817 min` 的峰：

| 指标 | 数值 |
|---|---:|
| 两峰顶距离 | 0.1884 min |
| 区间重叠宽度 | 0.0063 min |
| 重叠/较窄区间 | 5.05% |
| 谷底/较小峰顶 | 6.38% |
| 中间峰 SNR | 169.69 |

新规则判断其“重叠不明显且存在深谷”，因此保留。使用 `special_v2 + threshold=0.6` 对 `test3_3` 重新端到端推理后，该通道输出三个峰：

| 峰 | RT | 来源 | 峰分 |
|---|---:|---|---:|
| peak#1 | 7.7719 | model | 0.896 |
| peak#2 | 7.9817 | model | 0.815 |
| peak#3 | 8.2751 | signal_rule | 0.717 |

验证输出位于 `project_review/test3_3_rule_fixcheck/`。

## 3. 传统实际 RT 与预期 RT 的正常容差

传统算法的实际峰 RT 与预期 RT 的绝对偏差在 `0.1 min` 内定义为正常：

```text
trad_expected_rt_diff_abs
    = abs(trad_rt_peak - trad_rt_expected)

trad_expected_rt_excess_abs
    = max(0, trad_expected_rt_diff_abs - 0.10)

conf_trad_expected_rt
    = 2 ^ (-trad_expected_rt_excess_abs / trad_rt_half_confidence_min)
```

因此：

- 绝对偏差 `<= 0.10 min`：`conf_trad_expected_rt=1`，属于正常范围；
- 绝对偏差 `> 0.10 min`：只对超过 0.10 min 的部分降分；
- 新增审计字段：`trad_expected_rt_excess_abs`、`trad_expected_rt_within_tolerance`。

### spinetoram L-1 回归案例

`test3_3 / chrom641 / spinetoram L-1`：

| 指标 | 数值 |
|---|---:|
| 传统实际 RT | 7.614 min |
| 传统预期 RT | 7.521 min |
| 绝对偏差 | 0.093 min |
| 超出正常容差的偏差 | 0 min |
| AI 与传统实际峰 RT 差 | 0.0002 min |
| 对称面积差 | 0.452% |
| 最终置信度 | 0.932 |

新规则结果为 `GREEN`，`review_required=False`，不再产生 `TRAD_REFERENCE_LOW_QUALITY` 黄色预警。

## 4. 验证

执行：

```powershell
python -m unittest tests.test_massnova_peak_dedup tests.test_comparison_confidence_v2 tests.test_weak_extra_alert_rule
```

结果：11 项测试全部通过。相关模块也已通过 `py_compile` 和 JSON 配置语法检查。

此前生成的随机 3000 图属于旧规则历史结果，不会原地覆盖。下一次完整重跑 test3 推理和置信度时会自动使用本文件记录的新规则。
