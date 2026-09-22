# Comparison confidence v2.1 run report

- Schema: `comparison_confidence_v2_1`
- Traditional input: `D:\yinlibo\MRMPFormer\data\MASSNOVA_Tra_Test3.xlsx`
- AI input: `D:\yinlibo\MRMPFormer\output\inference\massnova_test3\all.csv`
- Peak/event rows: 51700
- Samples: 58

## Match states

| value | count |
|---|---:|
| `MATCHED_UNIQUE` | 38154 |
| `AI_ONLY_EXTRA` | 9804 |
| `NO_TRAD_SAMPLE` | 2472 |
| `TRAD_ONLY_AI_MISS` | 1186 |
| `REFERENCE_MISSING` | 45 |
| `MATCHED_AMBIGUOUS` | 39 |

## Alert levels

| value | count |
|---|---:|
| `GREEN` | 34498 |
| `RED` | 11407 |
| `YELLOW` | 3275 |
| `UNAVAILABLE` | 2520 |

## Interpretation

`score_ai` is the raw MRMPFormer score, `score_signal` is a transparent signal-rule
score, and `score_peak` selects the source-appropriate intrinsic score. Formally
matched events compare against the assigned traditional peak. `AI_ONLY_EXTRA`
events compare diagnostically against the nearest traditional peak in the same
sample/channel and receive an unmatched-status penalty without changing the
one-to-one assignment. Missing traditional samples remain unavailable; AI misses
receive zero. `final_rule_risk_score` is `100 * (1-final_confidence)`.
Thresholds remain provisional until calibrated against manual review labels.

## Configuration

```json
{
  "match_rt_gate_min": 0.5,
  "match_max_cost": 1.0,
  "match_rt_weight": 0.7,
  "match_boundary_weight": 0.25,
  "match_peak_tiebreak_weight": 0.05,
  "ambiguity_margin": 0.1,
  "ambiguous_match_factor": 0.6,
  "extra_noncompetitive_factor": 0.75,
  "extra_competitive_factor": 0.5,
  "rt_half_confidence_min": 0.1,
  "signal_score_snr_pivot": 10.0,
  "signal_score_points_good": 10.0,
  "signal_score_snr_weight": 0.8,
  "signal_score_points_weight": 0.2,
  "confidence_weight_peak": 0.35,
  "confidence_weight_rt": 0.4,
  "confidence_weight_area": 0.25,
  "green_confidence_min": 0.8,
  "yellow_confidence_min": 0.6,
  "hard_red_rt_min": 0.3,
  "hard_red_area_diff_pct": 120.0,
  "trad_rt_half_confidence_min": 0.1,
  "trad_snr_low": 3.0,
  "trad_snr_good": 10.0,
  "trad_quality_weight_rt": 0.6,
  "trad_quality_weight_snr": 0.4,
  "trad_quality_green_min": 0.8,
  "trad_quality_yellow_min": 0.6
}
```
