# Comparison confidence v1 run report

- Schema: `comparison_confidence_v1`
- Traditional input: `D:\yinlibo\MRMPFormer\data\MASSNOVA_Tra_Test3.xlsx`
- AI input: `D:\yinlibo\MRMPFormer\output\inference\massnova_test3\all.csv`
- Peak/event rows: 51700
- Samples: 58

## Match states

| value | count |
|---|---:|
| `MATCHED_UNIQUE` | 38155 |
| `AI_ONLY_EXTRA` | 9804 |
| `NO_TRAD_SAMPLE` | 2472 |
| `TRAD_ONLY_AI_MISS` | 1186 |
| `REFERENCE_MISSING` | 45 |
| `MATCHED_AMBIGUOUS` | 38 |

## Alert levels

| value | count |
|---|---:|
| `GREEN` | 32622 |
| `INFO` | 9765 |
| `RED` | 5247 |
| `UNAVAILABLE` | 2517 |
| `YELLOW` | 1549 |

## Interpretation

`score_ai` is the raw MRMPFormer softmax score. `comparison_confidence` combines
match quality, AI score, RT agreement and area agreement. `final_rule_risk_score`
is `100 * (1 - comparison_confidence)`; hard rules determine `alert_level` and
`review_status`. Thresholds in this run are provisional and require calibration
against independent manual review labels.

## Configuration

```json
{
  "match_rt_gate_min": 0.5,
  "match_max_cost": 1.0,
  "match_rt_weight": 0.7,
  "match_boundary_weight": 0.25,
  "match_ai_tiebreak_weight": 0.05,
  "ambiguity_margin": 0.1,
  "ambiguous_match_factor": 0.6,
  "rt_half_confidence_min": 0.1,
  "area_half_confidence_log2": 1.0,
  "missing_ai_score_confidence": 0.25,
  "confidence_weight_ai": 0.35,
  "confidence_weight_rt": 0.4,
  "confidence_weight_area": 0.25,
  "green_confidence_min": 0.8,
  "yellow_confidence_min": 0.6,
  "hard_red_rt_min": 0.3,
  "hard_red_area_fold": 4.0,
  "trad_rt_half_confidence_min": 0.1,
  "trad_snr_low": 3.0,
  "trad_snr_good": 10.0,
  "trad_quality_weight_rt": 0.6,
  "trad_quality_weight_snr": 0.4,
  "trad_quality_green_min": 0.8,
  "trad_quality_yellow_min": 0.6
}
```
