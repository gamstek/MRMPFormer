# Comparison confidence v2 run report

- Schema: `comparison_confidence_v2`
- Traditional input: `D:\yinlibo\MRMPFormer\data\MASSNOVA_Tra_Test3.xlsx`
- AI input: `D:\yinlibo\MRMPFormer\model\project_review\massnova_v2_smoke_test3_1\all.csv`
- Peak/event rows: 39995
- Samples: 56

## Match states

| value | count |
|---|---:|
| `TRAD_ONLY_AI_MISS` | 39001 |
| `AI_ONLY_EXTRA` | 571 |
| `MATCHED_UNIQUE` | 365 |
| `REFERENCE_MISSING` | 45 |
| `MATCHED_AMBIGUOUS` | 13 |

## Alert levels

| value | count |
|---|---:|
| `RED` | 39102 |
| `INFO` | 557 |
| `YELLOW` | 281 |
| `UNAVAILABLE` | 45 |
| `GREEN` | 10 |

## Interpretation

`score_ai` is the raw MRMPFormer score, `score_signal` is a transparent signal-rule
score, and `score_peak` selects the source-appropriate intrinsic score. Matched
events receive `comparison_confidence`; every retained AI peak receives
`final_confidence`. `confidence_scope` distinguishes paired comparison from
AI-only intrinsic scoring. `final_rule_risk_score` is `100 * (1-final_confidence)`.
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
