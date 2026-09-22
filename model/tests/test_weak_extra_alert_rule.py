import math

import pandas as pd

from postprocessing.weak_extra_alert_rule import apply_weak_extra_alert_rule


def _row(row_id, uid, rt, status="AI_ONLY_EXTRA", source="signal_rule", height=5.0,
         area=5.0, chrom=None, relative_height=0.05):
    return {
        "ai_row_id": row_id,
        "sample_id": "test3_3",
        "uid": uid,
        "ai_chrom_index": row_id + 1 if chrom is None else chrom,
        "ai_rt_peak": rt,
        "ai_area": area,
        "ai_apex_intensity": height,
        "relative_height_baseline_corrected": relative_height,
        "match_status": status,
        "score_source": source,
        "alert_level": "RED",
        "alert_reasons": status,
        "review_required": True,
        "review_status": "PENDING",
    }


def test_weak_unpaired_signal_extra_becomes_non_review_info():
    rows = pd.DataFrame([
        _row(1, "compound-1", 5.0, chrom=1),
        _row(2, "compound-1", 9.0, source="model", area=100.0, chrom=1,
             relative_height=1.0),
    ])
    result = apply_weak_extra_alert_rule(rows)
    assert bool(result.loc[0, "weak_extra_ignored"])
    assert result.loc[0, "alert_level"] == "INFO"
    assert not bool(result.loc[0, "review_required"])
    assert "IGNORED_WEAK_AI_EXTRA" in result.loc[0, "alert_reasons"]


def test_matched_model_and_no_traditional_rows_are_protected():
    rows = pd.DataFrame([
        _row(1, "a-1", 5.0, status="MATCHED_UNIQUE", chrom=1),
        _row(2, "b-1", 6.0, source="model", chrom=2),
        _row(3, "c-1", 7.0, status="NO_TRAD_SAMPLE", chrom=3),
        _row(10, "a-1", 9.0, source="model", area=100.0, chrom=1, relative_height=1.0),
        _row(11, "b-1", 9.0, source="model", area=100.0, chrom=2, relative_height=1.0),
        _row(12, "c-1", 9.0, source="model", area=100.0, chrom=3, relative_height=1.0),
    ])
    result = apply_weak_extra_alert_rule(rows)
    assert not result.loc[:2, "weak_extra_ignored"].any()
    assert result.loc[:2, "alert_level"].eq("RED").all()


def test_coeluting_transition_pair_rescues_both_peaks():
    rows = pd.DataFrame([
        _row(1, "same-1", 5.00, chrom=1),
        _row(2, "same-2", 5.06, chrom=2),
        _row(3, "same-1", 9.00, source="model", area=100.0, chrom=1, relative_height=1.0),
        _row(4, "same-2", 9.00, source="model", area=100.0, chrom=2, relative_height=1.0),
    ])
    result = apply_weak_extra_alert_rule(rows, paired_transition_rt_tolerance_min=0.10)
    assert result.loc[:1, "paired_transition_supported"].all()
    assert not result.loc[:1, "weak_extra_ignored"].any()
    assert math.isclose(float(result.loc[0, "paired_transition_rt_abs"]), 0.06, abs_tol=1e-9)
