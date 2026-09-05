# -*- coding: utf-8 -*-
"""
[隔离实验] 特殊峰专项评估。

复用 evaluate_baseline 的对齐/命中逻辑（parse_labels_xlsx / match_image 等，零改动），
增加特殊峰子集口径：label 行 tag 非空 → 该行全部 GT 框计入特殊峰 GT；
通道名无法对齐到 feature 的行（如 联苯菊酯/联苯三唑醇 错配）自然退出统计。

用法（model/ 目录下）：
  python -m tools.evaluation.evaluate_special_isolation ^
      --labels <test2_v2.xlsx 或原版 test2.xlsx> ^
      --prediction_csvs test2_1=<prediction.csv> test2_2=... test2_3=... ^
      --feature_csvs   test2_1=<feature.csv>    test2_2=... test2_3=... ^
      --score 0.8 --tol 0.05 ^
      --output_dir ../output/special_peak_isolation/eval/<name>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.evaluation.evaluate_baseline import (
    _build_pred_by_img,
    _load_pred_rows,
    _parse_gt_peaks,
    match_image,
)
from preprocessing.coco_annotation import (
    group_labels_by_sample,
    label_key,
    map_samples_to_mzmls,
    parse_labels_xlsx,
)
from preprocessing.label_qc import check_label_rt_consistency, mark_excluded_labels

# 隔离口径：tag 列不在 parse_labels_xlsx 白名单，独立读回并以 (sample_id, compound, channel) 合并
_TAG_KEYS = ("sample_id", "comonent", "component", "channel", "tag")


def _read_tags(labels_path):
    """读取 {roi_id: tag}（v2 标签只改峰边界，tag 列保留原样并追加 ;v2 标记）。"""
    import pandas as pd
    try:
        df = pd.read_excel(labels_path)
    except Exception:
        return {}
    comp_col = "comonent" if "comonent" in df.columns else ("component" if "component" in df.columns else None)
    if comp_col is None or "tag" not in df.columns or "roi_id" not in df.columns:
        return {}
    tags = {}
    for _, r in df.iterrows():
        t = str(r["tag"]).strip() if pd.notna(r["tag"]) else ""
        if t:
            tags[int(r["roi_id"])] = t
    return tags


def evaluate_special(pred_feat_map, labels_path, tol, min_score, qc_label_rt_tol=1.0):
    tags = _read_tags(labels_path)
    labels = parse_labels_xlsx(labels_path)
    for rec in labels:
        rid = rec.get("roi_id")
        try:
            rid = int(rid)
        except (TypeError, ValueError):
            rid = None
        if rid is not None and rid in tags:
            rec["tag"] = tags[rid]
        else:
            rec["tag"] = ""

    excl_by_sample = {}
    if qc_label_rt_tol and qc_label_rt_tol > 0:
        _qc_rows, _keys = check_label_rt_consistency(labels, tol=qc_label_rt_tol)
        mark_excluded_labels(labels, _keys)
        for _sid, _comp, _ch in _keys:
            _kid = label_key(_comp, _ch)
            if _kid:
                excl_by_sample.setdefault(_sid, set()).add(_kid)
        labels = [r for r in labels if not r.get("_qc_excluded")]

    sample_order, groups = group_labels_by_sample(labels)
    stem2sample = map_samples_to_mzmls(list(pred_feat_map), sample_order, None)

    tp = fp = fn = 0
    sp_tp = sp_gt = 0
    audit = []
    for stem, paths in pred_feat_map.items():
        pred_rows = _load_pred_rows(paths["pred"], min_score)
        pred_by_img = _build_pred_by_img(pred_rows, min_score)

        by_key = {}
        for rec in groups[stem2sample[stem]]:
            k = label_key(rec.get("compound"), rec.get("channel"))
            if k:
                by_key.setdefault(k, rec)
            _raw = str(rec.get("compound") or "").strip()
            if _raw:
                by_key.setdefault(_raw, rec)
        _excl_nids = excl_by_sample.get(stem2sample[stem], set())

        feat = pd.read_csv(paths["feat"])
        for i, frow in feat.iterrows():
            n = i + 1
            native_id = str(frow["native_id"]).strip()
            if native_id in _excl_nids:
                continue
            img_name = next((im for im in pred_by_img if im.startswith(f"{n}_mz")), None)
            rows = pred_by_img.get(img_name, []) if img_name else []
            rec = by_key.get(native_id)
            gt_peaks = _parse_gt_peaks(rec) if rec is not None else []
            is_special = bool(rec is not None and str(rec.get("tag") or "").strip())

            pairs, fps, fns, _loose = match_image(rows, gt_peaks, tol, loose_tol=0)
            tp += len(pairs)
            fp += len(fps)
            fn += len(fns)
            if is_special:
                sp_tp += len(pairs)
                sp_gt += len(gt_peaks)
                for g in gt_peaks:
                    hit_pred = next((pr for pr, gg in pairs if gg == g), None)
                    audit.append({
                        "stem": stem, "native_id": native_id, "tag": str(rec.get("tag")).strip(),
                        "gt_start": g[0], "gt_end": g[1],
                        "hit": hit_pred is not None,
                        "pred_start": hit_pred.get("rt_min") if hit_pred else None,
                        "pred_end": hit_pred.get("rt_max") if hit_pred else None,
                        "score": hit_pred.get("score") if hit_pred else None,
                    })

    def prf(t, f, n):
        p = t / (t + f) if (t + f) else 0.0
        r = t / (t + n) if (t + n) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    P, R, F1 = prf(tp, fp, fn)
    return {
        "score": min_score, "tol": tol,
        "TP": tp, "FP": fp, "FN": fn,
        "precision": round(P, 4), "recall": round(R, 4), "f1": round(F1, 4),
        "special_tp": sp_tp, "special_gt": sp_gt,
        "special_rate": round(sp_tp / sp_gt, 4) if sp_gt else 0.0,
    }, pd.DataFrame(audit)


def main():
    ap = argparse.ArgumentParser(description="[隔离实验] 特殊峰专项评估")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--prediction_csvs", nargs="*", required=True)
    ap.add_argument("--feature_csvs", nargs="*", required=True)
    ap.add_argument("--score", type=float, default=0.8)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--qc_label_rt_tol", type=float, default=1.0)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    preds = dict(item.split("=", 1) for item in args.prediction_csvs)
    feats = dict(item.split("=", 1) for item in args.feature_csvs)
    pred_feat_map = {k: {"pred": v, "feat": feats[k]} for k, v in preds.items()}

    metrics, audit = evaluate_special(pred_feat_map, args.labels, args.tol,
                                      args.score, args.qc_label_rt_tol)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "special_audit.csv", index=False, encoding="utf-8-sig")
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"labels": args.labels, **metrics}, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
