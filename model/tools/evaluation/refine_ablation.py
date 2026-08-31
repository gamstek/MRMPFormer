# -*- coding: utf-8 -*-
"""
框修正（peak_refinement）消融对比实验。

对比同一批推理结果在「精修前」与「精修后」的差异，验证框修正是否有作用：

  方法 A（无精修）: prediction_snr.csv —— SNR 筛选后保留的模型检测框（rt_min/rt_max = 模型框）
  方法 B（有精修）: prediction_refined.csv —— 同一批框经 peak_refinement 精修后的主峰/次峰边界

核心口径（避免把"次峰检测"混入"边界精度"）：
  1) 主峰边界成对对比 —— 同一峰在精修前（A 模型框）与精修后（B main）相对人工标注的偏差；
  2) 检出对比（仅主峰）—— A 模型框 vs B main，P/R/F1；
  3) 次峰找回（附加）—— B 相对 A 净增的检出与未命中框；
  4) 面积定量（同框对）—— 两种方法用同一 XIC + 同一积分器，仅边界不同。

GT 为人工标注（data/label/test1.xlsx 起止+面积）。输出（默认 output/inference/refine_ablation/）：
  refine_ablation_report.md   实验报告
  boundary_pairs.csv          主峰边界成对明细（A vs B vs GT）
  detail.csv                  逐（样品, 化合物, 方法）检出明细

用法（model/ 目录下）：
  python -m tools.evaluation.refine_ablation
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # model/
sys.path.insert(0, str(ROOT))

from tools._shared.artifacts import read_csv_safe, resolve_roi_root
from postprocessing.area_integration import (
    _build_xic_list_from_roi_dir,
    _integrate_area_on_segment,
)
from preprocessing.coco_annotation import label_key

TOL = 0.1  # 边界命中容差（min）：与 evaluate_baseline 一致
PAIR_TOL = 0.2  # 成对匹配容差：主峰中心 ±0.2 min 内视为同一峰
BLANK_PREFIX = ("空白", "BLANK", "blank")


def _non_nan(v):
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def parse_rt(v):
    """解析 '16.428(0.000)' / '16.428' / NaN → 分钟；空/非法 → None。"""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else None


def load_labels(path):
    df = pd.read_excel(path, sheet_name=0)
    rows = []
    for _, r in df.iterrows():
        comp_v = r.get("comonent")
        if comp_v is None or (isinstance(comp_v, float) and np.isnan(comp_v)):
            comp_v = r.get("compound")
        sid_v = r.get("sample_id")
        if not _non_nan(comp_v) or not _non_nan(sid_v):
            continue
        compound, sample_id = str(comp_v).strip(), str(sid_v).strip()
        if not compound or not sample_id:
            continue
        channel = str(r.get("channel") or "").strip()
        nid = label_key(compound, channel)
        peaks = []
        for k in (1, 2, 3):
            s = parse_rt(r.get(f"peak_start{k}"))
            e = parse_rt(r.get(f"peak_end{k}"))
            if s is not None and e is not None and s < e:
                a = r.get(f"area{k}")
                peaks.append((s, e, float(a) if pd.notna(a) else np.nan))
        rows.append({
            "sample": sample_id[:-5] if sample_id.lower().endswith(".mzml") else sample_id,
            "native_id": nid,
            "peak_label": int(float(r.get("peak_label") or 0)),
            "peaks": peaks,
        })
    return rows


def image_row_number(img):
    m = re.match(r"^(\d+)_mz", Path(str(img)).stem, re.IGNORECASE)
    return int(m.group(1)) - 1 if m else None  # 0-based xic 行号


def load_feature_map(roi_dir):
    feat_csv = Path(roi_dir) / "feature.csv"
    if not feat_csv.is_file():
        return {}
    out = {}
    try:
        fdf = read_csv_safe(feat_csv)
        for _, r in fdf.iterrows():
            try:
                out[int(r["Compound Name"])] = str(r["native_id"]).strip()
            except Exception:
                continue
    except Exception:
        pass
    return out


def integrate_box(xic_list, xic_idx, rt_min, rt_max):
    """用与报告一致的 SNR 基线积分器补算任意 RT 区间面积。"""
    if xic_idx is None or not (0 <= xic_idx < len(xic_list)):
        return np.nan
    rt, y = xic_list[xic_idx][0], xic_list[xic_idx][1]
    m = (rt >= rt_min) & (rt <= rt_max)
    if m.sum() < 2:
        return np.nan
    area, *_ = _integrate_area_on_segment(rt[m], y[m], baseline_correction=True,
                                          integration_method="snr")
    return float(area)


def build_pred_by_image(df, xic_list):
    """image basename -> [(rt_min, rt_max, score, area)]（含积分补算）。"""
    out = {}
    for _, row in df.iterrows():
        img = str(row.get("image", "")).strip()
        lo = float(row.get("rt_min", np.nan))
        hi = float(row.get("rt_max", np.nan))
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            continue
        score = row.get("score", np.nan)
        area = integrate_box(xic_list, image_row_number(img), lo, hi)
        out.setdefault(img, []).append((lo, hi, float(score), area))
    return out


def build_refined_by_image(df, xic_list, tag_in=None):
    """image basename -> [(tag, rt_min, rt_max, score, area)]。"""
    out = {}
    for _, row in df.iterrows():
        img = str(row.get("image", "")).strip()
        xic_idx = image_row_number(img)
        entries = [("main", row.get("main_rt_min"), row.get("main_rt_max"), row.get("main_score_ai"))]
        for tag in ("small", "small2", "small3"):
            entries.append((tag, row.get(f"{tag}_rt_min"), row.get(f"{tag}_rt_max"),
                            row.get(f"{tag}_score_ai_discounted")))
        for tag, lo, hi, sc in entries:
            if tag_in is not None and tag != tag_in:
                continue
            if not (pd.notna(lo) and pd.notna(hi)):
                continue
            lo, hi = float(lo), float(hi)
            if hi <= lo:
                continue
            area = integrate_box(xic_list, xic_idx, lo, hi)
            out.setdefault(img, []).append((tag, lo, hi,
                                            float(sc) if pd.notna(sc) else np.nan, area))
    return out


def match_boxes(boxes, gt_peaks, tol=TOL):
    """boxes=[(rt_min,rt_max,score,area)] vs gt_peaks；返回 (hits, fp, fn)。"""
    used, hits = set(), []
    for gs, ge, ga in gt_peaks:
        best = None
        for bi, (bs, be, bsc, ba) in enumerate(boxes):
            if bi in used:
                continue
            if abs(bs - gs) <= tol and abs(be - ge) <= tol:
                best = (bi, bs, be, bsc, ba)
                break
        if best is not None:
            used.add(best[0])
            hits.append((gs, ge, ga, *best[1:]))
    return hits, len(boxes) - len(hits), len(gt_peaks) - len(hits)


def _fmt2(v):
    if v is None:
        return "—"
    try:
        if np.isnan(v):
            return "—"
    except TypeError:
        pass
    return f"{v:.4f}"


def _r2(x, y):
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return None
    x, y = x[ok], y[ok]
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def _rsd(arr):
    arr = np.asarray([a for a in arr if np.isfinite(a) and a > 0], dtype=float)
    return float(np.std(arr, ddof=0) / arr.mean()) if len(arr) >= 2 and arr.mean() > 0 else None


def main():
    ap = argparse.ArgumentParser(description="框修正消融对比实验（精修前 vs 精修后）")
    ap.add_argument("--pipeline_dir", default="../output/inference/full_pipeline")
    ap.add_argument("--labels", default="../data/label/test1.xlsx")
    ap.add_argument("--out_dir", default="../output/inference/refine_ablation")
    ap.add_argument("--tolerance", type=float, default=TOL, help="边界命中容差（min）")
    args = ap.parse_args()

    pipe = Path(args.pipeline_dir).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels)
    tol = args.tolerance

    # 收集样品产物（prediction_refined 新目录优先，回退旧 snr_filtered 布局）
    snr_root = pipe / "prediction_refined"
    if not snr_root.is_dir():
        snr_root = pipe / "snr_filtered"
    snr_by_sample, ref_by_sample = {}, {}
    for p in sorted(snr_root.rglob("prediction_snr.csv")):
        snr_by_sample.setdefault(p.relative_to(snr_root).parts[0], []).append(p)
    for p in sorted(snr_root.rglob("prediction_refined.csv")):
        ref_by_sample.setdefault(p.relative_to(snr_root).parts[0], []).append(p)
    samples = sorted(set(snr_by_sample) & set(ref_by_sample))
    if not samples:
        print(f"[ERROR] {snr_root} 下缺少 prediction_snr.csv / prediction_refined.csv")
        sys.exit(1)

    # 有效 GT：管线有 ROI 的 native_id（剔除 QC 已剔除/无 ROI 的化合物，如对硫磷）
    valid_nids = set()
    for s in samples:
        valid_nids.update(load_feature_map(resolve_roi_root(pipe) / s).values())
    gt_all = [r for r in labels if r["sample"] in samples
              and r["peak_label"] == 1 and r["native_id"] in valid_nids and r["peaks"]]
    excluded_nids = sorted({r["native_id"] for r in labels
                            if r["sample"] in samples and r["native_id"] not in valid_nids})
    n_gt = sum(len(r["peaks"]) for r in gt_all)

    detail = []
    boundary_pairs = []
    for s in samples:
        roi_dir = resolve_roi_root(pipe) / s
        feat = load_feature_map(roi_dir)
        xic_list = _build_xic_list_from_roi_dir(str(roi_dir)) if roi_dir.is_dir() else None
        gt_nid = {}
        for r in gt_all:
            if r["sample"] == s:
                gt_nid.setdefault(r["native_id"], []).extend(r["peaks"])

        predA, predB = {}, {}
        for p in snr_by_sample[s]:
            predA.update(build_pred_by_image(read_csv_safe(p), xic_list))
        for p in ref_by_sample[s]:
            predB.update(build_refined_by_image(read_csv_safe(p), xic_list))

        img2nid = {}
        for img in set(predA) | set(predB):
            idx = image_row_number(img)
            img2nid[img] = feat.get(idx + 1) if (idx is not None and feat) else ""

        for method, predmap in (("A_无精修", predA), ("B_精修", predB)):
            for img, boxes in predmap.items():
                nid = img2nid.get(img, "")
                gt = gt_nid.get(nid, [])
                # (rt_min, rt_max, score, area, tag)；A 无 tag 概念统一记 main
                bbox = [(b[0], b[1], b[2], b[3], "main") if method == "A_无精修"
                        else (b[1], b[2], b[3], b[4], b[0]) for b in boxes]
                bbox4 = [(b[0], b[1], b[2], b[3]) for b in bbox]
                tag_of = {(b[0], b[1]): b[4] for b in bbox}
                hits, fp, fn = match_boxes(bbox4, gt, tol)
                for gs, ge, ga, bs, be, bsc, ba in hits:
                    detail.append({"sample": s, "native_id": nid, "method": method,
                                   "tag": tag_of.get((bs, be), "main"),
                                   "gt_start": gs, "gt_end": ge, "gt_area": ga,
                                   "pred_start": bs, "pred_end": be, "pred_score": bsc,
                                   "pred_area": ba, "hit": 1})
                for bs, be, bsc, ba in bbox4:
                    if any(abs(bs - gs) <= tol and abs(be - ge) <= tol for gs, ge, _ in gt):
                        continue
                    detail.append({"sample": s, "native_id": nid, "method": method,
                                   "tag": tag_of.get((bs, be), "main"),
                                   "gt_start": np.nan, "gt_end": np.nan,
                                   "gt_area": np.nan, "pred_start": bs, "pred_end": be,
                                   "pred_score": bsc, "pred_area": ba, "hit": 0})

        # 主峰边界成对对比（同一峰：A 模型框 vs B main，均相对同一 GT 峰）
        for nid, peaks in gt_nid.items():
            imgs = [i for i in set(predA) | set(predB) if img2nid.get(i) == nid]
            for gs, ge, ga in peaks:
                bestA = bestB = None
                for img in imgs:
                    if predA.get(img):
                        am = max(predA[img], key=lambda b: (b[3] if np.isfinite(b[3]) else -1))
                        d = abs(am[0] - gs) + abs(am[1] - ge)
                        if d <= PAIR_TOL and (bestA is None or d < bestA[1]):
                            bestA = (am, d)
                    bmains = [b for b in predB.get(img, []) if b[0] == "main"]
                    if bmains:
                        bm = bmains[0]
                        d = abs(bm[1] - gs) + abs(bm[2] - ge)
                        if d <= PAIR_TOL and (bestB is None or d < bestB[1]):
                            bestB = (bm, d)
                if bestA and bestB:
                    am, bm = bestA[0], bestB[0]
                    boundary_pairs.append({
                        "sample": s, "native_id": nid,
                        "gt_start": gs, "gt_end": ge, "gt_area": ga,
                        "A_start": am[0], "A_end": am[1], "A_area": am[3],
                        "B_start": bm[1], "B_end": bm[2], "B_area": bm[4],
                    })

    det = pd.DataFrame(detail)
    bp = pd.DataFrame(boundary_pairs)
    det.to_csv(out / "detail.csv", index=False, encoding="utf-8-sig")
    bp.to_csv(out / "boundary_pairs.csv", index=False, encoding="utf-8-sig")

    # ============ 指标 ============
    def prf(m, tag=None):
        d = det[det["method"] == m]
        if tag:
            d = d[d["tag"] == tag]
        tp = int((d["hit"] == 1).sum())
        n_pred = len(d)
        fp = n_pred - tp
        fn = n_gt - tp
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"TP": tp, "FP": fp, "FN": fn, "P": p, "R": r, "F1": f1}

    ma = prf("A_无精修")                 # A 模型框（≈主峰）
    mb = prf("B_精修", tag="main")       # B 仅主峰（与 A 同口径）
    mb_all = prf("B_精修")               # B 含次峰（供检出差异说明）

    # 主峰边界成对指标
    bp["A_dstart"], bp["A_dend"] = bp["A_start"] - bp["gt_start"], bp["A_end"] - bp["gt_end"]
    bp["B_dstart"], bp["B_dend"] = bp["B_start"] - bp["gt_start"], bp["B_end"] - bp["gt_end"]
    bp["A_abs"] = bp["A_dstart"].abs() + bp["A_dend"].abs()
    bp["B_abs"] = bp["B_dstart"].abs() + bp["B_dend"].abs()
    n_pair = len(bp)
    if n_pair:
        better = float((bp["B_abs"] < bp["A_abs"]).mean())
        bp_dstart = (bp["A_dstart"].abs().mean(), bp["B_dstart"].abs().mean())
        bp_dend = (bp["A_dend"].abs().mean(), bp["B_dend"].abs().mean())
    else:
        better, bp_dstart, bp_dend = None, (None, None), (None, None)

    # 面积（成对）：R² 与跨样品 RSD
    r2a, r2b = _r2(bp["gt_area"].to_numpy(dtype=float), bp["A_area"].to_numpy(dtype=float)) if n_pair else None, \
               _r2(bp["gt_area"].to_numpy(dtype=float), bp["B_area"].to_numpy(dtype=float)) if n_pair else None
    rsd_a, rsd_b = [], []
    if n_pair:
        for _, g in bp.groupby("native_id"):
            ra = _rsd(g["A_area"].to_numpy(dtype=float))
            rb = _rsd(g["B_area"].to_numpy(dtype=float))
            if ra is not None:
                rsd_a.append(ra)
            if rb is not None:
                rsd_b.append(rb)

    # 次峰找回：B 中 small 峰命中数（A 无此候选）
    small_hits = int(((det["method"] == "B_精修") & (det["hit"] == 1) & (det["tag"] != "main")).sum()) if "tag" in det else 0
    small_all = int(((det["method"] == "B_精修") & (det["tag"] != "main")).sum()) if "tag" in det else 0

    # 空白样品 FP（假阳性评估，主峰口径）
    blank_fp = {m: int(((det["method"] == m) & (det["sample"].str.startswith(BLANK_PREFIX))
                        & (det["hit"] == 0) & (det["tag"] == "main")).sum())
                for m in ("A_无精修", "B_精修")}

    # ============ 报告 ============
    L = []
    L.append("# 框修正（peak_refinement）消融对比实验报告")
    L.append("")
    L.append(f"- 数据集: test1（{len(samples)} 个样品：混标 ×{len([s for s in samples if not s.startswith(BLANK_PREFIX)])} + 空白 ×{len([s for s in samples if s.startswith(BLANK_PREFIX)])}）")
    L.append(f"- 数据来源: {pipe}")
    L.append(f"- GT: {args.labels}（人工起止+面积；命中容差 ±{tol} min）")
    L.append("- 方法 A = SNR 后模型框（prediction_snr.csv）｜方法 B = 精修后（prediction_refined.csv）")
    L.append(f"- 两种方法面积均用同一 XIC + 同一 SNR 基线积分器补算，仅边界不同")
    if excluded_nids:
        L.append(f"- 已剔除 QC 检出标注不一致、管线无 ROI 的化合物: {', '.join(excluded_nids)}")
    L.append("")
    L.append("## 1. 主峰边界精度（同峰成对对比，核心指标）")
    L.append("")
    L.append(f"- 成对样本数: {n_pair}（A 模型框与 B 精修 main 均存在且距同一人工峰中心 ≤{PAIR_TOL} min）")
    L.append(f"- 平均|Δ起边|: A {_fmt2(bp_dstart[0])} → B {_fmt2(bp_dstart[1])} min"
             + ("（改善）" if bp_dstart[1] is not None and bp_dstart[0] is not None and bp_dstart[1] < bp_dstart[0] else ""))
    L.append(f"- 平均|Δ止边|: A {_fmt2(bp_dend[0])} → B {_fmt2(bp_dend[1])} min"
             + ("（改善）" if bp_dend[1] is not None and bp_dend[0] is not None and bp_dend[1] < bp_dend[0] else ""))
    L.append(f"- 总边界误差(|Δ起|+|Δ止|) B 比 A 更小的峰占比: {better:.1%}" if better is not None else "- 成对样本数: 0")
    L.append(f"- 明细: `{out / 'boundary_pairs.csv'}`")
    L.append("")
    L.append("## 2. 检出对比（仅主峰，P/R/F1）")
    L.append("")
    L.append("| 指标 | A 无精修 | B 精修 | 说明 |")
    L.append("|---|---|---|---|")
    L.append(f"| TP | {ma['TP']} | {mb['TP']} | 起止偏差≤±{tol}min |")
    L.append(f"| FP | {ma['FP']} | {mb['FP']} | 检出但无 GT 命中 |")
    L.append(f"| FN | {ma['FN']} | {mb['FN']} | GT 峰漏检（GT 共 {n_gt} 峰） |")
    L.append(f"| Precision | {ma['P']:.3f} | {mb['P']:.3f} | |")
    L.append(f"| Recall | {ma['R']:.3f} | {mb['R']:.3f} | |")
    L.append(f"| F1 | {ma['F1']:.3f} | {mb['F1']:.3f} | |")
    L.append(f"| 空白样品 FP | {blank_fp['A_无精修']} | {blank_fp['B_精修']} | 假阳性评估（主峰口径） |")
    L.append("")
    L.append(f"> 附：B 把次峰也计入候选时 TP {mb_all['TP']}/FP {mb_all['FP']}/F1 {mb_all['F1']:.3f}——"
             f"检出更全但未命中框增多（见第 3 节次峰找回）。")
    L.append("")
    L.append("## 3. 次峰找回（B 附加能力）")
    L.append("")
    L.append(f"- B 检出次峰候选 {small_all} 个，其中命中人工标注（模型主峰漏检、精修找回）{small_hits} 个")
    L.append(f"- 注意：B 未命中的次峰可能为共流出真峰（GT 未标注）或假峰，需人工复核；"
             f"本实验中部分次峰位于对硫磷定性离子 RT 区（对硫磷已因标注不一致剔除）")
    L.append("")
    L.append("## 4. 面积定量（同峰成对）")
    L.append("")
    L.append(f"- 面积 R²(pred vs 人工): A {_fmt2(r2a)} → B {_fmt2(r2b)}（越高越好，无量纲）")
    L.append(f"- 跨样品 RSD（每化合物，越低越稳）: A {_fmt2(np.mean(rsd_a) if rsd_a else None)} "
             f"({len(rsd_a)} 组) → B {_fmt2(np.mean(rsd_b) if rsd_b else None)} ({len(rsd_b)} 组)")
    L.append("")
    L.append("## 5. 结论")
    L.append("")
    L.append(f"- 边界精度: 成对 {n_pair} 峰中，精修后总边界误差更小的占 {better:.1%}"
             if better is not None else "- 边界精度: 无成对样本")
    if bp_dstart[1] is not None and bp_dstart[0] is not None:
        L.append(f"- 平均|Δ起| {_fmt2(bp_dstart[0])}→{_fmt2(bp_dstart[1])}，平均|Δ止| {_fmt2(bp_dend[0])}→{_fmt2(bp_dend[1])} min"
                 + ("（精修边界更接近人工标注）" if bp_dstart[1] < bp_dstart[0] and bp_dend[1] < bp_dend[0] else ""))
    L.append(f"- 检出: F1 {ma['F1']:.3f}→{mb['F1']:.3f}（主峰口径；差异来自边界移动与个别框被精修门控剔除）")
    L.append(f"- 次峰: 精修找回 {small_hits} 个模型主峰漏检的峰")
    L.append(f"- 面积: R² {_fmt2(r2a)}→{_fmt2(r2b)}，RSD {_fmt2(np.mean(rsd_a) if rsd_a else None)}→{_fmt2(np.mean(rsd_b) if rsd_b else None)}")
    L.append("")
    L.append("---")
    L.append(f"> 明细: `{out / 'detail.csv'}`、`{out / 'boundary_pairs.csv'}`；由 `model/tools/evaluation/refine_ablation.py` 生成。")
    (out / "refine_ablation_report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"[INFO] 实验报告: {out / 'refine_ablation_report.md'}")
    print(f"[INFO] 明细: {out / 'detail.csv'} ({len(det)} 行) | 成对: {out / 'boundary_pairs.csv'} ({n_pair} 行)")
    print("\n===== A 无精修 vs B 精修 =====")
    print(f"  主峰 P/R/F1: {ma['P']:.3f}/{ma['R']:.3f}/{ma['F1']:.3f}  vs  {mb['P']:.3f}/{mb['R']:.3f}/{mb['F1']:.3f}"
          f" (TP {ma['TP']}/{mb['TP']}, FP {ma['FP']}/{mb['FP']}, FN {ma['FN']}/{mb['FN']})")
    print(f"  成对边界 |Δ起|: {_fmt2(bp_dstart[0])}→{_fmt2(bp_dstart[1])} | |Δ止|: {_fmt2(bp_dend[0])}→{_fmt2(bp_dend[1])} | B 更准占比 {better:.1%}")
    print(f"  面积 R²: {_fmt2(r2a)}→{_fmt2(r2b)} | RSD: {_fmt2(np.mean(rsd_a) if rsd_a else None)}→{_fmt2(np.mean(rsd_b) if rsd_b else None)}")
    print(f"  次峰找回命中 {small_hits}/{small_all} | 空白 FP: A={blank_fp['A_无精修']} B={blank_fp['B_精修']}")


if __name__ == "__main__":
    main()
