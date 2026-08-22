# -*- coding: utf-8 -*-
"""
基线一键精度评测：跑推理管线 → 对齐人工标注 → 输出峰检测 + 定量指标。

指标：
  - 峰检测：Precision / Recall / F1
    匹配规则（本项目协议）：预测 RT 区间与标注 RT 区间的 tIoU（交集时长/并集时长）> 阈值
    （默认 0.95）判为命中；同一标注至多匹配一个预测（按置信度从高到低贪婪匹配），
    未匹配预测计 FP，未匹配标注计 FN。
  - RT 偏差：TP 对中 |pred_rt_min - gt_start| 与 |pred_rt_max - gt_end|（分钟，均值/中位数）
  - 面积 R²：TP 对 预测面积 vs 人工面积（xlsx area 列）散点线性拟合（复用 linear_fit_r2）
  - RSD：同一化合物通道跨进样的预测面积重复性（两点 RSD = std(ddof=0)/mean）

对齐：prediction.csv 的 image 名前缀 `N_mz...` 对应 feature.csv 第 N 行（1-based），
native_id「化合物名-1/-2」对齐标注 xlsx 行（与 coco_annotation 同一套规则）。

用法（model/ 目录下，系统终端执行；推理涉及画图需绕沙箱）：
  python -m tools.evaluation.evaluate_baseline \
      --mzmls ../data/test/mzml/20260715_shiyaoyuan_test_1.mzML \
              ../data/test/mzml/20260715_shiyaoyuan_test_2.mzML \
      --labels ../data/label/20260715_shiyaoyuan_test.xlsx \
      --model checkpoint/quanformer.pth \
      --output_dir ../output/evaluation/quanformer

  # 复用已跑好的 prediction.csv（跳过推理，纯计算指标）：
  python -m tools.evaluation.evaluate_baseline --run_inference 0 \
      --labels ../data/label/20260715_shiyaoyuan_test.xlsx \
      --output_dir ../output/evaluation/quanformer \
      --prediction_csvs 20260715_shiyaoyuan_test_1=<pred.csv> 20260715_shiyaoyuan_test_2=<pred.csv> \
      --feature_csvs   20260715_shiyaoyuan_test_1=<feature.csv> 20260715_shiyaoyuan_test_2=<feature.csv>

输出（output_dir 下）：
  evaluation_report.json  指标汇总（含协议参数：tIoU 阈值、score 阈值、模型、数据）
  match_details.csv       逐条匹配明细（TP/FP/FN + RT 区间 + score）
  area_pairs.csv          TP 对的 预测面积 vs 人工面积
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from postprocessing.evaluation.standard_curves import linear_fit_r2
from preprocessing.coco_annotation import (
    group_labels_by_sample,
    label_key,
    map_samples_to_mzmls,
    parse_labels_xlsx,
    parse_rt_field,
)

DEFAULT_TOL = 0.1        # 起止偏差容差（min）：检测口径命中判据
DEFAULT_QUANT_TOL = 0.2  # 起止偏差容差（min）：定量口径宽松配对
DEFAULT_SCORE = 0.90  # 预测框纳入评测的最低置信度（同时天然排除无检出的 score=0 占位行）
MODEL_ROOT = Path(__file__).resolve().parents[2]  # model/


def _parse_area(v):
    """xlsx area 字段（字符串）→ float；空/非法 → NaN。"""
    s = str(v).strip() if v is not None else ""
    if not s or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_gt_peaks(rec):
    """标注行 → [(start, end, area), ...]。

    多峰格式 peak_start1-3/peak_end1-3（真实标注文件，与 coco_annotation 同一规则），
    回退旧单数 peak_start/peak_end；面积按峰对应取 area1-3（回退 area）。
    """
    gt_peaks = []
    for k in (1, 2, 3):
        s = parse_rt_field(rec.get("peak_start%d" % k))
        e = parse_rt_field(rec.get("peak_end%d" % k))
        if s is not None and e is not None:
            gt_peaks.append((min(s, e), max(s, e), _parse_area(rec.get("area%d" % k))))
    if not gt_peaks:
        s = parse_rt_field(rec.get("peak_start"))
        e = parse_rt_field(rec.get("peak_end"))
        if s is not None and e is not None:
            gt_peaks.append((min(s, e), max(s, e), _parse_area(rec.get("area"))))
    return gt_peaks


# ---------- 核心指标 ----------

def match_image(pred_rows, gt_peaks, tol, loose_tol=0.2):
    """单张 ROI 图内两轮贪婪匹配（统一起止偏差口径）。

    pred_rows: 该图的预测行列表（已按 score 降序，元素为 dict，含 rt_min/rt_max/score/area）。
    gt_peaks: [(start, end), ...]（本场景每图至多 1 个标注峰）。

    严格轮（检测口径）：预测起止与人工起止偏差均 <= tol（分钟）判 TP；
    多候选时取总偏差最小者。未中预测进 fp_rows，剩余 gt 为 FN。
    宽松轮（定量口径，loose_tol>0 时）：在未中预测与剩余 gt 之间按起止偏差均
    <= loose_tol 且总偏差最小再配对，产出 loose_pairs —— 仅用于 面积R²/RSD/RT偏差
    等定量指标，不影响 P/R/F1 计数（即宽松配对在检测口径下仍分别计为 FP 与 FN）。

    返回 (tp_pairs, fp_rows, fn_peaks, loose_pairs)。
    """
    remaining = list(gt_peaks)
    tp_pairs, fp_rows = [], []
    for pr in pred_rows:
        lo, hi = pr.get("rt_min"), pr.get("rt_max")
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            fp_rows.append(pr)
            continue
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            fp_rows.append(pr)
            continue
        best_j, best_key = -1, None
        for j, g in enumerate(remaining):
            # +1e-9 容差吸收浮点误差（边界恰好相等时应判命中）
            if abs(lo - g[0]) <= tol + 1e-9 and abs(hi - g[1]) <= tol + 1e-9:
                key = -(abs(lo - g[0]) + abs(hi - g[1]))
                if best_key is None or key > best_key:
                    best_j, best_key = j, key
        if best_j >= 0:
            tp_pairs.append((pr, remaining.pop(best_j)))
        else:
            fp_rows.append(pr)

    loose_pairs = []
    if loose_tol > 0 and remaining and fp_rows:
        used = set()
        for g in list(remaining):
            best_i, best_key = -1, None
            for i, pr in enumerate(fp_rows):
                if i in used:
                    continue
                try:
                    lo, hi = float(pr["rt_min"]), float(pr["rt_max"])
                except (TypeError, ValueError, KeyError):
                    continue
                if abs(lo - g[0]) <= loose_tol + 1e-9 and abs(hi - g[1]) <= loose_tol + 1e-9:
                    key = -(abs(lo - g[0]) + abs(hi - g[1]))
                    if best_key is None or key > best_key:
                        best_i, best_key = i, key
            if best_i >= 0:
                loose_pairs.append((fp_rows[best_i], g))
                used.add(best_i)
                remaining.remove(g)
    return tp_pairs, fp_rows, remaining, loose_pairs


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ---------- 推理 ----------

def run_inference_for_mzml(mzml, model, out_dir, threshold, smooth_sigma,
                           labels=None, qc_label_rt_tol=None):
    """subprocess 调 inference.cli --mode pipeline（与手工执行完全一致），返回 (pred_csv, feature_csv)。

    labels: 人工标注 xlsx。传入时透传 --labels → ROI 由标注驱动（B 范式），
    与训练数据生成（coco_annotation）同一路径：仅标注命中通道生成 ROI、窗口中心=标注 rt。
    qc_label_rt_tol: 标注 RT 一致性 QC 阈值（min），缺省用 CLI 默认（1.0，与训练一致）。
    """
    cmd = [
        sys.executable, "-m", "inference.cli",
        "--mode", "pipeline",
        "--mzml", str(mzml),
        "--model", str(model),
        "--output_dir", str(out_dir),
        "--threshold", str(threshold),
        "--smooth_sigma", str(smooth_sigma),
    ]
    if labels:
        cmd += ["--labels", str(labels)]
        if qc_label_rt_tol is not None:
            cmd += ["--qc_label_rt_tol", str(qc_label_rt_tol)]
    print("[INFO] 推理:", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(MODEL_ROOT))
    if ret.returncode != 0:
        raise RuntimeError(f"推理失败 (exit={ret.returncode}): {mzml}")
    stem = Path(mzml).stem
    pred_csv = out_dir / "batch_predictions" / stem / "prediction.csv"
    feat_csv = out_dir / "xic-roi-batch" / stem / "feature.csv"
    if not pred_csv.is_file():
        raise FileNotFoundError(f"未找到推理输出: {pred_csv}")
    if not feat_csv.is_file():
        raise FileNotFoundError(f"未找到 XIC 输出: {feat_csv}")
    return pred_csv, feat_csv


def _load_pred_rows(pred_path, min_score):
    """读取预测框行列表 [{image, rt_min, rt_max, score, area}]，兼容两种格式：

    1. 窄表（newtest 原始输出 prediction.csv）：一行一个框，含 image/rt_min/rt_max/score/area；
    2. 宽表（post_newtest 框修正输出 prediction_refined.csv）：一行一个 ROI 图，
       主峰 main_rt_min/main_rt_max/main_score_ai + 次峰 small*/small2*/small3*_rt_*，
       展开为多个框行；无面积列（area 置 NaN，定量面积指标对该输入不可用）。
    """
    df = pd.read_csv(pred_path)
    cols = set(df.columns)
    if "rt_min" in cols:
        out = df[pd.to_numeric(df.get("score"), errors="coerce").fillna(0) >= min_score]
        return [dict(r) for _, r in out.iterrows()]
    if "main_rt_min" in cols:
        rows = []
        for _, r in df.iterrows():
            img = str(r["image"]).strip()
            for prefix, score_col in (("main", "main_score_ai"),
                                      ("small", "small_score_ai_discounted"),
                                      ("small2", "small2_score_ai_discounted"),
                                      ("small3", "small3_score_ai_discounted")):
                rt_min, rt_max = r.get(f"{prefix}_rt_min"), r.get(f"{prefix}_rt_max")
                if pd.isna(rt_min) or pd.isna(rt_max):
                    continue
                sc = r.get(score_col)
                if sc is None or (isinstance(sc, float) and np.isnan(sc)):
                    continue
                if float(sc) < min_score:
                    continue
                rows.append({"image": img, "rt_min": float(rt_min), "rt_max": float(rt_max),
                             "score": float(sc), "area": float("nan")})
        return rows
    raise ValueError(f"{pred_path}: 缺 image/rt_min（窄表）或 main_rt_min（宽表）列")


def _build_pred_by_img(pred_rows, min_score):
    pred_by_img = {}
    for r in pred_rows:
        pred_by_img.setdefault(str(r["image"]).strip(), []).append(dict(r))
    for v in pred_by_img.values():
        v.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return pred_by_img


# ---------- 评测主流程 ----------

def evaluate(pred_feat_map, labels_path, tol, min_score, quant_tol=0.2):
    """pred_feat_map: {stem: {"pred": path, "feat": path}}。返回 (metrics, details_df, area_df)。

    tol: 检测口径命中容差（min）——预测起止与人工起止偏差均 <= tol 判 TP。
    quant_tol: 定量口径（面积R²/RSD/RT偏差）宽松配对容差（min）——未达严格 TP 但
    起止偏差均 <= quant_tol 的预测-标注对参与定量指标；检测口径（P/R/F1）不受影响。
    """
    labels = parse_labels_xlsx(labels_path)
    sample_order, groups = group_labels_by_sample(labels)
    stem2sample = map_samples_to_mzmls(list(pred_feat_map), sample_order, None)

    tp = fp = fn = 0
    rt_devs_start, rt_devs_end = [], []          # 定量口径（严格 TP + 宽松配对）
    area_rows = []   # {stem, native_id, pred_area, manual_area, match}
    details = []

    for stem, paths in pred_feat_map.items():
        pred_rows = _load_pred_rows(paths["pred"], min_score)
        pred_by_img = _build_pred_by_img(pred_rows, min_score)

        # 标注：native_id → xlsx 行（与 coco_annotation 相同对齐规则）
        by_key = {}
        for rec in groups[stem2sample[stem]]:
            k = label_key(rec.get("compound"), rec.get("channel"))
            if k:
                by_key.setdefault(k, rec)

        feat = pd.read_csv(paths["feat"])
        if "native_id" not in feat.columns or "Compound Name" not in feat.columns:
            raise ValueError(f"{paths['feat']}: 缺 native_id/Compound Name 列")

        for i, frow in feat.iterrows():
            n = i + 1  # image 前缀 N_mz...（1-based，与 roi_safe_name_base 一致）
            native_id = str(frow["native_id"]).strip()
            img_name = next((im for im in pred_by_img if im.startswith(f"{n}_mz")), None)
            rows = pred_by_img.get(img_name, []) if img_name else []

            rec = by_key.get(native_id)
            gt_peaks = _parse_gt_peaks(rec) if rec is not None else []

            pairs, fps, fns, loose = match_image(rows, gt_peaks, tol, loose_tol=quant_tol)
            tp += len(pairs)
            fp += len(fps)
            fn += len(fns) + len(loose)  # 宽松配对不改变检测口径：该 gt 仍计 FN

            def _collect_quant(pr, g, match_label):
                rt_devs_start.append(abs(float(pr["rt_min"]) - g[0]))
                rt_devs_end.append(abs(float(pr["rt_max"]) - g[1]))
                if rec is not None:
                    gt_area = float(g[2])  # 多峰格式下按峰对应的 area1-3
                    pred_area = float(pr.get("area") or 0.0)
                    if np.isfinite(gt_area) and gt_area > 0 and pred_area > 0:
                        area_rows.append({"stem": stem, "native_id": native_id,
                                          "pred_area": pred_area, "manual_area": gt_area,
                                          "match": match_label})

            for pr, g in pairs:
                _collect_quant(pr, g, "strict")
                details.append({"stem": stem, "native_id": native_id, "result": "TP",
                                "gt_start": g[0], "gt_end": g[1],
                                "pred_start": pr["rt_min"], "pred_end": pr["rt_max"],
                                "score": pr.get("score"), "quant": 1})
            loose_by_pred = {id(pr): g for pr, g in loose}
            for pr in fps:
                g = loose_by_pred.get(id(pr))
                if g is not None:
                    _collect_quant(pr, g, "loose")
                details.append({"stem": stem, "native_id": native_id, "result": "FP",
                                "gt_start": gt_peaks[0][0] if gt_peaks else None,
                                "gt_end": gt_peaks[0][1] if gt_peaks else None,
                                "pred_start": pr.get("rt_min"), "pred_end": pr.get("rt_max"),
                                "score": pr.get("score"), "quant": 1 if g is not None else 0})
            for g in fns:
                details.append({"stem": stem, "native_id": native_id, "result": "FN",
                                "gt_start": g[0], "gt_end": g[1],
                                "pred_start": None, "pred_end": None, "score": None,
                                "quant": 0})

    p, r, f1 = prf(tp, fp, fn)
    area_df = pd.DataFrame(area_rows)
    if len(area_df) >= 2:
        _, _, area_r2 = linear_fit_r2(area_df["manual_area"].tolist(),
                                      area_df["pred_area"].tolist())
    else:
        area_r2 = float("nan")

    # RSD：同一 native_id 跨进样的预测面积
    rsds = []
    if not area_df.empty:
        for _, g in area_df.groupby("native_id"):
            arr = g["pred_area"].to_numpy(dtype=float)
            if len(arr) >= 2 and arr.mean() > 0:
                rsds.append(float(np.std(arr, ddof=0) / arr.mean()))

    def _mean(x):
        return float(np.mean(x)) if x else None

    def _med(x):
        return float(np.median(x)) if x else None

    metrics = {
        "tolerance": tol,
        "quant_tolerance": quant_tol,
        "min_score": min_score,
        "TP": tp, "FP": fp, "FN": fn,
        "precision": p, "recall": r, "f1": f1,
        "rt_start_dev_mean_min": _mean(rt_devs_start),
        "rt_start_dev_median_min": _med(rt_devs_start),
        "rt_end_dev_mean_min": _mean(rt_devs_end),
        "rt_end_dev_median_min": _med(rt_devs_end),
        "area_r2_pred_vs_manual": float(area_r2) if np.isfinite(area_r2) else None,
        "n_area_pairs": len(area_df),
        "n_area_pairs_strict": int((area_df.get("match") == "strict").sum()) if not area_df.empty else 0,
        "n_area_pairs_loose": int((area_df.get("match") == "loose").sum()) if not area_df.empty else 0,
        "rsd_mean": _mean(rsds),
        "rsd_median": _med(rsds),
        "n_rsd_compounds": len(rsds),
    }
    return metrics, pd.DataFrame(details), area_df


def _fmt(v, fmt="{:.4f}"):
    return fmt.format(v) if v is not None else "N/A"


def main():
    ap = argparse.ArgumentParser(description="基线一键精度评测（P/R/F1 + RT偏差 + 面积R2 + RSD）")
    ap.add_argument("--mzmls", nargs="*", default=[], help="mzML 列表（--run_inference 1 时推理）")
    ap.add_argument("--labels", default=None, help="人工标注 xlsx；也可由 --config 提供")
    ap.add_argument("--model", default="checkpoint/quanformer.pth", help="模型权重")
    ap.add_argument("--output_dir", default=None,
                    help="评测输出目录；也可由 --config 提供")
    ap.add_argument("--run_inference", type=int, default=1, choices=(0, 1),
                    help="1=先跑推理管线；0=复用已有 prediction.csv")
    ap.add_argument("--prediction_csvs", nargs="*", default=[],
                    help="run_inference=0 时：stem=path/to/prediction.csv")
    ap.add_argument("--feature_csvs", nargs="*", default=[],
                    help="run_inference=0 时：stem=path/to/feature.csv")
    ap.add_argument("--threshold", type=float, default=DEFAULT_SCORE, help="预测置信度阈值")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOL,
                    help="起止偏差容差（min）：检测口径命中判据（预测起止 vs 人工起止）")
    ap.add_argument("--quant_tolerance", type=float, default=DEFAULT_QUANT_TOL,
                    help="起止偏差容差（min）：定量口径宽松配对（面积R2/RSD/RT偏差），不影响 P/R/F1")
    ap.add_argument("--smooth_sigma", type=float, default=0.8, help="推理平滑参数（与管线一致）")
    ap.add_argument("--qc_label_rt_tol", type=float, default=None,
                    help="run_inference=1 时透传 inference.cli 的标注 RT 一致性 QC 阈值（min）；"
                         "缺省用 CLI 默认 1.0（与训练数据生成一致）；0=关闭")
    ap.add_argument("--config", type=str, default=None,
                    help="JSON 配置文件路径（作为默认参数，CLI 可覆盖；参数外置，仿 train.py --config）")

    # 参数配置外置：手动提取 --config（避免 parse_known_args 触发 required 校验），
    # 加载 JSON 作为默认值，CLI 参数仍可覆盖
    _cfg_path = None
    for _i, _tok in enumerate(sys.argv[1:]):
        if _tok == "--config" and _i + 1 < len(sys.argv[1:]):
            _cfg_path = sys.argv[2 + _i]
            break
        if _tok.startswith("--config="):
            _cfg_path = _tok.split("=", 1)[1]
            break
    if _cfg_path:
        with open(_cfg_path, encoding="utf-8") as _f:
            _cfg = json.load(_f)
        _cfg.pop("config", None)
        _cfg = {_k: _v for _k, _v in _cfg.items() if not _k.startswith("_")}  # 过滤 _comment_* 注释键
        ap.set_defaults(**_cfg)
        print(f"[INFO] 已加载评估配置: {_cfg_path}")
    args = ap.parse_args()
    if not args.labels:
        ap.error("--labels 必填（命令行或 --config 提供）")
    if not args.output_dir:
        ap.error("--output_dir 必填（命令行或 --config 提供）")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_feat_map = {}
    if args.run_inference:
        if not args.mzmls:
            ap.error("--run_inference 1 需要 --mzmls")
        for mzml in args.mzmls:
            stem = Path(mzml).stem
            pred_csv, feat_csv = run_inference_for_mzml(
                mzml, args.model, out_dir / "_pipeline" / stem,
                args.threshold, args.smooth_sigma,
                labels=args.labels, qc_label_rt_tol=args.qc_label_rt_tol)
            pred_feat_map[stem] = {"pred": str(pred_csv), "feat": str(feat_csv)}
    else:
        feats = dict(item.split("=", 1) for item in args.feature_csvs)
        for item in args.prediction_csvs:
            stem, path = item.split("=", 1)
            if stem not in feats:
                ap.error(f"--feature_csvs 缺少 {stem}（image↔native_id 对齐必需）")
            pred_feat_map[stem] = {"pred": path, "feat": feats[stem]}

    metrics, details_df, area_df = evaluate(
        pred_feat_map, args.labels, args.tolerance, args.threshold,
        quant_tol=args.quant_tolerance)

    report = {
        "model": str(args.model),
        "mzmls": [str(m) for m in args.mzmls],
        "labels": str(args.labels),
        "tolerance": args.tolerance,
        "quant_tolerance": args.quant_tolerance,
        "metrics": metrics,
    }
    with open(out_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    details_df.to_csv(out_dir / "match_details.csv", index=False, encoding="utf-8-sig")
    if not area_df.empty:
        area_df.to_csv(out_dir / "area_pairs.csv", index=False, encoding="utf-8-sig")

    m = metrics
    n_gt_peaks = m['TP'] + m['FN']
    _rsd = f"{m['rsd_median'] * 100:.2f}%" if m['rsd_median'] is not None else "N/A"
    bar = 64
    print()
    print("=" * bar)
    print("评估报告")
    print("=" * bar)
    print(f"样品   : {len(pred_feat_map)} 个 | 标注 {Path(args.labels).name} | score>={args.threshold:.2f}")
    print("-" * bar)
    print(f"检测   | 命中判据: 预测起止 vs 人工起止 偏差均 <= ±{args.tolerance:.2f} min")
    print(f"       | TP {m['TP']} | FP {m['FP']} | FN {m['FN']} (标注峰共 {n_gt_peaks})")
    print(f"       | P {_fmt(m['precision'])} | R {_fmt(m['recall'])} | F1 {_fmt(m['f1'])}")
    print("-" * bar)
    print(f"定量   | 配对容差 ±{args.quant_tolerance:.2f} min | {m['n_area_pairs']} 对"
          f" (严格 {m['n_area_pairs_strict']} + 宽松 {m['n_area_pairs_loose']})")
    print(f"       | 面积 R2     {_fmt(m['area_r2_pred_vs_manual'], '{:.5f}')} (预测 vs 人工)")
    print(f"       | RT 偏差     起 {_fmt(m['rt_start_dev_median_min'])} / "
          f"止 {_fmt(m['rt_end_dev_median_min'])} min (中位)")
    print(f"       | RSD         {_rsd} (中位, n={m['n_rsd_compounds']} 化合物)")
    print("-" * bar)
    print(f"产物   : {out_dir}/evaluation_report.json | match_details.csv"
          + (" | area_pairs.csv" if not area_df.empty else ""))
    print("=" * bar)


if __name__ == "__main__":
    main()
