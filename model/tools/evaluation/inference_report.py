# -*- coding: utf-8 -*-
"""
推理报告生成器：汇总 pipeline 输出中各样品的精修结果（prediction_refined.csv），
为精修框补算峰面积（复用 area_integration.integrate_from_refined_dataframe），
并输出（Step 7c 新命名，旧固定名接口已删除）：

  <output_dir>/all.csv                    合并明细（每样品 × 每 ROI 一行）
  <output_dir>/inference_report_<实验名>.md  可读推理报告（运行信息/样本摘要/化合物×样品面积矩阵/QC 汇总）
  <snr_root>/<样品>/prediction_refined_with_area.csv   各样品补面积后的明细

用法（在 model/ 目录下）：
  python -m tools.evaluation.inference_report --output_dir ../output/inference/pipeline_demo --exp_name demo
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # model/
sys.path.insert(0, str(ROOT))

from tools._shared.artifacts import read_csv_safe
from postprocessing.area_integration import (
    _build_xic_list_from_roi_dir,
    integrate_from_refined_dataframe,
)

# native_id 形如 "6-涕灭威-1"（序号-化合物名-离子通道 1/2）
NATIVE_ID_RE = re.compile(r"^(\d+)-(.+)-([12])$")
# image 形如 "1_mz208.0000_q3116.0000_6-涕灭威-1_snr1185.64.jpeg"
IMAGE_COMPOUND_RE = re.compile(r"_(\d+)-(.+?)-([12])(?:_snr[\d.e+-]*)?\.(?:jpe?g|png)$", re.I)


def image_row_number(img):
    """从 ROI 文件名前缀解析 1-based feature 行号：N_mz* -> N。"""
    m = re.match(r"^(\d+)_mz", Path(str(img)).stem, re.IGNORECASE)
    return int(m.group(1)) if m else None

MAIN_COLS = ["image", "mz", "q3", "main_rt_min", "main_rt_max", "main_rt_peak",
             "main_height", "main_score_ai", "main_snr", "main_area",
             "main_intensity_max", "main_point_counts"]
SMALL_TAGS = [("small", "small_area"), ("small2", "small2_area"), ("small3", "small3_area")]


def parse_compound_from_nid(nid):
    m = NATIVE_ID_RE.match(str(nid or "").strip())
    if m:
        return {"compound": m.group(2), "ion": int(m.group(3))}
    return {"compound": str(nid or "").strip(), "ion": None}


def parse_compound_from_image(img):
    m = IMAGE_COMPOUND_RE.search(str(img or "").replace("\\", "/"))
    if m:
        return {"compound": m.group(2), "ion": int(m.group(3))}
    return None


def load_feature_map(roi_dir):
    """feature.csv: 'Compound Name'(1-based) -> native_id。"""
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


def collect_refined(snr_root):
    """snr_root 下收集 样品 -> [refined_csv,...]（支持 <样品>/SNR_box_*/ 与 <样品>/ 布局）。"""
    out = {}
    if not Path(snr_root).is_dir():
        return out
    for p in sorted(Path(snr_root).rglob("prediction_refined.csv")):
        rel = p.relative_to(snr_root).parts
        sample = rel[0] if len(rel) >= 2 else p.parent.name
        out.setdefault(sample, []).append(p)
    return out


def small_count(row):
    """次峰数：仅统计存在实际 RT 区间的次峰（small/small2/small3）。"""
    n = 0
    for tag, _ in SMALL_TAGS:
        if pd.notna(row.get(f"{tag}_rt_min")) and pd.notna(row.get(f"{tag}_rt_max")):
            n += 1
    return n


def build_rows(refined_by_sample, roi_root, do_integrate, method):
    """逐样品积分并构造合并明细行。"""
    all_rows = []
    per_sample = {}
    for sample in sorted(refined_by_sample):
        feat_map = load_feature_map(roi_root / sample)
        xic_list = None
        if (roi_root / sample).is_dir():
            xic_list = _build_xic_list_from_roi_dir(str(roi_root / sample))
        n_rows = 0
        n_area = 0
        for refined_csv in refined_by_sample[sample]:
            df = read_csv_safe(refined_csv)
            if do_integrate and xic_list is not None:
                df = integrate_from_refined_dataframe(
                    df, xic_list, baseline_correction=True,
                    integration_method=method)
                try:
                    df.to_csv(refined_csv.with_name("prediction_refined_with_area.csv"),
                              index=False, encoding="utf-8-sig")
                except Exception as e:
                    print(f"[WARN] 面积表写入失败 {refined_csv}: {e}")
            for _, row in df.iterrows():
                img = str(row.get("image", "")).strip()
                n = image_row_number(img)  # 1-based feature 行号（文件名权威，勿用 compound_name 列）
                nid = feat_map.get(n) if (n is not None and feat_map) else None
                comp = parse_compound_from_nid(nid) if nid else parse_compound_from_image(img)
                rec = {"sample": sample, "image": img, "native_id": nid or ""}
                rec["compound"] = (comp or {}).get("compound", "")
                rec["ion"] = (comp or {}).get("ion")
                for c in MAIN_COLS:
                    rec[c] = row.get(c, np.nan) if c != "image" else img
                for tag, acol in SMALL_TAGS:
                    rec[acol] = row.get(acol, 0.0)
                rec["small_count"] = small_count(row)
                rec["has_secondary_gate"] = int(bool(row.get("has_secondary_gate")))
                rec["gate_ok_for_adjustment"] = int(bool(row.get("gate_ok_for_adjustment")))
                rec["lr_repredict_applied"] = int(bool(row.get("lr_repredict_applied")))
                rec["detected"] = bool(np.isfinite(rec["main_area"]) and rec["main_area"] > 0)
                all_rows.append(rec)
                n_rows += 1
                if rec["detected"]:
                    n_area += 1
        per_sample[sample] = {
            "n_roi": len(feat_map),
            "n_refined": n_rows,
            "n_detected": n_area,
            "n_with_small": 0,
            "mean_score": None, "mean_snr": None, "mean_area": None,
        }
        # 二次统计（用合并后的行）
    if all_rows:
        adf = pd.DataFrame(all_rows)
        for sample, g in adf.groupby("sample"):
            ps = per_sample.get(sample, {})
            ps["n_with_small"] = int((g["small_count"] > 0).sum())
            ps["mean_score"] = float(np.nanmean(g["main_score_ai"])) if g["main_score_ai"].notna().any() else None
            ps["mean_snr"] = float(np.nanmean(g["main_snr"])) if g["main_snr"].notna().any() else None
            ps["mean_area"] = float(np.nanmean(g.loc[g["detected"], "main_area"])) if (g["detected"]).any() else None
    return all_rows, per_sample


def load_timing(out_root):
    tl = out_root / "pipeline_timing_runs.jsonl"
    if tl.is_file():
        try:
            with open(tl, encoding="utf-8") as f:
                first = json.loads(f.readline())
            return first.get("timestamp"), first.get("mode"), first.get("n_samples"), first.get("total_seconds")
        except Exception:
            pass
    return None, None, None, None


def load_timing_record(out_root):
    """读取 pipeline_timing_runs.jsonl 的最后一条记录（= 本次运行的分阶段耗时）。"""
    tl = Path(out_root) / "pipeline_timing_runs.jsonl"
    if not tl.is_file():
        return None
    try:
        last = None
        with open(tl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last) if last else None
    except Exception:
        return None


def render_timing_section(record):
    """本次运行各模块耗时表（数据源：pipeline 计时记录）。"""
    L = []
    L.append("## 5. 运行用时（各模块）")
    L.append("")
    stage_seconds = (record or {}).get("stage_seconds") or {}
    total = float((record or {}).get("total_seconds") or 0.0)
    if not stage_seconds:
        L.append("（本次运行未采集分模块计时；pipeline 模式默认采集，`--no_timing` 只影响日志落盘）")
        L.append("")
        return "\n".join(L)
    if total <= 0:
        total = sum(float(v) for v in stage_seconds.values())
    L.append(f"- 本次总耗时: {total:.2f} s | 记录时间: {(record or {}).get('timestamp', '未知')}")
    L.append("")
    L.append("| 模块 | 耗时 (s) | 占比 |")
    L.append("|---|---|---|")
    for name, sec in stage_seconds.items():
        sec = float(sec)
        pct = (100.0 * sec / total) if total > 0 else 0.0
        L.append(f"| {name} | {sec:.2f} | {pct:.1f}% |")
    L.append("")
    return "\n".join(L)


def qc_summary_text(out_root):
    qc_dir = Path(out_root).parent.parent / "QC" / out_root.name
    if not qc_dir.is_dir():
        qc_dir = Path(out_root) / "QC"
    p = qc_dir / "qc_summary.md"
    if p.is_file():
        return p, p.read_text(encoding="utf-8")
    return None, None


def render_matrix(adf, samples):
    """化合物 × 样品：主峰面积（—=未检出/无该 ROI）。"""
    if adf.empty:
        return "（无数据）"
    compounds = []
    for c in sorted(adf["compound"].dropna().unique()):
        g = adf[adf["compound"] == c]
        n_det = int(g["detected"].sum())
        compounds.append((c, n_det, g))
    lines = ["| 化合物 | 检出/样品 |" + "".join(f" {s} |" for s in samples),
             "|---|---|" + "|".join(["---"] * len(samples)) + "|"]
    for c, n_det, g in compounds:
        cells = []
        for s in samples:
            sub = g[(g["sample"] == s) & (g["detected"])]
            cells.append(f"{sub['main_area'].max():.1f}" if len(sub) else "—")
        lines.append(f"| {c} | {n_det}/{len(samples)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


PIPELINE_STAGES = [
    ("1 ROI 生成(xic_extraction)", "标注驱动按 ±1min 窗切 EIC 生成 400x300 ROI 图（QC：低强度/少点数通道剔除）",
     "xic_roi/<样品>/*.jpeg"),
    ("2 模型预测(predictor)", "DETR 检测峰框（score 阈值过滤）+ linear 积分",
     "predictions_model/<样品>/model_prediction_<样品>.csv"),
    ("3 SNR 筛选(snr_filter)", "估计框外噪声，SNR<阈值剔除",
     "prediction_refined/<样品>/prediction_snr.csv"),
    ("4 框修正(peak_refinement)", "二次峰检测 + 边界精修 + 置信度/SNR 门控（本报告数据源）",
     "prediction_refined/<样品>/prediction_refined.csv"),
    ("5 推理报告(本工具)", "为精修框补算峰面积 + 跨样品汇总（本报告 + all.csv）",
     "inference_report_<实验名>.md / all.csv"),
]

COLUMN_GUIDE = [
    ("model_prediction_<样品>.csv", "score(模型置信度) / peak_start,peak_end(检测框 RT 范围) / retention_time(峰顶 RT) / "
                                    "area(检测框积分面积) / intensity_max / point_counts / snr / peak_index(图内峰序号)"),
    ("prediction_snr.csv", "SNR 筛选通过行，列同 model_prediction_*.csv"),
    ("prediction_refined.csv", "主峰 main_rt_min/main_rt_max/main_rt_peak/main_height/main_score_ai/main_snr；"
                               "次峰 small/small2/small3（*_rt_min/max/height/height_ratio/source）；"
                               "门控 has_secondary_gate/gate_ok_for_adjustment/lr_repredict_applied"),
    ("prediction_refined_with_area.csv", "在 prediction_refined.csv 基础上补 main_area/small*_area（SNR 基线积分），"
                                         "area 列=main_area"),
    ("all.csv", "全部样品合并明细（sample/compound/ion/RT/高度/置信度/SNR/面积/次峰/检出标记）"),
]


def _appendix(out_root, snr_root):
    L = []
    L.append("## 6. 管线各阶段说明（模型输出之后）")
    L.append("")
    L.append("| 阶段 | 做什么 | 产物位置 |")
    L.append("|---|---|---|")
    for name, what, where in PIPELINE_STAGES:
        L.append(f"| {name} | {what} | `{where}` |")
    L.append("")
    L.append("## 7. 输出文件与关键列说明")
    L.append("")
    L.append("| 文件 | 关键列含义 |")
    L.append("|---|---|")
    for fname, desc in COLUMN_GUIDE:
        L.append(f"| `{fname}` | {desc} |")
    L.append("")
    L.append(f"> 各文件位于 `{out_root}`（相对 `model/`）。"
             f"精修明细 CSV：`{snr_root}/<样品>/SNR_box_*/prediction_refined*.csv`")
    return "\n".join(L)


def render_report(out_root, snr_root, adf, per_sample, samples, timestamp, mode,
                  n_samples, total_seconds, qc_path, qc_text, timing_record=None):
    L = []
    L.append("# 推理报告")
    L.append("")
    L.append(f"- 生成时间: {timestamp or '未知'}")
    L.append(f"- 推理模式: `{mode or 'pipeline'}` | 输出目录: `{out_root}`")
    if n_samples is not None:
        L.append(f"- 样品数: {n_samples} | 总耗时: {total_seconds:.1f}s" if total_seconds else f"- 样品数: {n_samples}")
    L.append("")
    # 样本摘要
    L.append("## 1. 样本摘要")
    L.append("")
    L.append("| 样品 | ROI 图 | 精修峰 | 检出(面积>0) | 带次峰 | 平均置信度 | 平均 SNR | 平均面积 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in samples:
        ps = per_sample.get(s, {})
        L.append(f"| {s} | {ps.get('n_roi', '-')} | {ps.get('n_refined', '-')} | "
                 f"{ps.get('n_detected', '-')} | {ps.get('n_with_small', '-')} | "
                 f"{_fmt(ps.get('mean_score'))} | {_fmt(ps.get('mean_snr'))} | {_fmt(ps.get('mean_area'))} |")
    L.append("")
    # 化合物矩阵
    L.append("## 2. 化合物 × 样品 检出面积矩阵（主峰, intensity·min）")
    L.append("")
    L.append("> 单元格 = 该样品该化合物主峰面积（—=未检出）；离子通道合并取最大值。空白样品出现数值需警惕假阳性。")
    L.append("")
    L.append(render_matrix(adf, samples))
    L.append("")
    # 明细说明
    L.append("## 3. 合并明细")
    L.append("")
    L.append(f"- 明细表：`{out_root / 'all.csv'}`（每样品×每 ROI 一行：RT/高度/置信度/SNR/面积/次峰/门控）")
    L.append(f"- 各样品补面积明细：`{snr_root}/<样品>/prediction_refined_with_area.csv`")
    L.append("")
    # QC
    if qc_text:
        L.append("## 4. QC 汇总")
        L.append("")
        L.append(f"> 来源：`{qc_path}`")
        L.append("")
        L.append(qc_text.strip())
        L.append("")
    # 运行用时（各模块）
    L.append(render_timing_section(timing_record))
    L.append("")
    L.append(_appendix(out_root, snr_root))
    L.append("")
    L.append("---")
    L.append("> 由 `model/tools/evaluation/inference_report.py` 生成。")
    return "\n".join(L)


def _fmt(v):
    return f"{v:.3f}" if v is not None else "—"


def generate_for_pipeline(output_dir, snr_root=None, roi_root=None,
                          exp_name=None,
                          integration_method="snr", do_integrate=True, verbose=True,
                          timing_record=None):
    """生成推理报告，返回摘要 dict。

    供 CLI 手动调用与 pipeline 模式跑完后自动调用共用。
    Step 7c：落盘文件改名为 inference_report_<实验名>.md + all.csv。
    """
    out_root = Path(output_dir).resolve()
    if snr_root is None:
        from tools._shared.artifacts import resolve_snr_root

        snr_root = resolve_snr_root(out_root)
    else:
        snr_root = Path(snr_root)
    if roi_root is None:
        from tools._shared.artifacts import resolve_roi_root

        roi_root = resolve_roi_root(out_root)
    else:
        roi_root = Path(roi_root)

    refined_by_sample = collect_refined(snr_root)
    if not refined_by_sample:
        raise FileNotFoundError(f"{snr_root} 下未找到 prediction_refined.csv")

    if verbose:
        print(f"[INFO] 推理报告: 发现 {len(refined_by_sample)} 个样品的精修结果")
    all_rows, per_sample = build_rows(
        refined_by_sample, roi_root, do_integrate=do_integrate, method=integration_method)
    adf = pd.DataFrame(all_rows)
    samples = sorted(per_sample)
    all_csv = out_root / "all.csv"
    adf.to_csv(all_csv, index=False, encoding="utf-8-sig")

    timestamp, mode, n_samples, total_seconds = load_timing(out_root)
    if timing_record is None:
        timing_record = load_timing_record(out_root)
    qc_path, qc_text = qc_summary_text(out_root)
    md = render_report(out_root, snr_root, adf, per_sample, samples,
                       timestamp, mode, n_samples, total_seconds, qc_path, qc_text,
                       timing_record=timing_record)
    exp_name = (exp_name or "").strip() or out_root.name
    report_md = out_root / ("inference_report_%s.md" % exp_name)
    report_md.write_text(md, encoding="utf-8")

    if verbose:
        print(f"[INFO] 合并明细: {all_csv} ({len(adf)} 行)")
        print(f"[INFO] 推理报告: {report_md}")
        print("\n===== 样本摘要 =====")
        for s in samples:
            ps = per_sample[s]
            print(f"  {s}: ROI={ps['n_roi']} 精修={ps['n_refined']} 检出={ps['n_detected']} "
                  f"次峰={ps['n_with_small']} 平均score={_fmt(ps['mean_score'])} "
                  f"平均SNR={_fmt(ps['mean_snr'])} 平均面积={_fmt(ps['mean_area'])}")
        blanks = [s for s in samples if s.startswith(("空白", "BLANK", "blank"))]
        if blanks:
            fp = adf[(adf["sample"].isin(blanks)) & (adf["detected"])]
            if len(fp):
                print(f"[WARN] 空白样品检出 {len(fp)} 个峰（疑似假阳性，详见报告矩阵）：")
                for _, r in fp.iterrows():
                    print(f"    {r['sample']} / {r['compound']}-{r['ion']} area={r['main_area']:.1f} "
                          f"score={r['main_score_ai']:.3f} snr={r['main_snr']:.1f}")

    return {
        "report_md": str(report_md),
        "all_csv": str(all_csv),
        "n_samples": len(samples),
        "n_rows": len(adf),
        "per_sample": per_sample,
    }


def main():
    ap = argparse.ArgumentParser(description="生成推理报告（合并精修结果 + 补算峰面积）")
    ap.add_argument("--output_dir", default="../output/inference/pipeline_demo",
                    help="pipeline 输出根目录（含 prediction_refined/、xic_roi/）")
    ap.add_argument("--exp_name", default=None,
                    help="实验名（落盘 inference_report_<实验名>.md；缺省用输出目录名）")
    ap.add_argument("--snr_root", default=None, help="默认 <output_dir>/prediction_refined")
    ap.add_argument("--roi_root", default=None, help="默认 <output_dir>/xic_roi（回退旧 xic-roi-batch）")
    ap.add_argument("--integration_method", default="snr",
                    choices=["snr", "peak_adaptive", "linear", "raw"],
                    help="面积积分方法（默认 snr=SNR 基线）")
    ap.add_argument("--no_integrate", action="store_true", help="不补算面积，仅汇总精修原始列")
    args = ap.parse_args()

    try:
        generate_for_pipeline(
            args.output_dir,
            snr_root=args.snr_root,
            roi_root=args.roi_root,
            exp_name=args.exp_name,
            integration_method=args.integration_method,
            do_integrate=not args.no_integrate,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
