# -*- coding: utf-8 -*-
"""对 massnova 在 MRM-XIC 模拟集（test_hard / test_medium / test_easy）上的输出做精度评测，
并生成带可视化的测试报告。

- 真值：<sim_root>/label/label.csv（每 sample×compound×ion 的主峰/双峰/三峰边界与面积）
        + <sim_root>/generation_manifest.csv（difficulty_conditions / noise_subtype）
- 预测：massnova 输出的 all.csv（逐峰 rt_min/rt_peak/rt_max/area/peak_score/boundary_source）
- 匹配：同一通道内按 apex RT 最近贪心一对一匹配，容差 --tol（min）
- 产出：<out_dir>/report.md + metrics_*.csv + matched/fp/fn 明细 + figures/*.png

用法（仓库根目录执行）：
    python model/tools/evaluation/evaluate_massnova_sim.py \
        --all_csv output/inference/massnova_test_hard/all.csv \
        --sim_root data/test_hard --exp_name test_hard
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

# 图中含中文标签：优先使用系统中文字体，避免出图时缺字
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_gt(sim_root: Path):
    """返回 (通道级 gt_ch, 峰级 gt_pk)。通道键 key = '<compound_id>|<ion_type>'。"""
    lab = pd.read_csv(sim_root / "label" / "label.csv", encoding="utf-8-sig", dtype=str)
    ch_rows, pk_rows = [], []
    for _, r in lab.iterrows():
        key = "%s|%s" % (r["compound_id"], r["ion_type"])
        peak_type = str(r.get("peak_type") or "").strip()
        peak_label = int(pd.to_numeric(r.get("peak_label"), errors="coerce") or 0)
        n_gt = int(pd.to_numeric(r.get("peak_number"), errors="coerce") or 0) if peak_label else 0
        ch_rows.append({"sample_id": r["sample_id"], "key": key, "peak_type": peak_type,
                        "peak_label": peak_label, "n_gt": n_gt})
        for k in (1, 2, 3):
            pos = pd.to_numeric(r.get(f"peak_position{k}"), errors="coerce")
            if pd.isna(pos):
                continue
            pk_rows.append({
                "sample_id": r["sample_id"], "key": key, "gt_no": k,
                "pos": float(pos),
                "start": float(pd.to_numeric(r.get(f"peak_start{k}"), errors="coerce")),
                "end": float(pd.to_numeric(r.get(f"peak_end{k}"), errors="coerce")),
                "area": float(pd.to_numeric(r.get(f"peak_area{k}"), errors="coerce")),
                "peak_type": peak_type,
            })
    gt_ch = pd.DataFrame(ch_rows)

    man_path = sim_root / "generation_manifest.csv"
    if man_path.exists():
        man = pd.read_csv(man_path, encoding="utf-8-sig", dtype=str)
        man["key"] = man["compound_id"] + "|" + man["ion_type"]
        man = man[["sample_id", "key", "noise_subtype", "difficulty_conditions", "theoretical_rt"]]
        gt_ch = gt_ch.merge(man, on=["sample_id", "key"], how="left")
    for col in ("noise_subtype", "difficulty_conditions"):
        if col not in gt_ch.columns:
            gt_ch[col] = ""
        gt_ch[col] = gt_ch[col].fillna("")

    gt_ch["通道类别"] = np.where(gt_ch["peak_type"] == "noise", "noise",
                            np.where(gt_ch["peak_type"].isin(["normal", ""]), "normal", "special"))
    return gt_ch, pd.DataFrame(pk_rows)


def load_pred(all_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(all_csv, encoding="utf-8-sig")
    if "uid" not in df.columns:
        raise SystemExit(f"all.csv 缺少 uid 列: {all_csv}")
    df = df.rename(columns={"mzml_stem": "sample_id", "uid": "key"})
    for col in ("rt_min", "rt_peak", "rt_max", "area", "peak_score", "model_score",
                "signal_score", "snr"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 匹配与指标
# ---------------------------------------------------------------------------

def greedy_match(pred_apex: np.ndarray, gt_pos: np.ndarray, tol: float):
    """按 |ΔRT| 升序贪心一对一匹配，返回 [(i_pred, j_gt, d_rt), ...]。"""
    pairs = []
    for i, pa in enumerate(pred_apex):
        for j, gp in enumerate(gt_pos):
            d = abs(float(pa) - float(gp))
            if d <= tol:
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_g, out = set(), set(), []
    for d, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j, d))
    return out


def evaluate(pred: pd.DataFrame, gt_ch: pd.DataFrame, gt_pk: pd.DataFrame, tol: float):
    """逐通道匹配，返回 (matched_df, fp_df, fn_df, ch_stats_df)。"""
    pred_g = {k: g.reset_index(drop=True) for k, g in pred.groupby(["sample_id", "key"])}
    gt_pk_g = {k: g.reset_index(drop=True) for k, g in gt_pk.groupby(["sample_id", "key"])} if len(gt_pk) else {}

    matched, fps, fns, ch_stats = [], [], [], []
    for _, ch in gt_ch.iterrows():
        sk = (ch["sample_id"], ch["key"])
        pr = pred_g.get(sk)
        gt = gt_pk_g.get(sk)
        n_gt = 0 if gt is None else len(gt)
        n_pred = 0 if pr is None else len(pr)
        hits = []
        if n_gt and n_pred:
            hits = greedy_match(pr["rt_peak"].to_numpy(dtype=float),
                                gt["pos"].to_numpy(dtype=float), tol)
        used_g = {j for _, j, _ in hits}
        used_p = {i for i, _, _ in hits}

        for i, j, d in hits:
            p, g = pr.iloc[i], gt.iloc[j]
            inter = max(0.0, min(p["rt_max"], g["end"]) - max(p["rt_min"], g["start"]))
            union = max(p["rt_max"], g["end"]) - min(p["rt_min"], g["start"])
            matched.append({
                "sample_id": sk[0], "key": sk[1], "peak_type": g["peak_type"],
                "通道类别": ch["通道类别"], "difficulty_conditions": ch["difficulty_conditions"],
                "noise_subtype": ch["noise_subtype"],
                "gt_pos": g["pos"], "pred_rt_peak": p["rt_peak"], "d_rt": d,
                "gt_start": g["start"], "gt_end": g["end"],
                "pred_rt_min": p["rt_min"], "pred_rt_max": p["rt_max"],
                "iou": (inter / union) if union > 0 else np.nan,
                "边界宽比": ((p["rt_max"] - p["rt_min"]) / (g["end"] - g["start"])
                         if (g["end"] - g["start"]) > 0 else np.nan),
                "gt_area": g["area"], "pred_area": p["area"],
                "面积比": (p["area"] / g["area"]) if g["area"] > 0 else np.nan,
                "peak_score": p["peak_score"], "boundary_source": p.get("boundary_source", ""),
                "snr": p["snr"],
            })
        if gt is not None:
            for j in range(n_gt):
                if j in used_g:
                    continue
                g = gt.iloc[j]
                fns.append({"sample_id": sk[0], "key": sk[1], "gt_pos": g["pos"],
                            "peak_type": g["peak_type"], "通道类别": ch["通道类别"],
                            "difficulty_conditions": ch["difficulty_conditions"],
                            "gt_area": g["area"]})
        if pr is not None:
            for i in range(n_pred):
                if i in used_p:
                    continue
                p = pr.iloc[i]
                fps.append({"sample_id": sk[0], "key": sk[1], "pred_rt_peak": p["rt_peak"],
                            "peak_type": ch["peak_type"], "通道类别": ch["通道类别"],
                            "difficulty_conditions": ch["difficulty_conditions"],
                            "noise_subtype": ch["noise_subtype"],
                            "peak_score": p["peak_score"],
                            "boundary_source": p.get("boundary_source", ""),
                            "snr": p["snr"], "area": p["area"]})
        ch_stats.append({
            "sample_id": sk[0], "key": sk[1], "peak_type": ch["peak_type"],
            "通道类别": ch["通道类别"], "difficulty_conditions": ch["difficulty_conditions"],
            "noise_subtype": ch["noise_subtype"],
            "n_gt": n_gt, "n_pred": n_pred, "n_tp": len(hits),
            "n_fn": n_gt - len(hits), "n_fp": n_pred - len(hits),
        })
    return (pd.DataFrame(matched), pd.DataFrame(fps), pd.DataFrame(fns), pd.DataFrame(ch_stats))


def headline(matched, fp, fn, ch_stats, n_gt_total, n_pred_total, tol):
    tp = len(matched)
    prec = tp / n_pred_total if n_pred_total else float("nan")
    rec = tp / n_gt_total if n_gt_total else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and not np.isnan(prec) and not np.isnan(rec)
                                             and (prec + rec) > 0) else float("nan")
    out = {
        "匹配容差(min)": tol,
        "GT 峰数": n_gt_total, "预测峰数": n_pred_total,
        "TP": tp, "FP": len(fp), "FN": len(fn),
        "Precision": prec, "Recall": rec, "F1": f1,
    }
    if len(matched):
        out["RT 绝对偏差中位数(min)"] = float(matched["d_rt"].median())
        out["边界 IoU 中位数"] = float(matched["iou"].median())
        out["边界宽比中位数"] = float(matched["边界宽比"].median())
        ratio = matched["面积比"].dropna()
        ratio = ratio[(ratio > 0) & np.isfinite(ratio)]
        if len(ratio):
            med = float(ratio.median())
            out["面积比中位数"] = med
            out["面积比在0.5-2x内比例"] = float(((ratio >= 0.5) & (ratio <= 2)).mean())
            # 单因子标定后的一致性（剔除质量单位/时间单位换算的固定倍数影响）
            calib = ratio / med
            out["标定后面积比 RSD"] = float(calib.std() / calib.mean()) if calib.mean() else float("nan")
        g = matched["gt_area"].to_numpy(dtype=float)
        p = matched["pred_area"].to_numpy(dtype=float)
        ok = (g > 0) & (p > 0) & np.isfinite(g) & np.isfinite(p)
        if int(ok.sum()) >= 3:
            out["面积 log-log 相关系数"] = float(np.corrcoef(np.log10(p[ok]), np.log10(g[ok]))[0, 1])
    noise = ch_stats[ch_stats["通道类别"] == "noise"]
    if len(noise):
        out["噪声通道数"] = len(noise)
        out["噪声通道误报率"] = float((noise["n_fp"] > 0).mean())
        out["噪声通道平均误报峰数"] = float(noise["n_fp"].mean())
    pos = ch_stats[ch_stats["通道类别"] != "noise"]
    if len(pos):
        out["含峰通道检出率(≥1峰)"] = float((pos["n_tp"] > 0).mean())
        out["含峰通道全召回比例"] = float((pos["n_tp"] == pos["n_gt"]).mean())
    return out


def group_metrics(ch_stats: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for name, g in ch_stats.groupby(by):
        gt = int(g["n_gt"].sum())
        tp = int(g["n_tp"].sum())
        rows.append({by: name, "通道数": int(len(g)), "GT 峰数": gt, "TP": tp,
                     "FN": int(g["n_fn"].sum()), "FP": int(g["n_fp"].sum()),
                     "召回率": (tp / gt) if gt else np.nan})
    return pd.DataFrame(rows).sort_values("GT 峰数", ascending=False)


def condition_recall(ch_stats: pd.DataFrame) -> pd.DataFrame:
    """difficulty_conditions 可含多个条件（; 或 , 分隔），逐条件统计召回。"""
    rows = []
    tmp = ch_stats.copy()
    tmp["difficulty_conditions"] = tmp["difficulty_conditions"].fillna("")
    exploded = tmp.assign(_c=tmp["difficulty_conditions"].str.split(r"[;,；|]")).explode("_c")
    exploded["_c"] = exploded["_c"].str.strip()
    exploded = exploded[exploded["_c"] != ""]
    for name, g in exploded.groupby("_c"):
        gt, tp = int(g["n_gt"].sum()), int(g["n_tp"].sum())
        rows.append({"困难条件": name, "通道数": int(len(g)), "GT 峰数": gt, "TP": tp,
                     "FN": int(g["n_fn"].sum()), "FP": int(g["n_fp"].sum()),
                     "召回率": (tp / gt) if gt else np.nan})
    return pd.DataFrame(rows).sort_values("召回率", ascending=False)


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _bar(ax, labels, values, title, ylabel, color="#4C72B0", fmt="{:.3f}", ylim=None):
    bars = ax.bar(range(len(values)), values, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    for b, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def make_figures(matched, fp, fn, ch_stats, by_type, by_cond, head, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = []

    # 1) 主指标
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _bar(axes[0], ["Precision", "Recall", "F1"],
         [head["Precision"], head["Recall"], head["F1"]], "整体指标", "值",
         ylim=(0, 1.05))
    cnt = [head["TP"], head["FP"], head["FN"]]
    bars = axes[1].bar(["TP", "FP", "FN"], cnt, color=["#55A868", "#C44E52", "#8172B2"])
    axes[1].set_title("匹配计数"); axes[1].set_ylabel("峰数"); axes[1].grid(axis="y", alpha=0.3)
    for b, v in zip(bars, cnt):
        axes[1].text(b.get_x() + b.get_width() / 2, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    fig.suptitle("massnova 模拟集测试 · 整体指标")
    p = fig_dir / "fig1_overall.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # 2) 按峰型召回
    if len(by_type):
        d = by_type[by_type["GT 峰数"] > 0]
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(d) + 3), 4))
        _bar(ax, [f"{t}\n(n={int(n)})" for t, n in zip(d.iloc[:, 0], d["GT 峰数"])],
             d["召回率"].tolist(), "按峰型召回率", "召回率", ylim=(0, 1.05))
        p = fig_dir / "fig2_recall_by_peak_type.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # 3) 按困难条件召回
    if len(by_cond):
        d = by_cond.sort_values("召回率")
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(d) + 2)))
        ax.barh([f"{c} (n={int(n)})" for c, n in zip(d["困难条件"], d["GT 峰数"])],
                d["召回率"], color="#DD8452")
        ax.set_xlim(0, 1.05); ax.set_xlabel("召回率"); ax.set_title("按困难条件召回率")
        ax.grid(axis="x", alpha=0.3)
        for i, v in enumerate(d["召回率"]):
            if not np.isnan(v):
                ax.text(v, i, " %.3f" % v, va="center", fontsize=8)
        p = fig_dir / "fig3_recall_by_condition.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # 4) 匹配质量：ΔRT / IoU / 面积比
    if len(matched):
        fig, axes = plt.subplots(1, 4, figsize=(19, 4))
        axes[0].hist(matched["d_rt"].dropna(), bins=40, color="#4C72B0")
        axes[0].set_title("apex RT 绝对偏差"); axes[0].set_xlabel("min"); axes[0].set_ylabel("峰数")
        axes[1].hist(matched["iou"].dropna(), bins=40, color="#55A868")
        axes[1].set_title("边界 IoU"); axes[1].set_xlabel("IoU")
        r = matched["面积比"].dropna()
        r = r[(r > 0) & np.isfinite(r)]
        if len(r):
            axes[2].hist(np.log10(r), bins=40, color="#C44E52")
            axes[2].axvline(0, color="k", linestyle="--", linewidth=1)
            axes[2].set_title("面积比 (log10 pred/GT)"); axes[2].set_xlabel("log10(比值)")
        g = matched["gt_area"].to_numpy(dtype=float)
        p = matched["pred_area"].to_numpy(dtype=float)
        ok = (g > 0) & (p > 0) & np.isfinite(g) & np.isfinite(p)
        if int(ok.sum()) >= 3:
            axes[3].scatter(np.log10(g[ok]), np.log10(p[ok]), s=6, alpha=0.4, color="#8172B2")
            axes[3].set_title("面积一致性 (log-log)")
            axes[3].set_xlabel("log10(GT 面积)"); axes[3].set_ylabel("log10(pred 面积)")
            axes[3].grid(alpha=0.3)
        fig.suptitle("匹配质量分布")
        p = fig_dir / "fig4_match_quality.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # 5) FP 构成
    if len(fp):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        vc = fp["通道类别"].value_counts()
        _bar(axes[0], vc.index.tolist(), vc.values.tolist(), "误报来源（通道类别）", "FP 峰数",
             color="#C44E52", fmt="{:.0f}")
        nz = fp[fp["通道类别"] == "noise"]["noise_subtype"].replace("", "未标注").value_counts()
        if len(nz):
            _bar(axes[1], nz.index.tolist(), nz.values.tolist(), "噪声子类误报", "FP 峰数",
                 color="#937860", fmt="{:.0f}")
        else:
            axes[1].axis("off")
        fig.suptitle("误报（FP）构成")
        p = fig_dir / "fig5_fp_composition.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # 6) 通道级检出 / 噪声误报
    cats = ["normal", "special", "noise"]
    det, fpr = [], []
    for c in cats:
        g = ch_stats[ch_stats["通道类别"] == c]
        if c == "noise":
            det.append(np.nan); fpr.append(float((g["n_fp"] > 0).mean()) if len(g) else np.nan)
        else:
            det.append(float((g["n_tp"] > 0).mean()) if len(g) else np.nan)
            fpr.append(float((g["n_fp"] > 0).mean()) if len(g) else np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _bar(axes[0], cats, det, "通道级检出率（≥1峰命中）", "比例", ylim=(0, 1.05))
    _bar(axes[1], cats, fpr, "通道级误报率（≥1个FP）", "比例", color="#C44E52", ylim=(0, 1.05))
    fig.suptitle("通道级表现")
    p = fig_dir / "fig6_channel_level.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)

    # 7) 峰分分布（命中 vs 误报）
    if len(matched) and len(fp):
        fig, ax = plt.subplots(figsize=(7, 4))
        m = matched["peak_score"].dropna()
        f = fp["peak_score"].dropna()
        bins = np.linspace(0, 1, 41)
        ax.hist(m, bins=bins, alpha=0.65, label="TP (n=%d)" % len(m), color="#55A868")
        ax.hist(f, bins=bins, alpha=0.65, label="FP (n=%d)" % len(f), color="#C44E52")
        ax.set_xlabel("peak_score"); ax.set_ylabel("峰数"); ax.set_title("峰分分布：命中 vs 误报")
        ax.legend(); ax.grid(alpha=0.3)
        p = fig_dir / "fig7_score_distribution.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); figs.append(p)
    return figs


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df is None or not len(df):
        return "_（无数据）_\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "-"
            return float_fmt.format(v) if abs(v) < 1e6 else "%.3g" % v
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, exp_name: str, head, by_type, by_cond, by_noise, figs,
                 ch_stats, fp, fn, matched, tol, pred_csv: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# massnova 模拟集测试报告 · {exp_name}\n")
    lines.append("## 1. 测试设置\n")
    lines.append(f"- 预测来源：`{pred_csv}`（massnova 全谱全峰识别输出 all.csv）")
    lines.append(f"- 匹配规则：同通道内 apex RT 最近贪心一对一匹配，容差 **±{tol} min**")
    lines.append("- 真值来源：`label/label.csv`（V5.0 标注）、`generation_manifest.csv`（困难条件/噪声子类）")
    lines.append("- 指标口径：TP=匹配成功的预测峰；FP=未匹配预测峰；FN=未匹配真值峰\n")

    lines.append("## 2. 核心指标\n")
    kv = pd.DataFrame([{"指标": k, "值": v} for k, v in head.items()])
    lines.append(_md_table(kv))
    lines.append("\n> 注：`面积比` 的绝对倍数受两侧面积单位/时间步长定义影响（massnova 按 RT 积分，"
                 "模拟标签按生成器口径），因此判断定量可用性请看 **标定后面积比 RSD** 与 "
                 "**面积 log-log 相关系数**，而非 `面积比在0.5-2x内比例`。\n")
    lines.append("\n![整体指标](figures/fig1_overall.png)\n")

    lines.append("## 3. 分组结果\n")
    lines.append("### 3.1 按峰型\n")
    lines.append(_md_table(by_type))
    if figs:
        lines.append("\n![按峰型召回](figures/fig2_recall_by_peak_type.png)\n")
    lines.append("### 3.2 按困难条件\n")
    lines.append(_md_table(by_cond))
    lines.append("\n![按困难条件召回](figures/fig3_recall_by_condition.png)\n")
    lines.append("### 3.3 噪声子类\n")
    lines.append(_md_table(by_noise))

    lines.append("\n## 4. 匹配质量与误报\n")
    lines.append("![匹配质量](figures/fig4_match_quality.png)\n")
    lines.append("![误报构成](figures/fig5_fp_composition.png)\n")
    lines.append("![通道级表现](figures/fig6_channel_level.png)\n")
    if (out_dir / "figures" / "fig7_score_distribution.png").exists():
        lines.append("![峰分分布](figures/fig7_score_distribution.png)\n")

    lines.append("## 5. 典型样例\n")
    lines.append(f"- 漏检（FN）最多的困难条件：\n")
    if len(fn):
        cond = fn["difficulty_conditions"].replace("", "无附加条件")
        cond = cond.str.split(r"[;,；|]").explode().str.strip()
        t = cond.value_counts().head(10)
        lines.append(_md_table(pd.DataFrame({"条件": t.index, "FN 峰数": t.values})))
    lines.append(f"\n- 误报（FP）峰分分布：中位数 "
                 f"{float(fp['peak_score'].median()) if len(fp) else float('nan'):.3f}\n")
    lines.append(f"\n- 明细文件：`matched_pairs.csv` / `false_positives.csv` / `false_negatives.csv` / "
                 f"`channel_stats.csv`\n")

    lines.append("\n## 6. 复现命令\n")
    lines.append("```powershell")
    lines.append("# 1) 跑 massnova（模拟集走 JSON XIC 适配）")
    lines.append("python model/tools/evaluation/run_massnova_sim.py --sim_root data/<dataset> --limit 0")
    lines.append("# 2) 评测 + 出图 + 报告")
    lines.append("python model/tools/evaluation/evaluate_massnova_sim.py "
                 "--all_csv output/inference/massnova_<dataset>/all.csv --sim_root data/<dataset>")
    lines.append("```\n")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir / "report.md"


# ---------------------------------------------------------------------------

def build_args(argv=None):
    ap = argparse.ArgumentParser(description="massnova 模拟集精度评测 + 测试报告")
    ap.add_argument("--all_csv", type=str, default=None,
                    help="massnova all.csv 路径；缺省 output/inference/massnova_<exp_name>/all.csv")
    ap.add_argument("--sim_root", type=str, default="data/test_hard")
    ap.add_argument("--exp_name", type=str, default=None, help="缺省取 sim_root 目录名")
    ap.add_argument("--out_dir", type=str, default=None, help="报告输出目录（缺省 all.csv 同级 report/）")
    ap.add_argument("--tol", type=float, default=0.2, help="RT 匹配容差（min）")
    return ap.parse_args(argv)


def main(argv=None):
    args = build_args(argv)
    sim_root = Path(args.sim_root).resolve()
    exp_name = args.exp_name or sim_root.name
    all_csv = Path(args.all_csv) if args.all_csv else \
        Path("output/inference") / ("massnova_%s" % exp_name) / "all.csv"
    if not all_csv.exists():
        raise SystemExit(f"未找到 massnova 输出: {all_csv}（先运行 run_massnova_sim.py）")
    out_dir = Path(args.out_dir) if args.out_dir else all_csv.parent / "report"

    gt_ch, gt_pk = load_gt(sim_root)
    pred = load_pred(all_csv)

    # 部分运行（如 --limit/--max_channels 子集）时，只评测预测覆盖到的样本，
    # 否则未跑样本的真值会全部计入 FN，指标失真
    covered = sorted(set(pred["sample_id"].unique()))
    gt_ch = gt_ch[gt_ch["sample_id"].isin(covered)].copy()
    gt_pk = gt_pk[gt_pk["sample_id"].isin(covered)].copy() if len(gt_pk) else gt_pk
    print(f"[INFO] 评测样本 {len(covered)} 个：{', '.join(covered[:5])}"
          + (" ..." if len(covered) > 5 else ""))

    matched, fp, fn, ch_stats = evaluate(pred, gt_ch, gt_pk, float(args.tol))

    n_gt_total = int(len(gt_pk))
    n_pred_total = int(len(pred))
    head = headline(matched, fp, fn, ch_stats, n_gt_total, n_pred_total, float(args.tol))

    by_type = group_metrics(ch_stats, "peak_type")
    by_cond = condition_recall(ch_stats)
    by_noise = group_metrics(ch_stats[ch_stats["通道类别"] == "noise"], "noise_subtype") \
        if (ch_stats["通道类别"] == "noise").any() else pd.DataFrame()

    figs = make_figures(matched, fp, fn, ch_stats, by_type, by_cond, head, out_dir / "figures")

    out_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out_dir / "matched_pairs.csv", index=False, encoding="utf-8-sig")
    fp.to_csv(out_dir / "false_positives.csv", index=False, encoding="utf-8-sig")
    fn.to_csv(out_dir / "false_negatives.csv", index=False, encoding="utf-8-sig")
    ch_stats.to_csv(out_dir / "channel_stats.csv", index=False, encoding="utf-8-sig")
    by_type.to_csv(out_dir / "metrics_by_peak_type.csv", index=False, encoding="utf-8-sig")
    by_cond.to_csv(out_dir / "metrics_by_condition.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([head]).to_csv(out_dir / "metrics_headline.csv", index=False, encoding="utf-8-sig")

    report = write_report(out_dir, exp_name, head, by_type, by_cond, by_noise, figs,
                          ch_stats, fp, fn, matched, float(args.tol), all_csv)

    print("[DONE] 评测完成")
    print("  Precision=%.4f  Recall=%.4f  F1=%.4f  (TP=%d FP=%d FN=%d)"
          % (head["Precision"], head["Recall"], head["F1"], head["TP"], head["FP"], head["FN"]))
    print(f"  报告: {report}")
    print(f"  图: {out_dir / 'figures'}（{len(figs)} 张）")


if __name__ == "__main__":
    main()
