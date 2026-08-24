# -*- coding: utf-8 -*-
"""
标注数据质量 QC：RT 一致性检查（跨样品极差 + 样品内双离子极差）。

判定规则（详见 docs/plan_qc.md）：
- A 跨样品：同 (compound, channel) 在各 sample 间 rt 极差 > tol → 疑似实验有误
    - 组内样品数 >=3：仅剔偏离组中位数 > tol 的样品行（多数派可信）
    - 组内样品数 ==2：两行都剔（无法仲裁谁错）
- B 双离子：同 (sample_id, compound) 各 channel 间 rt 极差 > tol → 通道张冠李戴/干扰峰误标
    - 两通道都剔（无法判断定量/定性谁错）

超阈值行为：终端 WARN 警示人工复核 + 返回 exclude_keys（调用方负责剔除，不生成 ROI / 不进训练 bbox）。
"""
import re

import pandas as pd


def _parse_rt_field(s):
    """解析 '16.428(0.000)' / '16.428' → 16.428（分钟）；空/非法 → None。
    与 coco_annotation.parse_rt_field 逻辑一致（内联以解除对 pyopenms import 链的依赖）。"""
    s = (s or "").strip()
    if not s:
        return None
    m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else None


def check_label_rt_consistency(labels, tol=1.0):
    """对标注行列表做 RT 一致性检查。

    Args:
        labels: parse_labels_xlsx 输出的 list[dict]（键含 compound/channel/rt/sample_id）
        tol: 极差阈值（分钟）；<=0 关闭检查

    Returns:
        qc_rows: list[dict] — 全量检查结果（含保留行），列见 _row()
        exclude_keys: set[(sample_id, compound, channel)] — 需剔除的行身份三元组
    """
    qc_rows = []
    exclude_keys = set()

    if tol is None or tol <= 0:
        return qc_rows, exclude_keys

    # 可检查行：rt 能解析出数值
    rows = []
    for idx, rec in enumerate(labels):
        rt = _parse_rt_field(rec.get("rt"))
        if rt is None:
            continue
        rows.append({
            "idx": idx,
            "sample_id": (rec.get("sample_id") or "").strip(),
            "compound": (rec.get("compound") or "").strip(),
            "channel": (rec.get("channel") or "").strip(),
            "rt": rt,
        })
    if not rows:
        return qc_rows, exclude_keys

    df = pd.DataFrame(rows)

    def _row(check_type, r, group_median, rt_range, n_group, action, warn):
        qc_rows.append({
            "check_type": check_type,
            "sample_id": r["sample_id"],
            "compound": r["compound"],
            "channel": r["channel"],
            "rt": float(r["rt"]),
            "group_median": None if group_median is None else float(group_median),
            "rt_range": None if rt_range is None else float(rt_range),
            "n_group": int(n_group),
            "action": action,          # excluded / kept
            "suggest_review": bool(warn),
        })

    # ===== A. 跨样品极差：groupby (compound, channel) =====
    for (compound, channel), g in df.groupby(["compound", "channel"]):
        n = len(g)
        if n < 2:
            for _, r in g.iterrows():
                _row("cross_sample", r, None, None, n, "kept", False)
            continue
        rt_range = float(g["rt"].max() - g["rt"].min())
        median = float(g["rt"].median())
        exceed = rt_range > tol
        if exceed:
            print(
                f"[WARN][QC] 跨样品 RT 极差 {rt_range:.3f} min > {tol} min，疑似实验有误，请人工复核: "
                f"化合物「{compound}」通道「{channel}」"
                + "".join(f" | {r['sample_id']}@{r['rt']:.3f}" for _, r in g.iterrows())
            )
        for _, r in g.iterrows():
            if not exceed:
                _row("cross_sample", r, median, rt_range, n, "kept", False)
            elif n >= 3:
                dev = abs(r["rt"] - median)
                excl = dev > tol
                _row("cross_sample", r, median, rt_range, n,
                     "excluded" if excl else "kept", excl)
                if excl:
                    exclude_keys.add((r["sample_id"], r["compound"], r["channel"]))
            else:  # n == 2，无法仲裁，两行都剔
                _row("cross_sample", r, median, rt_range, n, "excluded", True)
                exclude_keys.add((r["sample_id"], r["compound"], r["channel"]))

    # ===== B. 样品内双离子极差：groupby (sample_id, compound) =====
    for (sample_id, compound), g in df.groupby(["sample_id", "compound"]):
        n = len(g)
        if n < 2:
            for _, r in g.iterrows():
                _row("ion_pair", r, None, None, n, "kept", False)
            continue
        rt_range = float(g["rt"].max() - g["rt"].min())
        median = float(g["rt"].median())
        exceed = rt_range > tol
        if exceed:
            print(
                f"[WARN][QC] 双离子 RT 极差 {rt_range:.3f} min > {tol} min，疑似实验有误，请人工复核: "
                f"样品「{sample_id}」化合物「{compound}」"
                + "".join(f" | {r['channel']}@{r['rt']:.3f}" for _, r in g.iterrows())
            )
        for _, r in g.iterrows():
            if not exceed:
                _row("ion_pair", r, median, rt_range, n, "kept", False)
            else:  # 两通道都剔
                _row("ion_pair", r, median, rt_range, n, "excluded", True)
                exclude_keys.add((r["sample_id"], r["compound"], r["channel"]))

    return qc_rows, exclude_keys


def mark_excluded_labels(labels, exclude_keys):
    """给命中 exclude_keys 的标注行打 _qc_excluded 标记（不删行，保持行序对齐）。"""
    n = 0
    for rec in labels:
        key = ((rec.get("sample_id") or "").strip(),
               (rec.get("compound") or "").strip(),
               (rec.get("channel") or "").strip())
        if key in exclude_keys:
            rec["_qc_excluded"] = True
            n += 1
    return n


def write_qc_table(qc_rows, out_path):
    """QC 结果表写出（CSV，utf-8-sig）。返回行数。"""
    from pathlib import Path
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(qc_rows).to_csv(p, index=False, encoding="utf-8-sig")
    return len(qc_rows)


# ---------------------------------------------------------------------------
# 人工预警报告（qc_alert.md）
# ---------------------------------------------------------------------------
# 设计：把「未通过 QC 的标注」以人工可读的形式沉淀为预警报告，随每次 QC 执行
# （训练构建 / 推理管线 / 评估）写入统一文件名 qc_alert.md，命中剔除时同时
# 在终端打出 [ALERT] 行。人工只需读 qc_alert.md 即可完成复核，无需翻 CSV。
_GUIDANCE = {
    "ion_pair": (
        "双离子 RT 极差超阈值：定量/定性离子同化合物应共流出；极差大说明通道归属错误"
        "或标注到干扰峰。两通道均已剔除（不生成 ROI、不进训练、不参与评估指标）。"
        "请人工核实：检查方法表 ert、确认通道归属，修正后重建数据。"
    ),
    "cross_sample": (
        "跨样品 RT 极差超阈值：同化合物同通道跨样品 RT 漂移异常。组内样品数>=3 时仅剔"
        "偏离组中位数的样品行（多数派可信）；n=2 时双方都剔。请人工复核是否标注错误或"
        "仪器漂移。"
    ),
}


def build_qc_alert_markdown(qc_rows, source="", tol=None) -> str:
    """生成人工预警报告 Markdown 文本。

    内容：头部摘要（来源/时间/阈值/检查项/剔除/复核数）+ 结论 + 预警清单表
    （仅 action=excluded）+ 处置指引。无告警时给出明确"未发现异常"结论。
    """
    import datetime

    excluded = [r for r in qc_rows if r.get("action") == "excluded"]
    review = [r for r in qc_rows if r.get("suggest_review")]
    lines = [
        "# QC 人工预警报告（标注 RT 一致性）",
        "",
        "- 来源: %s" % (source or "未指定"),
        "- 生成时间: %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "- QC 阈值: %s" % ("%s min" % tol if tol else "关闭"),
        "- 检查项: %d 项 | 剔除标注: %d 行 | 需人工复核: %d 项"
        % (len(qc_rows), len(excluded), len(review)),
        "",
    ]
    if not excluded and not review:
        lines += ["**结论: 未发现异常，无需人工干预。**", ""]
        return "\n".join(lines) + "\n"

    lines.append("**结论: 发现 %d 项剔除告警，请人工核实标注后修正。**" % len(excluded))
    lines += ["", "## 一、预警清单（action=excluded）",
              "| 检查类型 | 样品 | 化合物 | 通道 | RT(min) | 组中位 | 极差(min) | 处置 |",
              "|---|---|---|---|---|---|---|---|"]
    for r in excluded:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r.get("check_type", ""), r.get("sample_id", ""), r.get("compound", ""),
            r.get("channel", ""),
            ("%.3f" % float(r["rt"])) if r.get("rt") is not None else "",
            ("%.3f" % float(r["group_median"])) if r.get("group_median") is not None else "",
            ("%.3f" % float(r["rt_range"])) if r.get("rt_range") is not None else "",
            r.get("action", "")))
    lines += ["", "## 二、处置指引"]
    for ct, guide in _GUIDANCE.items():
        lines += ["- **%s**：%s" % (ct, guide)]
    lines += ["", "## 三、说明",
              "- 剔除行已生效于本轮：不生成 ROI / 不进训练 bbox / 不参与评估指标计算；",
              "- 复核修正标注文件后需重新构建数据与重跑相关环节。", ""]
    return "\n".join(lines) + "\n"


def write_qc_alert(qc_rows, out_path, source="", tol=None) -> int:
    """写人工预警报告 qc_alert.md（utf-8）。返回告警条数（剔除行数）。"""
    from pathlib import Path
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_qc_alert_markdown(qc_rows, source=source, tol=tol),
                 encoding="utf-8")
    return sum(1 for r in qc_rows if r.get("action") == "excluded")
