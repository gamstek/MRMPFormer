# -*- coding: utf-8 -*-
"""单通道 case 诊断：逐 decoder 层 / 逐 query 的框与分数，并与部署 ONNX 对齐校验。

背景：massnova Phase3b 把每个候选 apex±1min 的 400x300 无轴 ROI 图送模型，
ONNX 返回 [Q,4] 框 + [Q] 分数（部署模型 Q=3）。本工具对指定 (mzML, uid) 通道
复现同一输入，逐候选窗口输出：

  - 每层分类分：L1/L2/L3（class_embed 为三层共享头，逐层 logits 各自 softmax）
  - 每层 FDR 精化后的左右边界：base(初始框，无 FDR) / L1 / L2 / L3
  - 逐 query：分数、RT 区间、框宽、中心相对 apex 偏移
  - 复算 massnova 的贪心分配，把最终峰归属到 (候选 apex, query) —— 用于判断
    query2/query3 是否产出重复/虚假框
  - 与部署 ONNX（同权重导出）在同图上的 L3 分数/框做一致性校验

用法（model/ 目录下）：
  python -m tools.diagnostics.trace_channel_queries \
      --mzml ../data/mzml/test2/test2_1.mzML --uid 甲羧除草醚-2 \
      --gt 16.78,17.33
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # model/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from inference import massnova  # noqa: E402
from utils.roi_rt_mapping import box_x_to_rt_minutes, rt_window_bounds_minutes  # noqa: E402
from utils.torch_device import load_torch_checkpoint, resolve_torch_device  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_LAYERS = ("L1", "L2", "L3")
_LAYER_ALPHA = {"base": 0.25, "L1": 0.40, "L2": 0.65, "L3": 1.0}
_LAYER_LW = {"base": 4.0, "L1": 5.0, "L2": 6.0, "L3": 7.0}
_QUERY_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")

# 与 massnova.run_massnova_on_mzml 的 Phase1 参数子集保持一致
_PHASE1_KEYS = ("baseline_percentile", "baseline_mode", "min_peak_ratio",
                "prominence_ratio", "min_prominence_abs", "min_peak_gap_points",
                "min_peak_width_min", "void_time_min", "max_peaks_per_channel",
                "valley_ratio", "min_valley_central_frac")


def build_args(config_path):
    """massnova.build_parser 默认值 + configs/massnova.json 覆盖（与推理同口径）。"""
    cfg = {}
    cfg_path = Path(config_path)
    if cfg_path.exists():
        cfg = {k: v for k, v in json.loads(cfg_path.read_text(encoding="utf-8")).items()
               if not k.startswith("_")}
        cfg.pop("mzml", None)
        cfg.pop("batch_dir", None)
        print("[INFO] 已加载配置: %s" % cfg_path)
    parser = massnova.build_parser()
    if cfg:
        parser.set_defaults(**cfg)
    return parser.parse_args([])


def load_model(model_path):
    """从 .pth checkpoint（含训练 args）重建模型并加载权重。"""
    device = resolve_torch_device(verbose=False)
    ckpt = load_torch_checkpoint(model_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    train_args = ckpt.get("args")
    if train_args is None:
        raise ValueError("checkpoint 缺少 args，无法重建模型结构: %s" % model_path)
    train_args.device = str(device)
    from models import build_model
    if not getattr(train_args, "model", None):
        train_args.model = "mrmpformer_v1"
    result = build_model(train_args)
    model = result[0] if isinstance(result, tuple) else result
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    print("[INFO] 模型已加载: %s | device=%s | num_queries=%d dec_layers=%d cascade=%s"
          % (model_path, device, getattr(train_args, "num_queries", -1),
             getattr(train_args, "dec_layers", -1), getattr(train_args, "fdr_cascade", None)))
    return model, device


def pick_channel(features, uid):
    for feat in features:
        if str(feat["uid"]) == str(uid):
            return feat
    near = sorted({str(f["uid"]) for f in features if str(uid).split("-")[0] in str(f["uid"])})
    raise SystemExit("[ERROR] 通道 uid 不存在: %s\n候选: %s" % (uid, near or "无"))


def window_for(apex, rt, half_min):
    """与 massnova 渲染窗口一致：apex±1min 夹到数据范围，再按 window_half_min 收窄。"""
    rt_lo, rt_hi = rt_window_bounds_minutes(apex, rt)
    if half_min and half_min > 0:
        rt_lo = max(float(rt_lo), apex - float(half_min))
        rt_hi = min(float(rt_hi), apex + float(half_min))
        if rt_hi <= rt_lo:
            rt_lo, rt_hi = rt_window_bounds_minutes(apex, rt)
    return float(rt_lo), float(rt_hi)


def trace_channel(model, device, rt, y, apexes, half_min, tmp_dir, threshold):
    """逐候选窗口前向，返回窗口记录与逐 (层, query) 结果。"""
    records, layer_rows = [], []
    for ci, apex_idx in enumerate(apexes, start=1):
        apex = float(rt[int(apex_idx)])
        rt_lo, rt_hi = window_for(apex, rt, half_min)
        mask = (rt >= rt_lo) & (rt <= rt_hi)
        win_rt, win_int = rt[mask], y[mask]
        if win_rt.size < 2:
            continue
        jpeg = Path(tmp_dir) / ("win_%03d.jpeg" % ci)
        massnova.render_roi_jpeg(win_rt, win_int, rt_lo, rt_hi, str(jpeg))

        with Image.open(jpeg) as im:
            tensor = _TRANSFORM(im.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)

        scores = {
            "L1": out["aux_outputs"][0]["pred_logits"].softmax(-1)[0, :, 0],
            "L2": out["aux_outputs"][1]["pred_logits"].softmax(-1)[0, :, 0],
            "L3": out["pred_logits"].softmax(-1)[0, :, 0],
        }
        init_lr = out["initial_edges_ltrb"][0]                 # [Q,4] xyxy 归一化
        refined = [r[0] for r in out["refined_lr"]]            # 3×[Q,2] 左/右归一化
        n_q = int(scores["L3"].shape[0])

        rec = {"cand_no": ci, "apex": apex, "rt_lo": rt_lo, "rt_hi": rt_hi,
               "win_rt": win_rt, "win_int": win_int, "jpeg": jpeg, "queries": []}
        for q in range(n_q):
            lr = {"base": (float(init_lr[q, 0]), float(init_lr[q, 2]))}
            for k, lname in enumerate(_LAYERS):
                lr[lname] = (float(refined[k][q, 0]), float(refined[k][q, 1]))
            qrec = {"query": q + 1, "lr": lr, "scores": {k: float(scores[k][q]) for k in _LAYERS}}
            qrec["rt"] = {
                lname: (box_x_to_rt_minutes(lr[lname][0] * 400.0, rt_lo, rt_hi),
                        box_x_to_rt_minutes(lr[lname][1] * 400.0, rt_lo, rt_hi))
                for lname in ("base",) + _LAYERS
            }
            rec["queries"].append(qrec)
            for lname in ("base",) + _LAYERS:
                left, right = qrec["rt"][lname]
                layer_rows.append({
                    "cand_no": ci, "apex_rt": apex, "window_lo": rt_lo, "window_hi": rt_hi,
                    "query": q + 1, "stage": lname,
                    "score": qrec["scores"].get(lname, np.nan),
                    "rt_min": left, "rt_max": right, "width_min": right - left,
                    "contains_apex": bool(left <= apex <= right),
                    "center_offset_min": 0.5 * (left + right) - apex,
                })
        records.append(rec)
    return records, pd.DataFrame(layer_rows)


def onnx_crosscheck(onnx_path, tmp_dir, records, threshold):
    """同一批窗口图上跑部署 ONNX，回填 L3 分数/框，校验 PyTorch 追踪保真。"""
    try:
        from inference.onnx_window_predictor import OnnxWindowPredictor

        predictor = OnnxWindowPredictor(onnx_path, use_gpu=0)
    except Exception as exc:  # onnxruntime 缺失或模型不可读时不阻断主流程
        print("[WARN] ONNX 校验跳过: %s" % exc)
        return None
    results = {Path(r["image_path"]).name: r
               for r in predictor(images_path=tmp_dir, threshold=float(threshold))}
    diffs = []
    for rec in records:
        res = results.get(Path(rec["jpeg"]).name)
        if not res:
            continue
        boxes = np.asarray(res["boxes"]).reshape(-1, 4)
        scores = np.asarray(res["scores"]).reshape(-1)
        for qrec in rec["queries"]:
            pt_score = qrec["scores"]["L3"]
            # ONNX 只返回 score>threshold 的 query，未返回即视为低于阈值
            hit = [i for i in range(len(scores))
                   if abs(float(scores[i]) - pt_score) < 1e-4]
            if hit:
                i = hit[0]
                lo = box_x_to_rt_minutes(boxes[i, 0], rec["rt_lo"], rec["rt_hi"])
                hi = box_x_to_rt_minutes(boxes[i, 2], rec["rt_lo"], rec["rt_hi"])
                diffs.append((abs(lo - qrec["rt"]["L3"][0]), abs(hi - qrec["rt"]["L3"][1]),
                              abs(float(scores[i]) - pt_score)))
            elif pt_score > float(threshold):
                diffs.append((np.nan, np.nan, np.nan))  # ONNX 未返回但 PyTorch 高于阈值
    if not diffs:
        print("[WARN] ONNX 校验无配对结果")
        return None
    arr = np.asarray(diffs, dtype=float)
    stat = {"n_pair": int(arr.shape[0]),
            "max_left_drt": float(np.nanmax(arr[:, 0])) if arr.shape[0] else np.nan,
            "max_right_drt": float(np.nanmax(arr[:, 1])) if arr.shape[0] else np.nan,
            "max_score_diff": float(np.nanmax(arr[:, 2])) if arr.shape[0] else np.nan,
            "n_missing": int(np.count_nonzero(np.isnan(arr[:, 0])))}
    print("[INFO] ONNX 对齐: 配对 %d 个 | 左右边界最大偏差 %.6f / %.6f min | 分数最大偏差 %.6f | "
          "未返回 %d" % (stat["n_pair"], stat["max_left_drt"], stat["max_right_drt"],
                         stat["max_score_diff"], stat["n_missing"]))
    return stat


def replicate_greedy(records, threshold):
    """复算 massnova 的候选→框贪心分配，把每个最终峰归属到 (候选, query)。"""
    cands = []
    for rec in records:
        for qrec in rec["queries"]:
            left, right = qrec["rt"]["L3"]
            score = qrec["scores"]["L3"]
            if not (right > left and left <= rec["apex"] <= right):
                continue
            cands.append({"cand_no": rec["cand_no"], "apex": rec["apex"], "query": qrec["query"],
                          "score": score, "left": left, "right": right,
                          "dist": abs(0.5 * (left + right) - rec["apex"])})
    assigned, used = {}, set()
    for c in sorted(cands, key=lambda t: (-t["score"], t["dist"])):
        box_key = (c["cand_no"], c["query"])
        if c["cand_no"] in assigned or box_key in used:
            continue
        used.add(box_key)
        c["validated"] = bool(c["score"] > float(threshold))
        assigned[c["cand_no"]] = c
    return assigned, cands


def plot_ladder(records, gt, out_png):
    """每个候选窗口一张子图：窗口曲线 + 逐 query 的 base/L1/L2/L3 精化阶梯。"""
    n = len(records)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, max(2.1, 2.2 * n)), squeeze=False)
    for ax, rec in zip(axes[:, 0], records):
        ax.plot(rec["win_rt"], rec["win_int"], color="black", lw=1.0, zorder=1)
        ymax = float(np.max(rec["win_int"])) if rec["win_int"].size else 1.0
        ax.set_ylim(-0.05 * ymax, 1.42 * ymax)
        if gt:
            ax.axvspan(gt[0], gt[1], color="#2ecc71", alpha=0.13, zorder=0)
        ax.axvline(rec["apex"], color="red", ls=":", lw=1.0, zorder=2)
        for qi, qrec in enumerate(rec["queries"]):
            color = _QUERY_COLORS[qi % len(_QUERY_COLORS)]
            for li, stage in enumerate(("base",) + _LAYERS):
                left, right = qrec["rt"][stage]
                y0 = ymax * (1.30 - 0.30 * qi - 0.075 * li)
                ax.hlines(y0, left, right, color=color, alpha=_LAYER_ALPHA[stage],
                          lw=_LAYER_LW[stage], zorder=3)
                ax.text(right, y0, " %s %.3f" % (stage, qrec["scores"].get(stage, np.nan)),
                        fontsize=6.5, va="center", color=color, zorder=4)
        ax.set_title("cand%02d  apex=%.3f min  win=[%.3f, %.3f]"
                     % (rec["cand_no"], rec["apex"], rec["rt_lo"], rec["rt_hi"]), fontsize=9)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Retention Time (min)")
    handles = [plt.Line2D([], [], color=_QUERY_COLORS[i % len(_QUERY_COLORS)], lw=3,
                          label="query%d" % (i + 1)) for i in range(3)]
    for stage in ("base",) + _LAYERS:
        handles.append(plt.Line2D([], [], color="grey", lw=_LAYER_LW[stage],
                                  alpha=_LAYER_ALPHA[stage],
                                  label="%s%s" % (stage, "(初始框,无FDR)" if stage == "base" else "")))
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=4)
    fig.suptitle("逐 decoder 层 × 逐 query 边界精化阶梯", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_channel_overlay(rt, y, records, layer_df, assigned, gt, threshold, out_png):
    """整条通道：候选 apex + 每窗每 query 的 L3 框（标分数），并标出被采用的框。"""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(rt, y, color="black", lw=1.0)
    if gt:
        ax.axvspan(gt[0], gt[1], color="#2ecc71", alpha=0.15, label="GT 峰")
    ymax = float(np.max(y)) if y.size else 1.0
    used_boxes = {(c["cand_no"], c["query"]) for c in assigned.values()}
    for rec in records:
        ax.axvline(rec["apex"], color="red", ls=":", lw=0.8)
        for qi, qrec in enumerate(rec["queries"]):
            left, right = qrec["rt"]["L3"]
            score = qrec["scores"]["L3"]
            if not (right > left and left <= rec["apex"] <= right):
                continue
            if score <= float(threshold):
                continue
            color = _QUERY_COLORS[qi % len(_QUERY_COLORS)]
            pick = (rec["cand_no"], qrec["query"]) in used_boxes
            ax.hlines(ymax * (0.30 + 0.16 * qi), left, right, color=color,
                      alpha=0.95 if pick else 0.35,
                      lw=6.0 if pick else 3.0, zorder=3)
            ax.text(right, ymax * (0.30 + 0.16 * qi),
                    " c%dQ%d %.2f%s" % (rec["cand_no"], qrec["query"], score,
                                        "" if pick else " (被贪心丢弃)"),
                    fontsize=7, va="center", color=color, zorder=4)
    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("Intensity")
    ax.set_title("通道级：模型 L3 框（>%.2f 且含 apex）与贪心采用结果" % float(threshold))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_summary(layer_df, out_png):
    """分 query/分层 的分数、框宽、中心偏移统计。"""
    d = layer_df[layer_df["stage"].isin(_LAYERS)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for qi, q in enumerate(sorted(d["query"].unique())):
        color = _QUERY_COLORS[(q - 1) % len(_QUERY_COLORS)]
        for li, stage in enumerate(_LAYERS):
            sub = d[(d["query"] == q) & (d["stage"] == stage)]
            axes[0, 0].scatter([q + 0.12 * (li - 1)] * len(sub), sub["score"],
                               color=color, alpha=_LAYER_ALPHA[stage], s=28)
        axes[0, 1].scatter([q] * len(sub), sub["width_min"], color=color, s=28)
        axes[1, 0].scatter([q] * len(sub), sub["center_offset_min"], color=color, s=28)
    axes[0, 0].set_title("逐层 L3 分数（同一 x 位置三点 = L1/L2/L3）")
    axes[0, 0].set_ylabel("score")
    axes[0, 1].set_title("L3 框宽 (min)")
    axes[0, 1].set_ylabel("width")
    axes[1, 0].set_title("L3 框中心相对 apex 偏移 (min)")
    axes[1, 0].set_ylabel("offset")
    grouped = d.groupby(["stage", "query"])["score"].mean().unstack()
    grouped.plot(kind="bar", ax=axes[1, 1], color=list(_QUERY_COLORS))
    axes[1, 1].set_title("逐层平均分数（stage × query）")
    axes[1, 1].set_ylabel("mean score")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="单通道 case：逐 decoder 层/逐 query 输出诊断")
    ap.add_argument("--mzml", required=True, help="mzML 文件路径")
    ap.add_argument("--uid", required=True, help="通道 uid，如 甲羧除草醚-2")
    ap.add_argument("--model", default="checkpoint/mrmpformerv2.pth", help="PyTorch 权重（含 args）")
    ap.add_argument("--onnx", default="checkpoint/mrmpformerv2.onnx", help="部署 ONNX（对齐校验）")
    ap.add_argument("--config", default="configs/massnova.json", help="massnova 参数配置")
    ap.add_argument("--out_dir", default=None, help="输出目录（缺省 ../output/diagnostics/trace_queries/<样本>_<uid>）")
    ap.add_argument("--gt", default=None, help="人工标注峰 RT 区间 lo,hi（用于叠加参考）")
    ap.add_argument("--window_half_min", type=float, default=None, help="覆盖窗口半宽（缺省取配置）")
    ap.add_argument("--no_onnx", action="store_true", help="跳过 ONNX 对齐校验")
    args = ap.parse_args()

    ns = build_args(args.config)
    half_min = float(args.window_half_min if args.window_half_min is not None
                     else getattr(ns, "scan_window_half_min", 1.0))
    gt = None
    if args.gt:
        lo, hi = (float(v) for v in str(args.gt).split(","))
        gt = (lo, hi)

    mzml = Path(args.mzml).resolve()
    exp = Path(args.out_dir) if args.out_dir else (
        ROOT.parent / "output" / "diagnostics" / "trace_queries"
        / ("%s_%s" % (mzml.stem, args.uid)))
    exp = exp.resolve()
    exp.mkdir(parents=True, exist_ok=True)

    # 1) Phase0：XIC 抽取（与 massnova 同口径）
    features, qc_excluded = massnova.extract_full_xics(
        str(mzml), smooth_sigma=float(ns.smooth_sigma),
        min_chrom_points=int(ns.pipeline_min_chrom_points),
        min_max_intensity=float(ns.pipeline_min_max_intensity))
    feat = pick_channel(features, args.uid)
    rt = np.asarray(feat["rt"], dtype=np.float64)
    y = np.asarray(feat["intensity"], dtype=np.float64)
    print("[INFO] 通道命中: uid=%s chrom_index=%s | %d 点 | RT [%.3f, %.3f] | 峰值 %.3g"
          % (feat["uid"], feat["chrom_index"], rt.size, rt.min(), rt.max(), y.max()))

    # 2) Phase1：候选枚举（与推理一致）
    sp = massnova._scan_params_from_args(ns)
    apexes = massnova.enumerate_peaks(rt, y, **{k: v for k, v in sp.items() if k in _PHASE1_KEYS})
    print("[INFO] Phase1 候选 %d 个: %s" % (len(apexes), [round(float(rt[i]), 3) for i in apexes]))

    # 3) 逐窗口前向（PyTorch 含各层解密输出）
    model, device = load_model(str(Path(args.model).resolve()))
    with tempfile.TemporaryDirectory(prefix="trace_queries_") as tmp:
        records, layer_df = trace_channel(model, device, rt, y, apexes, half_min, tmp,
                                          float(ns.threshold))
        if not args.no_onnx and Path(args.onnx).exists():
            onnx_crosscheck(str(Path(args.onnx).resolve()), tmp, records, float(ns.threshold))

    layer_df.to_csv(exp / "trace_layers_queries.csv", index=False, encoding="utf-8-sig")
    assigned, _cands = replicate_greedy(records, float(ns.threshold))
    print("\n===== 逐 decoder 层 × 逐 query（L3 框为最终框）=====")
    show = layer_df[layer_df["stage"] == "L3"].copy()
    show["adopted"] = [bool(assigned.get(int(c), {}).get("query") == int(q))
                       for c, q in zip(show["cand_no"], show["query"])]
    print(show[["cand_no", "apex_rt", "query", "score", "rt_min", "rt_max", "width_min",
                "contains_apex", "adopted"]].to_string(index=False, float_format="%.4f"))
    print("\n===== 逐层分数（stage × query 平均）=====")
    print(layer_df[layer_df["stage"].isin(_LAYERS)]
          .groupby(["stage", "query"])["score"].mean().unstack().to_string(float_format="%.4f"))

    plot_ladder(records, gt, exp / "fig1_layer_query_ladder.png")
    plot_channel_overlay(rt, y, records, layer_df, assigned, gt, float(ns.threshold),
                         exp / "fig2_channel_query_boxes.png")
    plot_summary(layer_df, exp / "fig3_query_stats.png")
    print("\n[DONE] 输出目录: %s" % exp)
    for name in ("trace_layers_queries.csv", "fig1_layer_query_ladder.png",
                 "fig2_channel_query_boxes.png", "fig3_query_stats.png"):
        print("  - %s" % name)


if __name__ == "__main__":
    main()
