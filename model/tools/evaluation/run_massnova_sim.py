# -*- coding: utf-8 -*-
"""在 MRM-XIC 模拟数据集（easy / medium / hard）上运行 massnova 全谱全峰识别。

模拟集（如 data/test_hard）只提供 xic_data/*.json（每条 XIC 含 retention_time /
intensity / q1 / q3 / compound_id / ion_type），没有 mzML。本脚本把每条 XIC 适配成
massnova Phase0 的 features 结构（临时替换 extract_full_xics），其余相位
（候选枚举 → 模型前置验证 → 信号兜底精修 → 门控 → 输出与报告）完全复用
inference.massnova，保证与真实 mzML 路径同口径。

用法（仓库根目录执行）：
    python model/tools/evaluation/run_massnova_sim.py --sim_root data/test_hard --limit 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

# 让脚本可从仓库根直接运行：把 model/ 加入 sys.path，使 inference/utils 可导入
MODEL_DIR = Path(__file__).resolve().parents[2]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from inference import massnova  # noqa: E402  （需先注入 sys.path）

DEFAULT_CONFIG = MODEL_DIR / "configs" / "massnova.json"

# 冒烟测试用：>0 时每个样本只保留前 N 条通道（由 --max_channels 设置，主流程不变）
_MAX_CHANNELS = 0


def _json_extract_full_xics(json_path, smooth_sigma=0.8, min_chrom_points=0,
                            min_max_intensity=0.0, verbose=True):
    """massnova Phase0 的模拟集替代实现：JSON → features（结构/QC 口径对齐 mzML 路径）。

    uid 采用 ``<compound_id>|<ion_type>``，与 label.csv 的
    (compound_id, ion_type) 主键一一对应，便于评测脚本还原真值。
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    features = []
    qc_excluded = []

    for i, xic in enumerate(data.get("xics", [])):
        uid = "%s|%s" % (xic.get("compound_id"), xic.get("ion_type"))
        q1 = xic.get("q1")
        rt = np.asarray(xic.get("retention_time", []), dtype=np.float64)
        raw = np.asarray(xic.get("intensity", []), dtype=np.float64)

        if rt.size == 0 or q1 is None:
            qc_excluded.append({"chrom_index": i, "uid": uid, "q1": q1, "reason": "empty",
                                "n_points": int(rt.size), "max_intensity": 0.0})
            continue

        intensity = (gaussian_filter1d(raw, sigma=float(smooth_sigma))
                     if smooth_sigma > 0 else raw.astype(np.float64, copy=False))
        n_pts = int(rt.size)
        imax = float(np.max(intensity)) if n_pts else 0.0

        if min_chrom_points > 0 and n_pts < int(min_chrom_points):
            qc_excluded.append({"chrom_index": i, "uid": uid, "q1": float(q1),
                                "reason": "too_few_points", "n_points": n_pts,
                                "max_intensity": imax})
            continue
        if min_max_intensity > 0.0 and imax < float(min_max_intensity):
            qc_excluded.append({"chrom_index": i, "uid": uid, "q1": float(q1),
                                "reason": "low_max_intensity", "n_points": n_pts,
                                "max_intensity": imax})
            continue

        features.append({
            "chrom_index": i,
            "uid": uid,
            "compound_name": str(xic.get("compound_id") or ""),
            "q1": float(q1),
            "rt": rt,
            "intensity": intensity,
        })

    if not features:
        raise ValueError("模拟样本无有效 transition（全部被剔除）: %s" % json_path)
    if _MAX_CHANNELS > 0:
        features = features[:_MAX_CHANNELS]
    if verbose:
        print(f"[INFO] massnova Phase0(sim): {len(data.get('xics', []))} 条 XIC → "
              f"{len(features)} 条有效通道（剔除 {len(qc_excluded)}）")
    return features, qc_excluded


def build_args(argv=None):
    """复用 massnova 的参数定义 + configs/massnova.json 作为默认值，再叠加模拟集参数。"""
    ap = massnova.build_parser()
    ap.add_argument("--sim_root", type=str, default=str(Path("data") / "test_hard"),
                    help="模拟数据集根目录（含 xic_data/、label/、generation_manifest.csv）")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                    help="massnova 参数配置 JSON（作为默认值，命令行可覆盖）")
    ap.add_argument("--out_root", type=str, default=None,
                    help="输出根目录；缺省 <repo>/output/inference/massnova_<数据集名>")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个样本（0=全部）")
    ap.add_argument("--max_channels", type=int, default=0,
                    help="每个样本仅处理前 N 条通道（0=全部；用于冒烟测试快速验证流程）")
    ap.add_argument("--no_model_plots", action="store_true",
                    help="关闭 model_plots/ 与三阶段 2min 窗口图（批量测试建议开启，避免海量出图）")
    ap.add_argument("--eval_after", action="store_true",
                    help="推理结束后自动调用 evaluate_massnova_sim.py 生成测试报告与可视化")

    pre = ap.parse_args(argv)
    cfg_path = Path(pre.config)
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        cfg.pop("mzml", None)      # 模拟集不走 mzML 输入
        cfg.pop("batch_dir", None)
        ap.set_defaults(**cfg)
        print(f"[INFO] 已加载 massnova 参数: {cfg_path}")
    else:
        print(f"[WARN] 未找到配置 {cfg_path}，使用内置默认值")

    args = ap.parse_args(argv)
    if args.no_model_plots:
        args.plot = False

    # 模型路径：配置里是相对 model/ 的路径，这里解析成绝对路径，避免受 CWD 影响
    if args.model:
        mp = Path(args.model)
        if not mp.is_absolute():
            mp = (MODEL_DIR / mp) if (MODEL_DIR / mp).exists() else Path.cwd() / mp
        args.model = str(mp)

    sim_root = Path(args.sim_root).resolve()
    if not sim_root.is_dir():
        ap.error(f"--sim_root 不存在: {sim_root}")
    args.sim_root = str(sim_root)
    if not args.exp_name:
        args.exp_name = sim_root.name
    if not args.out_root:
        args.out_root = str(sim_root.parent.parent / "output" / "inference"
                            / ("massnova_%s" % args.exp_name))
    return args


def main(argv=None):
    global _MAX_CHANNELS
    args = build_args(argv)
    _MAX_CHANNELS = max(0, int(args.max_channels))
    sim_root = Path(args.sim_root)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    xic_dir = sim_root / "xic_data"
    files = sorted(xic_dir.glob("*.json"))
    if args.limit and args.limit > 0:
        files = files[:int(args.limit)]
    if not files:
        raise SystemExit(f"未找到模拟样本 JSON: {xic_dir}")

    print(f"[INFO] massnova(sim) 数据集={args.exp_name} 样本={len(files)} "
          f"模型={args.model} 阈值={args.threshold} → {out_root}")

    # 关键一步：把 Phase0 换成 JSON 读取，其余流程完全复用 massnova
    original_extract = massnova.extract_full_xics
    massnova.extract_full_xics = _json_extract_full_xics
    t0 = time.perf_counter()
    sample_infos = []
    try:
        for idx, path in enumerate(files, start=1):
            key = path.stem
            try:
                info = massnova.run_massnova_on_mzml(str(path), key, args, out_root)
                sample_infos.append(info)
                print(f"[{idx}/{len(files)}] {key}: {info['n_channels']} 通道 / "
                      f"{info['n_peaks']} 峰")
            except Exception as exc:  # 单样本失败不阻断整体测试
                print(f"[ERROR] massnova(sim) 失败 {key}: {exc}")
    finally:
        massnova.extract_full_xics = original_extract

    elapsed = time.perf_counter() - t0
    total_peaks = sum(i["n_peaks"] for i in sample_infos)
    total_ch = sum(i["n_channels"] for i in sample_infos)
    stage_seconds = {}
    for info in sample_infos:
        for name, sec in (info.get("stage_seconds") or {}).items():
            stage_seconds[name] = stage_seconds.get(name, 0.0) + float(sec)
    rep = {}
    if sample_infos:
        try:
            rep = massnova.write_massnova_report(out_root, args.exp_name, sample_infos,
                                                 args, total_seconds=elapsed,
                                                 stage_seconds=stage_seconds)
        except Exception as exc:
            print(f"[WARN] massnova 推理报告生成失败（不影响主流程）: {exc}")
    print(f"[DONE] {len(sample_infos)} 样本 / {total_ch} 通道 / {total_peaks} 峰 | "
          f"耗时 {elapsed:.1f}s → {out_root}")
    if rep:
        print(f"[DONE] 推理报告: {rep.get('report_md')}")
    print(f"[DONE] 汇总 CSV: {out_root / 'all.csv'}")

    if args.eval_after and sample_infos:
        print("[INFO] 开始评测并生成测试报告 ...")
        from tools.evaluation import evaluate_massnova_sim
        evaluate_massnova_sim.main([
            "--all_csv", str(out_root / "all.csv"),
            "--sim_root", str(sim_root),
            "--out_dir", str(out_root / "report"),
        ])


if __name__ == "__main__":
    main()
