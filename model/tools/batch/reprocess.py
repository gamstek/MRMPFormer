# -*- coding: utf-8 -*-
"""
合并 batch_post_newtest_under_snr_filtered.py 与 rerun_snr_under_snr_filtered.py。

统一入口：
  python -m <包名>.batch.reprocess --stage snr ...
  python -m <包名>.batch.reprocess --stage post ...
  python -m <包名>.batch.reprocess --stage snr-post ...

特性：
  - --stage 使用明确枚举：snr / post / snr-post
  - --dry-run：只打印处理目录和参数
  - 批量处理单个样品失败时，由 --stop-on-error 控制是否继续（默认继续）
  - 正式主流水线的默认值是唯一权威来源

注意：因不能修改副本外文件，本模块内硬编码的默认参数应与正式流水线保持一致。
若正式流水线参数有变，需同步更新本模块。
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# 项目根目录
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# === SNR 默认参数（与 mzml_box_outside_snr_pipeline 保持一致） ===
SNR_DEFAULTS = {
    "min_snr": 3.0,
    "smooth_sigma": 0.8,
    "min_noise_points": 5,
    "min_chrom_points": 0,
    "min_chrom_max_intensity": 0.0,
}


# === Postprocess 默认参数（与 peak_refinement post_newtest 保持一致） ===
POST_DEFAULTS = {
    "small_peak_rt_tol": 0.3,
    "min_confidence": 0.99,
    "min_snr": 3.0,
    "min_secondary_ratio": 0.05,
    "noise_barrier_ratio": 0.5,
    "small_noise_window_half": 0.30,
    "main_boundary_noise_percentile": 20.0,
    "edge_max_span_min": 0.6,
    "edge_noise_percentile": 25.0,
    "small_boundary_pad": 0.08,
    "main_double_split_min_valley_drop_ratio": 0.06,
    "main_double_split_min_peak_above_valley_ratio": 0.08,
    "main_double_split_min_peak_sep_ratio_of_span": 0.12,
    "enable_valley_fallback": True,
    "plot_sigma": 0.8,
    "plot_dir_name": "refined_plots",
    "output_name": "prediction_refined.csv",
}


def _snr_subdir_name(min_snr: float) -> str:
    """根据 min_snr 生成 SNR 子目录名，与 pipeline 一致。"""
    t = float(min_snr)
    if t < 0:
        return "SNR_box_all"
    if abs(t - round(t)) < 1e-9:
        return "SNR_box_%d" % int(round(t))
    return "SNR_box_%s" % ("%.10g" % t)


def _find_mzml(mzml_dir: Path, stem: str) -> Optional[Path]:
    """在 mzML 目录下按 stem 查找 .mzML 文件。"""
    for ext in (".mzML", ".mzml"):
        p = mzml_dir / (stem + ext)
        if p.is_file():
            return p
    return None


def _discover_samples(snr_root: Path) -> List[Path]:
    """枚举 snr_filtered 下的样品子目录。"""
    return sorted(p for p in snr_root.iterdir() if p.is_dir())


def run_snr(
    snr_root: Path,
    result_root: Path,
    mzml_dir: Path,
    prediction_basename: str,
    min_snr: float,
    smooth_sigma: float,
    min_noise_points: int,
    min_chrom_points: int,
    min_chrom_max_intensity: float,
    plot_bundle: Optional[Path],
    run_post: bool,
    post_kwargs: dict,
    dry_run: bool,
    stop_on_error: bool,
) -> int:
    """运行 SNR 重跑（可选链式 post_newtest）。"""
    from postprocessing.snr_filter import run as snr_run

    batch_root = result_root / "batch_predictions"
    xic_root = result_root / "xic-roi-batch"

    if not batch_root.is_dir():
        print("[ERROR] 未找到 batch_predictions: %s" % batch_root)
        return 1
    if not xic_root.is_dir():
        print("[ERROR] 未找到 xic-roi-batch: %s" % xic_root)
        return 1
    if not mzml_dir.is_dir():
        print("[ERROR] mzML 目录不存在: %s" % mzml_dir)
        return 1

    samples = _discover_samples(snr_root)
    if not samples:
        print("[WARN] %s 下没有子目录" % snr_root)
        return 0

    cli = float(min_snr)
    eff = float("-inf") if cli < 0 else cli
    snr_sub = _snr_subdir_name(cli)
    n_ok, n_fail = 0, 0

    for samp in samples:
        stem = samp.name
        pred = batch_root / stem / prediction_basename
        roi = xic_root / stem / "roi_windows.csv"
        mz = _find_mzml(mzml_dir, stem)
        out_parent = snr_root / stem

        if mz is None:
            print("[SKIP] 无 mzML: %s / %s" % (mzml_dir, stem))
            n_fail += 1
            if stop_on_error:
                break
            continue
        if not pred.is_file():
            print("[SKIP] 无 prediction: %s" % pred)
            n_fail += 1
            if stop_on_error:
                break
            continue
        if not roi.is_file():
            print("[SKIP] 无 roi_windows: %s" % roi)
            n_fail += 1
            if stop_on_error:
                break
            continue

        snr_kw = {}
        if plot_bundle is not None:
            snr_kw["shared_snr_plots_root"] = str(plot_bundle)
            snr_kw["snr_plot_filename_prefix"] = stem

        print("=" * 60)
        print("[RUN SNR] %s" % stem)
        print("=" * 60)
        if dry_run:
            print("  mzml:", mz)
            print("  pred:", pred)
            print("  roi :", roi)
            print("  out :", out_parent)
            n_ok += 1
            continue

        rc = snr_run(
            str(mz), str(pred), str(out_parent),
            eff, cli, str(roi),
            float(smooth_sigma), int(min_noise_points),
            int(min_chrom_points), float(min_chrom_max_intensity),
            **snr_kw,
        )
        if rc != 0:
            print("[FAIL] SNR 返回码 %s: %s" % (rc, stem))
            n_fail += 1
            if stop_on_error:
                break
            continue
        n_ok += 1

        if not run_post:
            continue

        snr_run_dir = out_parent / snr_sub
        pred_in = snr_run_dir / "prediction.csv"
        if not pred_in.is_file():
            print("[WARN] 无 %s，跳过 post: %s" % (pred_in, stem))
            continue

        _run_post_one(stem, snr_run_dir, xic_root, plot_bundle, post_kwargs, dry_run)

    print("[DONE SNR] 成功 %d，失败/跳过 %d" % (n_ok, n_fail))
    return 0 if n_fail == 0 else 1


def _run_post_one(
    stem: str,
    snr_run_dir: Path,
    xic_root: Path,
    plot_bundle: Optional[Path],
    post_kwargs: dict,
    dry_run: bool,
):
    """对单个样品运行 post_newtest。"""
    from postprocessing import peak_refinement

    post_cli = [
        "post_newtest",
        "--results_dir", str(snr_run_dir.resolve()),
        "--xic_dir", str((xic_root / stem).resolve()),
        "--output_name", str(post_kwargs.get("output_name", "prediction_refined.csv")),
        "--small_peak_rt_tol", str(post_kwargs.get("small_peak_rt_tol", 0.3)),
        "--min_confidence", str(post_kwargs.get("min_confidence", 0.99)),
        "--min_snr", str(post_kwargs.get("min_snr", 3.0)),
        "--min_secondary_ratio", str(post_kwargs.get("min_secondary_ratio", 0.05)),
        "--noise_barrier_ratio", str(post_kwargs.get("noise_barrier_ratio", 0.5)),
        "--small_noise_window_half", str(post_kwargs.get("small_noise_window_half", 0.30)),
        "--main_boundary_noise_percentile", str(post_kwargs.get("main_boundary_noise_percentile", 20.0)),
        "--edge_max_span_min", str(post_kwargs.get("edge_max_span_min", 0.6)),
        "--edge_noise_percentile", str(post_kwargs.get("edge_noise_percentile", 25.0)),
        "--small_boundary_pad", str(post_kwargs.get("small_boundary_pad", 0.08)),
        "--main_double_split_min_valley_drop_ratio", str(post_kwargs.get("main_double_split_min_valley_drop_ratio", 0.06)),
        "--main_double_split_min_peak_above_valley_ratio", str(post_kwargs.get("main_double_split_min_peak_above_valley_ratio", 0.08)),
        "--main_double_split_min_peak_sep_ratio_of_span", str(post_kwargs.get("main_double_split_min_peak_sep_ratio_of_span", 0.12)),
        "--enable_valley_fallback",
    ]
    if post_kwargs.get("plot", False):
        post_cli.extend([
            "--plot",
            "--plot_sigma", str(post_kwargs.get("plot_sigma", 0.8)),
            "--plot_dir_name", str(post_kwargs.get("plot_dir_name", "refined_plots")),
        ])
        if plot_bundle is not None:
            post_cli.extend([
                "--plot_output_parent", str(plot_bundle),
                "--plot_file_prefix", stem,
            ])

    print("[RUN POST] %s" % stem)
    if dry_run:
        print("  ", " ".join(post_cli))
        return

    post_args = peak_refinement.build_parser().parse_args(post_cli)
    peak_refinement.run_post_newtest(post_args)


def run_post_only(
    snr_root: Path,
    result_root: Path,
    snr_subdir: str,
    output_name: str,
    small_peak_rt_tol: float,
    min_confidence: float,
    main_double_split_min_peak_sep_ratio_of_span: Optional[float],
    plot: bool,
    plot_sigma: float,
    plot_dir_name: str,
    plot_output_parent: Optional[Path],
    extra_args: List[str],
    dry_run: bool,
    stop_on_error: bool,
) -> int:
    """仅运行 post_newtest（不重跑 SNR）。"""
    xic_root = result_root / "xic-roi-batch"
    if not xic_root.is_dir():
        print("[ERROR] 未找到 xic-roi-batch: %s" % xic_root)
        return 1

    samples = _discover_samples(snr_root)
    if not samples:
        print("[WARN] %s 下无子目录" % snr_root)
        return 0

    wf = _REPO_ROOT / "postprocessing" / "peak_refinement.py"
    if not wf.is_file():
        print("[ERROR] 未找到: %s" % wf)
        return 1

    n_ok, n_skip = 0, 0
    for samp in samples:
        stem = samp.name
        results_dir = samp / snr_subdir
        pred = results_dir / "prediction.csv"
        xic_dir = xic_root / stem
        xic_npy = xic_dir / "xic_matrix.npy"

        if not pred.is_file():
            print("[SKIP] %s 无 %s" % (stem, pred))
            n_skip += 1
            if stop_on_error:
                break
            continue
        if not xic_npy.is_file():
            print("[SKIP] %s 无 %s" % (stem, xic_npy))
            n_skip += 1
            if stop_on_error:
                break
            continue

        cmd = [
            sys.executable, str(wf),
            "post_newtest",
            "--results_dir", str(results_dir),
            "--xic_dir", str(xic_dir),
            "--output_name", str(output_name),
            "--small_peak_rt_tol", str(small_peak_rt_tol),
            "--min_confidence", str(min_confidence),
            "--min_snr", "3.0",
            "--min_secondary_ratio", "0.05",
            "--noise_barrier_ratio", "0.5",
            "--small_noise_window_half", "0.30",
            "--main_boundary_noise_percentile", "20.0",
            "--edge_max_span_min", "0.6",
            "--edge_noise_percentile", "25.0",
            "--small_boundary_pad", "0.08",
            "--enable_valley_fallback",
        ]
        if main_double_split_min_peak_sep_ratio_of_span is not None:
            cmd.extend([
                "--main_double_split_min_peak_sep_ratio_of_span",
                str(main_double_split_min_peak_sep_ratio_of_span),
            ])
        if plot:
            cmd.extend([
                "--plot",
                "--plot_sigma", str(plot_sigma),
                "--plot_dir_name", str(plot_dir_name),
            ])
            if plot_output_parent:
                cmd.extend([
                    "--plot_output_parent", str(plot_output_parent),
                    "--plot_file_prefix", stem,
                ])
        cmd.extend(extra_args)

        print("=" * 60)
        print("[RUN POST] %s" % stem)
        print("=" * 60)
        if dry_run:
            print(" ", " ".join(cmd))
            n_ok += 1
            continue

        r = subprocess.run(cmd, cwd=str(_REPO_ROOT))
        if r.returncode != 0:
            print("[FAIL] %s 返回码 %s" % (stem, r.returncode))
            n_skip += 1
            if stop_on_error:
                break
        else:
            n_ok += 1

    print("[DONE POST] 成功 %d，跳过/失败 %d" % (n_ok, n_skip))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="批量重跑 SNR 筛选 / post_newtest 框修正",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅重跑 SNR
  python -m <包名>.batch.reprocess --stage snr --snr_filtered_dir "D:\\...\\snr_filtered"

  # 仅重跑 post_newtest
  python -m <包名>.batch.reprocess --stage post --snr_filtered_dir "D:\\...\\snr_filtered"

  # 先 SNR 再 post
  python -m <包名>.batch.reprocess --stage snr-post --snr_filtered_dir "D:\\...\\snr_filtered" --run-post

  # 试运行
  python -m <包名>.batch.reprocess --stage snr --snr_filtered_dir "..." --dry-run
""",
    )
    ap.add_argument("--stage", required=True, choices=("snr", "post", "snr-post"),
                    help="运行阶段：snr=仅SNR, post=仅post_newtest, snr-post=SNR+post")
    ap.add_argument("--snr_filtered_dir", required=True, help="snr_filtered 根目录")
    ap.add_argument("--result_root", default=None,
                    help="含 batch_predictions、xic-roi-batch 的目录，默认 snr_filtered 的父目录")
    ap.add_argument("--mzml_dir", default=None,
                    help="mzML 目录；默认 <result_root 上一级>/mzml")

    # SNR 参数
    ap.add_argument("--prediction_basename", default="prediction.csv")
    ap.add_argument("--min_snr", type=float, default=SNR_DEFAULTS["min_snr"])
    ap.add_argument("--smooth_sigma", type=float, default=SNR_DEFAULTS["smooth_sigma"])
    ap.add_argument("--min_noise_points", type=int, default=SNR_DEFAULTS["min_noise_points"])
    ap.add_argument("--min_chrom_points", type=int, default=SNR_DEFAULTS["min_chrom_points"])
    ap.add_argument("--min_chrom_max_intensity", type=float, default=SNR_DEFAULTS["min_chrom_max_intensity"])

    # Post 参数
    ap.add_argument("--snr_subdir", default="SNR_box_3", help="post 阶段各样品下的 SNR 子目录名")
    ap.add_argument("--output_name", default=POST_DEFAULTS["output_name"])
    ap.add_argument("--small_peak_rt_tol", type=float, default=POST_DEFAULTS["small_peak_rt_tol"])
    ap.add_argument("--min_confidence", type=float, default=POST_DEFAULTS["min_confidence"])
    ap.add_argument("--main_double_split_min_peak_sep_ratio_of_span", type=float, default=None)

    # 绘图
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_sigma", type=float, default=POST_DEFAULTS["plot_sigma"])
    ap.add_argument("--plot_dir_name", default=POST_DEFAULTS["plot_dir_name"])
    ap.add_argument("--plot_output_parent", default=None)
    ap.add_argument("--plot_bundle", default=None, help="SNR 筛选 JPEG 输出根目录")

    # 运行控制
    ap.add_argument("--run_post", action="store_true", help="(snr-post 模式) SNR 后运行 post_newtest")
    ap.add_argument("--dry_run", action="store_true", help="仅打印，不执行业务")
    ap.add_argument("--stop_on_error", action="store_true", help="遇错停止，默认继续处理下一样品")
    ap.add_argument("extra", nargs="*", help="其余参数原样传给 post_newtest")
    args = ap.parse_args()

    snr_root = Path(args.snr_filtered_dir).expanduser().resolve()
    if not snr_root.is_dir():
        print("[ERROR] snr_filtered 目录不存在: %s" % snr_root)
        sys.exit(1)

    result_root = Path(args.result_root).expanduser().resolve() if args.result_root else snr_root.parent
    if args.mzml_dir:
        mzml_dir = Path(args.mzml_dir).expanduser().resolve()
    else:
        mzml_dir = (result_root.parent / "mzml").resolve()

    plot_bundle = Path(args.plot_bundle).expanduser().resolve() if args.plot_bundle else None
    if plot_bundle:
        plot_bundle.mkdir(parents=True, exist_ok=True)

    post_kwargs = {
        "output_name": args.output_name,
        "small_peak_rt_tol": args.small_peak_rt_tol,
        "min_confidence": args.min_confidence,
        "min_snr": args.min_snr,
        "min_secondary_ratio": POST_DEFAULTS["min_secondary_ratio"],
        "plot": args.plot,
        "plot_sigma": args.plot_sigma,
        "plot_dir_name": args.plot_dir_name,
    }

    print("[INFO] stage=%s  snr_root=%s  result_root=%s" % (args.stage, snr_root, result_root))
    print("[INFO] dry_run=%s  stop_on_error=%s" % (args.dry_run, args.stop_on_error))

    if args.stage in ("snr", "snr-post"):
        return run_snr(
            snr_root=snr_root,
            result_root=result_root,
            mzml_dir=mzml_dir,
            prediction_basename=args.prediction_basename,
            min_snr=args.min_snr,
            smooth_sigma=args.smooth_sigma,
            min_noise_points=args.min_noise_points,
            min_chrom_points=args.min_chrom_points,
            min_chrom_max_intensity=args.min_chrom_max_intensity,
            plot_bundle=plot_bundle,
            run_post=args.run_post and args.stage == "snr-post",
            post_kwargs=post_kwargs,
            dry_run=args.dry_run,
            stop_on_error=args.stop_on_error,
        )
    elif args.stage == "post":
        plot_output_parent = Path(args.plot_output_parent).expanduser().resolve() if args.plot_output_parent else None
        return run_post_only(
            snr_root=snr_root,
            result_root=result_root,
            snr_subdir=args.snr_subdir,
            output_name=args.output_name,
            small_peak_rt_tol=args.small_peak_rt_tol,
            min_confidence=args.min_confidence,
            main_double_split_min_peak_sep_ratio_of_span=args.main_double_split_min_peak_sep_ratio_of_span,
            plot=args.plot,
            plot_sigma=args.plot_sigma,
            plot_dir_name=args.plot_dir_name,
            plot_output_parent=plot_output_parent,
            extra_args=args.extra,
            dry_run=args.dry_run,
            stop_on_error=args.stop_on_error,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
