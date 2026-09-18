import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from inference.massnova import write_massnova_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--sample-prefix", required=True)
    parser.add_argument("--sample-count", required=True, type=int)
    args_cli = parser.parse_args()

    output_dir = Path(args_cli.output_dir).resolve()
    config = json.loads(Path(args_cli.config).read_text(encoding="utf-8-sig"))
    run_args = SimpleNamespace(
        **{key: value for key, value in config.items() if not key.startswith("_")}
    )

    sample_infos = []
    missing = []
    for index in range(1, args_cli.sample_count + 1):
        stem = f"{args_cli.sample_prefix}_{index}"
        sample_dir = output_dir / "prediction_refined" / stem
        peak_path = sample_dir / "massnova_peaks.csv"
        summary_path = sample_dir / "scan_summary.csv"
        if not peak_path.is_file() or not summary_path.is_file():
            missing.append(stem)
            continue

        peaks = pd.read_csv(peak_path)
        summary = pd.read_csv(summary_path)
        sample_infos.append(
            {
                "key": stem,
                "mzml_stem": stem,
                "n_channels": int(len(summary)),
                "n_candidates": int(
                    pd.to_numeric(summary["n_peaks"], errors="coerce").fillna(0).sum()
                ),
                "n_peaks": int(len(peaks)),
                "peak_rows": peaks.to_dict("records"),
            }
        )

    if missing:
        raise RuntimeError("Missing sample outputs: " + ", ".join(missing))

    report = write_massnova_report(
        output_dir,
        args_cli.exp_name,
        sample_infos,
        run_args,
        total_seconds=None,
    )
    print(f"samples={len(sample_infos)}")
    print(f"peaks={sum(item['n_peaks'] for item in sample_infos)}")
    print(f"all_csv={report['all_csv']}")
    print(f"report_md={report['report_md']}")


if __name__ == "__main__":
    main()
