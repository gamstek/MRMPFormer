import json
import os
from pathlib import Path


MODEL = Path(__file__).resolve().parents[1]
ROOT = MODEL.parent
SOURCE = ROOT / "data" / "mzml" / "test3"
FIRST_OUT = ROOT / "output" / "inference" / "massnova_test3_centwave_special_v2"
PENDING = MODEL / "project_review" / "cw_test3_pending"
CONFIG = MODEL / "project_review" / "massnova_centwave_test3_remaining.json"
REMAINING_OUT = ROOT / "output" / "inference" / "massnova_test3_centwave_special_v2_remaining"


def main():
    done_root = FIRST_OUT / "prediction_model"
    done = {p.name for p in done_root.iterdir() if p.is_dir()} if done_root.exists() else set()
    PENDING.mkdir(parents=True, exist_ok=True)
    for src in sorted(SOURCE.glob("*.mzML")):
        if src.stem in done:
            continue
        dst = PENDING / src.name
        if not dst.exists():
            os.link(src, dst)

    with (MODEL / "configs" / "massnova.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.update(
        batch_dir=str(PENDING),
        mzml=None,
        scan_finder="centwave",
        exp_name="test3_centwave_special_v2_remaining",
        output_dir=str(REMAINING_OUT),
    )
    with CONFIG.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"done={len(done)} pending={len(list(PENDING.glob('*.mzML')))}")
    print(CONFIG)


if __name__ == "__main__":
    main()
