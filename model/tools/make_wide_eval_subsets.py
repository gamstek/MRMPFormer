"""Create source-specific read-only evaluation views of wide fine-tune data."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "output/wide_finetune/coco"
OUTPUT = ROOT / "output/wide_finetune/eval_subsets"


def read_split(split):
    return json.loads((DATASET / split / f"{split}_coco.json").read_text(encoding="utf-8"))


def save_split(name, split, images, annotations, categories):
    destination = OUTPUT / name / split
    destination.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(DATASET / split / image["file_name"], destination / image["file_name"])
    (destination / f"{split}_coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories},
                   ensure_ascii=False), encoding="utf-8"
    )


def main():
    train, val = read_split("train"), read_split("val")
    one_train = train["images"][:1]
    train_annotations = [a for a in train["annotations"] if a["image_id"] == one_train[0]["id"]]
    for name, source in (("wide_20251111", "20251111"), ("normal_replay", "normal_replay")):
        selected = [image for image in val["images"] if image["source"] == source]
        ids = {image["id"] for image in selected}
        annotations = [a for a in val["annotations"] if a["image_id"] in ids]
        if not selected:
            raise ValueError(f"empty evaluation source {source}")
        save_split(name, "train", one_train, train_annotations, train["categories"])
        save_split(name, "val", selected, annotations, val["categories"])
        print(f"{name}: {len(selected)} images, {len(annotations)} boxes")


if __name__ == "__main__":
    main()
