"""Generate made-up logs using leak-free EMNIST ByClass test characters."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import OUTPUT_DIR  # noqa: E402
from mlreader.synthetic import (  # noqa: E402
    build_log_truth,
    capture_page,
    draw_test_characters,
    load_leak_free_test_split,
    load_template,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=6642)
    parser.add_argument("--week-ending", default="2026-09-27", help="Sunday as YYYY-MM-DD")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--fault-row", type=int, help="Insert an intentional row-mile discrepancy in each log")
    parser.add_argument("--fault-total", action="store_true", help="Make the written weekly total differ by one mile")
    parser.add_argument("--quality", choices=("clean", "rotated", "perspective_shadow", "mixed"), default="mixed")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "synthetic_logs")
    args = parser.parse_args()
    week_ending = date.fromisoformat(args.week_ending)
    test_images, test_labels, source_indices, excluded_count = load_leak_free_test_split()
    template = load_template()
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    condition_cycle = ["clean", "rotated", "perspective_shadow"]
    for index in range(args.count):
        condition = condition_cycle[index % len(condition_cycle)] if args.quality == "mixed" else args.quality
        truth = build_log_truth(
            week_ending=week_ending,
            rows=args.rows,
            seed=args.seed + index,
            discrepancy_row=args.fault_row,
            total_offset=1 if args.fault_total else 0,
        )
        page, sources = draw_test_characters(truth, test_images, test_labels, rng, source_indices)
        captured, _ = capture_page(page, condition, seed=args.seed + index)
        stem = f"log_{index + 1:03d}_{condition}"
        image_path = args.output / f"{stem}.jpg"
        cv2.imwrite(str(image_path), captured)
        truth_path = args.output / f"{stem}.truth.json"
        record = {
            "image": image_path.name,
            "condition": condition,
            "truth_file": truth_path.name,
            "truth": truth,
            "character_sources": sources,
            "excluded_test_images_shared_with_train_or_validation": excluded_count,
            "split": "EMNIST ByClass test only",
        }
        truth_path.write_text(json.dumps(record, indent=2) + "\n")
        manifest.append({"image": image_path.name, "condition": condition, "truth_file": truth_path.name})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest)} synthetic logs to {args.output} (test images shared with train/val excluded: {excluded_count}).")


if __name__ == "__main__":
    main()
