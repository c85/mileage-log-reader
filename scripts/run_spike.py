"""SCRUM-12: measure page registration and ML-7 cell-grid detection on 15 synthetic captures."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import OUTPUT_DIR, REPO_ROOT  # noqa: E402
from mlreader.layout import FIELDS  # noqa: E402
from mlreader.registration import extract_cells, register_page  # noqa: E402
from mlreader.synthetic import capture_page, load_template  # noqa: E402


def _line_coverage(gray, rect, edge):
    x0, y0, x1, y1 = rect
    if edge == "left":
        strip = gray[y0 + 7 : y1 - 7, max(0, x0 - 2) : x0 + 3]
        present = np.any(strip < 170, axis=1)
    elif edge == "right":
        strip = gray[y0 + 7 : y1 - 7, x1 - 2 : min(gray.shape[1], x1 + 3)]
        present = np.any(strip < 170, axis=1)
    elif edge == "top":
        strip = gray[max(0, y0 - 2) : y0 + 3, x0 + 7 : x1 - 7]
        present = np.any(strip < 170, axis=0)
    else:
        strip = gray[y1 - 2 : min(gray.shape[0], y1 + 3), x0 + 7 : x1 - 7]
        present = np.any(strip < 170, axis=0)
    if strip.size == 0:
        return 0.0
    return float(np.mean(present))


def grid_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = [location.rect for locations in FIELDS.values() for location in locations]
    passed = 0
    edges = []
    for rect in boxes:
        coverages = [_line_coverage(gray, rect, edge) for edge in ("left", "right", "top", "bottom")]
        edges.extend(coverages)
        if min(coverages) >= 0.55:
            passed += 1
    odometer_counts = {
        name: len(FIELDS[name])
        for name in ("rows.1.odometer_start", "rows.1.odometer_end")
    }
    return {
        "box_count": len(boxes),
        "boxes_detected": passed,
        "box_success_rate": passed / max(len(boxes), 1),
        "edge_coverage_mean": float(np.mean(edges)) if edges else 0.0,
        "odometer_cell_counts": odometer_counts,
        "odometer_count_ok": all(count == 6 for count in odometer_counts.values()),
    }


def run():
    template = load_template()
    output_dir = OUTPUT_DIR / "spike_registration"
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = ["clean"] * 5 + ["rotated"] * 5 + ["perspective_shadow"] * 5
    records = []
    for index, condition in enumerate(conditions, start=1):
        captured, expected_corners = capture_page(template, condition, seed=6642 + index)
        registered = register_page(captured)
        file_path = output_dir / f"spike_{index:02d}_{condition}.jpg"
        cv2.imwrite(str(file_path), captured)
        record = {
            "sample": index,
            "condition": condition,
            "image": str(file_path.relative_to(REPO_ROOT)),
            "registration_ok": registered.ok,
            "registration_method": registered.method,
            "registration_reason": registered.reason,
        }
        if registered.ok:
            actual = np.asarray(registered.corners, dtype=np.float32)
            expected = np.asarray(expected_corners, dtype=np.float32)
            errors = np.linalg.norm(actual - expected, axis=1)
            record["corner_error_px_mean"] = float(errors.mean())
            record["corner_error_px_max"] = float(errors.max())
            grid = grid_score(registered.image)
            record.update(grid)
            record["accepted"] = bool(
                errors.mean() <= 18 and grid["box_success_rate"] >= 0.80 and grid["odometer_count_ok"]
            )
        else:
            record["accepted"] = False
        records.append(record)

    grouped = {}
    for condition in sorted(set(conditions)):
        subset = [record for record in records if record["condition"] == condition]
        grouped[condition] = {
            "photos": len(subset),
            "registered": sum(record["registration_ok"] for record in subset),
            "accepted": sum(record["accepted"] for record in subset),
            "mean_corner_error_px": float(np.mean([r["corner_error_px_mean"] for r in subset if "corner_error_px_mean" in r])) if any("corner_error_px_mean" in r for r in subset) else None,
            "mean_box_success_rate": float(np.mean([r.get("box_success_rate", 0.0) for r in subset])),
            "failure_causes": [r["registration_reason"] for r in subset if not r["registration_ok"]],
        }
    total_accepted = sum(record["accepted"] for record in records)
    result = {
        "ticket": "SCRUM-12",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_size": len(records),
        "training_performed": False,
        "recognition_accuracy_claimed": False,
        "synthetic_only": True,
        "conditions": grouped,
        "accepted_photos": total_accepted,
        "go_no_go": "GO for the measured clean, small-rotation and moderate-perspective synthetic conditions; unsupported page boundaries remain needs-review cases." if total_accepted == len(records) else "LIMITED GO: proceed only for conditions with measured registration and box-detection success; send failures to review.",
        "photos": records,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    report = [
        "# Registration spike results",
        "",
        f"The SCRUM-12 spike evaluated {len(records)} synthetic blank-form captures and did not train or score handwriting recognition.",
        "",
        "| Capture condition | Photos | Registered | Box-grid accepted | Mean corner error | Mean box success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, group in grouped.items():
        corner_error = f"{group['mean_corner_error_px']:.1f}px" if group["mean_corner_error_px"] is not None else "n/a"
        report.append(
            f"| {condition} | {group['photos']} | {group['registered']} | {group['accepted']} | "
            f"{corner_error} | {group['mean_box_success_rate']:.1%} |"
        )
    report.extend(
        [
            "",
            f"**Recommendation:** {result['go_no_go']}",
            "",
            "Cell-box success requires a registered page, at least 80% of printed character boxes to pass four-edge coverage, and exactly six mapped cells in each odometer field. The capture set is programmatically warped from the provided blank ML-7 template; it is not evidence about handwritten-cell segmentation or model accuracy.",
            "",
            "Failure cases, corner coordinates and per-box scores are in `outputs/spike_registration/results.json`.",
        ]
    )
    (REPO_ROOT / "docs" / "spike_results.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"accepted": total_accepted, "tested": len(records), "conditions": grouped, "recommendation": result["go_no_go"]}, indent=2))


if __name__ == "__main__":
    run()
