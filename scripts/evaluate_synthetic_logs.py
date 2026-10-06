"""Measure character, field, extraction and auto-post error metrics on synthetic logs."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import CONFIG_DIR, OUTPUT_DIR  # noqa: E402
from mlreader.emnist import CLASSES  # noqa: E402
from mlreader.pipeline import read_form  # noqa: E402
from mlreader.registration import PREPROCESSING_VERSION  # noqa: E402


def _field_read(document, key):
    if key.startswith("header."):
        return document["header"][key.split(".", 1)[1]]
    if key.startswith("footer."):
        return document["footer"][key.split(".", 1)[1]]
    _, row, field = key.split(".")
    selected = next((r for r in document["rows"] if r["row_number"] == int(row)), None)
    return selected["fields"][field] if selected else None


def _summarize(records):
    total_chars = correct_chars = digit_count = digit_correct = letter_count = letter_correct = 0
    total_fields = exact_fields = total_rows = exact_rows = auto_rows = wrong_auto_rows = 0
    extraction_failures = recognition_failures = 0
    by_condition = defaultdict(list)
    confusion = np.zeros((len(CLASSES), len(CLASSES) + 1), dtype=np.int64)
    field_errors = Counter()
    for record in records:
        condition = record["condition"]
        by_condition[condition].append(record)
        document = record["prediction"]
        truth = record["truth"]
        extraction_failures += len(document.get("cell_extraction_failures", []))
        recognition_failures += len(document.get("recognition_failures", []))
        for key, expected in truth["fields"].items():
            field = _field_read(document, key)
            predicted = field.get("raw_value", "?" * len(expected)) if field else "?" * len(expected)
            total_fields += 1
            exact_fields += predicted == expected
            if predicted != expected:
                field_errors[key.split(".")[-1]] += 1
            for actual, guessed in zip(expected, predicted.ljust(len(expected), "?")[: len(expected)]):
                total_chars += 1
                correct_chars += actual == guessed
                if actual.isdigit():
                    digit_count += 1
                    digit_correct += actual == guessed
                else:
                    letter_count += 1
                    letter_correct += actual == guessed
                actual_i = CLASSES.index(actual)
                guessed_i = CLASSES.index(guessed) if guessed in CLASSES else len(CLASSES)
                confusion[actual_i, guessed_i] += 1
        predicted_rows = {row["row_number"]: row for row in document.get("rows", [])}
        for expected_row in truth["rows"]:
            row_number = expected_row["row_number"]
            predicted_row = predicted_rows.get(row_number)
            row_keys = [f"rows.{row_number}.{name}" for name in ("date", "client", "odometer_start", "odometer_end", "miles")]
            is_correct = bool(predicted_row) and all(
                (_field_read(document, key) or {}).get("raw_value") == truth["fields"][key]
                for key in row_keys
            )
            total_rows += 1
            exact_rows += is_correct
            validation = predicted_row.get("validation", {}) if predicted_row else {}
            if validation.get("route") == "auto_post":
                auto_rows += 1
                wrong_auto_rows += not is_correct
    result = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_warning_cells": sum(record["prediction"].get("preprocessing", {}).get("cells_requiring_review", 0) for record in records),
        "logs": len(records),
        "character_accuracy": correct_chars / total_chars if total_chars else None,
        "character_count": total_chars,
        "digit_accuracy": digit_correct / digit_count if digit_count else None,
        "digit_count": digit_count,
        "capital_letter_accuracy": letter_correct / letter_count if letter_count else None,
        "capital_letter_count": letter_count,
        "field_exact_accuracy": exact_fields / total_fields if total_fields else None,
        "field_count": total_fields,
        "row_exact_accuracy": exact_rows / total_rows if total_rows else None,
        "row_count": total_rows,
        "cell_extraction_failures": extraction_failures,
        "recognition_failures": recognition_failures,
        "auto_post_row_share": auto_rows / total_rows if total_rows else None,
        "auto_post_rows": auto_rows,
        "auto_post_residual_error_rate": wrong_auto_rows / auto_rows if auto_rows else None,
        "incorrect_auto_post_rows": wrong_auto_rows,
        "field_types_with_errors": dict(field_errors),
        "by_condition": {},
    }
    for condition, subset in by_condition.items():
        condition_chars = condition_correct = condition_rows = condition_rows_correct = 0
        condition_auto = condition_wrong_auto = 0
        for item in subset:
            prediction = item["prediction"]
            truth = item["truth"]
            for key, expected in truth["fields"].items():
                field = _field_read(prediction, key)
                guessed = field.get("raw_value", "?" * len(expected)) if field else "?" * len(expected)
                guessed = guessed.ljust(len(expected), "?")[: len(expected)]
                condition_chars += len(expected)
                condition_correct += sum(a == b for a, b in zip(expected, guessed))
            predicted_rows = {row["row_number"]: row for row in prediction.get("rows", [])}
            for expected_row in truth["rows"]:
                row_number = expected_row["row_number"]
                predicted_row = predicted_rows.get(row_number)
                keys = [f"rows.{row_number}.{name}" for name in ("date", "client", "odometer_start", "odometer_end", "miles")]
                correct = bool(predicted_row) and all(
                    (_field_read(prediction, key) or {}).get("raw_value") == truth["fields"][key]
                    for key in keys
                )
                condition_rows += 1
                condition_rows_correct += correct
                route = predicted_row.get("validation", {}).get("route") if predicted_row else None
                if route == "auto_post":
                    condition_auto += 1
                    condition_wrong_auto += not correct
        by_condition_result = {
            "logs": len(subset),
            "character_accuracy": condition_correct / condition_chars if condition_chars else None,
            "row_exact_accuracy": condition_rows_correct / condition_rows if condition_rows else None,
            "auto_post_row_share": condition_auto / condition_rows if condition_rows else None,
            "auto_post_residual_error_rate": condition_wrong_auto / condition_auto if condition_auto else None,
        }
        result["by_condition"][condition] = by_condition_result
    result["character_confusion"] = confusion
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=OUTPUT_DIR / "synthetic_logs" / "manifest.json")
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "synthetic_eval")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 10, 5))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = []
    evaluation_roles = set()
    for item in manifest:
        image_path = args.manifest.parent / item["image"]
        truth_record = json.loads((args.manifest.parent / item["truth_file"]).read_text())
        role = truth_record.get(
            "intended_use", item.get("intended_use", "held_out_test")
        )
        if item.get("intended_use", role) != role:
            raise SystemExit(f"Conflicting intended-use metadata for {item['image']}")
        evaluation_roles.add(role)
        prediction = read_form(image_path, today=args.as_of)
        if not prediction.get("model_available"):
            raise SystemExit("No trained classifier is available. Run scripts/train_model.py before reporting synthetic-log accuracy.")
        records.append(
            {
                "condition": item["condition"],
                "truth": truth_record["truth"],
                "prediction": prediction,
            }
        )
    if len(evaluation_roles) > 1:
        raise SystemExit("Do not mix development and held-out evaluation forms in one score run.")
    evaluation_role = evaluation_roles.pop() if evaluation_roles else "held_out_test"
    metrics = _summarize(records)
    confusion = metrics.pop("character_confusion")
    metrics["evaluation_role"] = evaluation_role
    if evaluation_role == "development":
        metrics["note"] = (
            "Development-set scores are for iteration only. Do not report them as "
            "independent held-out performance or use them to estimate business savings."
        )
    else:
        monthly_logs = 10000
        clerk_cost_per_log = 3.40
        all_auto_share = sum(
            bool(record["prediction"].get("rows"))
            and all(row.get("validation", {}).get("route") == "auto_post" for row in record["prediction"]["rows"])
            for record in records
        ) / max(len(records), 1)
        metrics["business_scenario"] = {
            "assumption": "A log costs $3.40 in AP handling if any row needs review; only a fully auto-posted log avoids that full per-log cost. Model, platform, QA and exception-management cost are not included.",
            "monthly_logs": monthly_logs,
            "current_monthly_keying_cost_usd": round(monthly_logs * clerk_cost_per_log, 2),
            "measured_fully_auto_log_share": all_auto_share,
            "scenario_monthly_gross_clerk_cost_avoided_usd": round(monthly_logs * clerk_cost_per_log * all_auto_share, 2),
            "scenario_remaining_manual_handling_usd": round(monthly_logs * clerk_cost_per_log * (1 - all_auto_share), 2),
            "monthly_rows_at_brief_volume": 70000,
        }
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.data_dir / "character_confusion.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["truth/prediction", *CLASSES, "?"])
        for label, row in zip(CLASSES, confusion):
            writer.writerow([label, *row.tolist()])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
