"""Score the pre-rendered synthetic ML-7 image fixtures against their labels."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import CONFIG_DIR, DATA_DIR, OUTPUT_DIR, REPO_ROOT  # noqa: E402
from mlreader.emnist import CLASSES  # noqa: E402
from mlreader.pipeline import read_form  # noqa: E402


GROUND_TRUTH_COLUMNS = {
    "FORM_ID", "ROW_NUMBER", "EMPLOYEE_ID", "WEEK_ENDING", "DATE_MMDD",
    "CLIENT", "ODOMETER_START", "ODOMETER_END", "MILES", "TOTAL_MILES",
}
ROW_FIELDS = ("date", "client", "odometer_start", "odometer_end", "miles")
FIELD_WIDTHS = {
    "header.employee_id": 6,
    "header.week_ending": 6,
    "footer.total_miles": 4,
    "rows.date": 4,
    "rows.client": 3,
    "rows.odometer_start": 6,
    "rows.odometer_end": 6,
    "rows.miles": 3,
}


def _one_value(rows, column, form_id):
    values = {row[column].strip() for row in rows}
    if len(values) != 1:
        raise ValueError(f"{form_id} has inconsistent {column} labels: {sorted(values)}")
    return values.pop()


def _fixed_label(value, width, field_name, form_id, row_number=None):
    location = f"{form_id} row {row_number}" if row_number is not None else form_id
    if len(value) != width:
        raise ValueError(f"{location} {field_name} label must be {width} characters: {value!r}")
    if field_name.endswith(".client") and not value.isalpha():
        raise ValueError(f"{location} {field_name} label must contain only letters: {value!r}")
    if (
        field_name != "header.employee_id"
        and not field_name.endswith(".client")
        and not value.isdigit()
    ):
        raise ValueError(f"{location} {field_name} label must contain only digits: {value!r}")
    if field_name == "header.employee_id" and not value.isalnum():
        raise ValueError(f"{location} employee ID must be alphanumeric: {value!r}")
    return value


def _labels_for_form(form_id, rows):
    employee = _one_value(rows, "EMPLOYEE_ID", form_id).upper()
    week_ending = datetime.strptime(
        _one_value(rows, "WEEK_ENDING", form_id), "%m/%d/%y"
    ).strftime("%m%d%y")
    total = _one_value(rows, "TOTAL_MILES", form_id)
    labels = {
        "header.employee_id": _fixed_label(employee, 6, "header.employee_id", form_id),
        "header.week_ending": _fixed_label(week_ending, 6, "header.week_ending", form_id),
        "footer.total_miles": _fixed_label(total, 4, "footer.total_miles", form_id),
    }
    row_labels = {}
    ordered_rows = sorted(rows, key=lambda row: int(row["ROW_NUMBER"]))
    numbers = [int(row["ROW_NUMBER"]) for row in ordered_rows]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"{form_id} row labels must be unique and contiguous from 1.")
    if numbers != list(range(1, 7)):
        raise ValueError(f"{form_id} must contain the six filled trip rows.")

    for row in ordered_rows:
        row_number = int(row["ROW_NUMBER"])
        values = {
            "date": row["DATE_MMDD"].strip(),
            "client": row["CLIENT"].strip().upper(),
            "odometer_start": row["ODOMETER_START"].strip(),
            "odometer_end": row["ODOMETER_END"].strip(),
            "miles": row["MILES"].strip(),
        }
        for name, value in values.items():
            key = f"rows.{row_number}.{name}"
            labels[key] = _fixed_label(
                value, FIELD_WIDTHS[f"rows.{name}"], key, form_id, row_number
            )
        row_labels[row_number] = values
    return labels, row_labels


def _field_read(document, key):
    if key.startswith("header."):
        return document.get("header", {}).get(key.split(".", 1)[1])
    if key.startswith("footer."):
        return document.get("footer", {}).get(key.split(".", 1)[1])
    _, row_text, name = key.split(".")
    row = next(
        (item for item in document.get("rows", []) if item["row_number"] == int(row_text)),
        None,
    )
    return row.get("fields", {}).get(name) if row else None


def _predicted_cells(field, width):
    characters = field.get("characters", []) if field else []
    predicted = []
    statuses = []
    for position in range(width):
        if position < len(characters):
            character = characters[position]
            predicted.append(character.get("character") or "?")
            statuses.append(character.get("extraction_status", "unknown"))
        else:
            predicted.append("?")
            statuses.append("not_read")
    return predicted, statuses


def _score_field(field, expected):
    predicted, statuses = _predicted_cells(field, len(expected))
    exact = all(
        status == "ok" and guess == truth
        for guess, status, truth in zip(predicted, statuses, expected)
    )
    correct = sum(
        status == "ok" and guess == truth
        for guess, status, truth in zip(predicted, statuses, expected)
    )
    return exact, predicted, statuses, correct


def _row_route(document, row_number):
    row = next(
        (item for item in document.get("rows", []) if item["row_number"] == row_number),
        None,
    )
    return row.get("validation", {}).get("route") if row else None


def _expected_check_status(document, check_path):
    parts = check_path.split(".")
    if len(parts) != 3 or parts[0] != "rows":
        raise ValueError(f"Unsupported expected check path: {check_path!r}")
    row_number = int(parts[1])
    row = next(
        (item for item in document.get("rows", []) if item["row_number"] == row_number),
        None,
    )
    if row is None:
        return None
    return row.get("validation", {}).get("checks", {}).get(parts[2])


def _case_score(case):
    document = case["prediction"]
    labels = case["labels"]
    row_labels = case["row_labels"]
    field_exact = field_count = char_correct = char_count = 0
    digit_correct = digit_count = letter_correct = letter_count = 0
    field_results = {}
    field_errors = []
    confusion = [[0] * (len(CLASSES) + 1) for _ in CLASSES]
    class_indices = {label: index for index, label in enumerate(CLASSES)}

    for key, expected in labels.items():
        exact, predicted, statuses, correct = _score_field(
            _field_read(document, key), expected
        )
        field_results[key] = exact
        field_count += 1
        field_exact += exact
        char_count += len(expected)
        char_correct += correct
        if not exact:
            field_errors.append(
                {
                    "form_id": case["form_id"],
                    "field": key,
                    "truth": expected,
                    "prediction": "".join(
                        guess if status == "ok" else "?"
                        for guess, status in zip(predicted, statuses)
                    ),
                }
            )
        for truth, guess, status in zip(expected, predicted, statuses):
            is_correct = status == "ok" and guess == truth
            if truth.isdigit():
                digit_count += 1
                digit_correct += is_correct
            else:
                letter_count += 1
                letter_correct += is_correct
            predicted_index = class_indices.get(guess, len(CLASSES)) if status == "ok" else len(CLASSES)
            confusion[class_indices[truth]][predicted_index] += 1

    predicted_rows = {
        row["row_number"]: row for row in document.get("rows", [])
    }
    exact_rows = 0
    auto_post_rows = 0
    incorrect_auto_post_rows = 0
    route_counts = Counter()
    for row_number in row_labels:
        exact = all(
            field_results.get(f"rows.{row_number}.{name}", False)
            for name in ROW_FIELDS
        )
        exact_rows += exact
        route = _row_route(document, row_number) or "not_read"
        route_counts[route] += 1
        if route == "auto_post":
            auto_post_rows += 1
            incorrect_auto_post_rows += not exact

    expected_checks = case["manifest_case"].get("expected_failed_checks", [])
    failed_check_results = []
    for check_path in expected_checks:
        passed = _expected_check_status(document, check_path)
        failed_check_results.append(
            {"check": check_path, "detected": passed is False, "passed": passed}
        )
    detected_checks = sum(item["detected"] for item in failed_check_results)
    actual_failed_checks = []
    for row in document.get("rows", []):
        checks = row.get("validation", {}).get("checks", {})
        actual_failed_checks.extend(
            f"rows.{row['row_number']}.{name}"
            for name, passed in checks.items()
            if passed is False
        )

    return {
        "form_id": case["form_id"],
        "case_type": case["manifest_case"]["case_type"],
        "condition": case["manifest_case"]["condition"],
        "related_form_group": case["manifest_case"].get("related_form_group", ""),
        "registration_ok": bool(document.get("registration", {}).get("ok")),
        "model_available": bool(document.get("model_available")),
        "character_count": char_count,
        "correct_characters": char_correct,
        "character_accuracy": char_correct / char_count if char_count else None,
        "digit_count": digit_count,
        "correct_digits": digit_correct,
        "digit_accuracy": digit_correct / digit_count if digit_count else None,
        "capital_letter_count": letter_count,
        "correct_capital_letters": letter_correct,
        "capital_letter_accuracy": letter_correct / letter_count if letter_count else None,
        "field_count": field_count,
        "exact_fields": field_exact,
        "field_exact_accuracy": field_exact / field_count if field_count else None,
        "trip_rows": len(row_labels),
        "exact_rows": exact_rows,
        "row_exact_accuracy": exact_rows / len(row_labels) if row_labels else None,
        "predicted_rows": len(predicted_rows),
        "extra_predicted_rows": len(set(predicted_rows) - set(row_labels)),
        "cell_extraction_successes": sum(
            status == "ok"
            for key, expected in labels.items()
            for _, status in zip(
                _predicted_cells(_field_read(document, key), len(expected))[0],
                _predicted_cells(_field_read(document, key), len(expected))[1],
            )
        ),
        "row_route_counts": dict(route_counts),
        "expected_failed_checks": expected_checks,
        "expected_failed_checks_detected": detected_checks,
        "all_expected_failed_checks_detected": (
            detected_checks == len(expected_checks) if expected_checks else None
        ),
        "failed_check_results": failed_check_results,
        "actual_failed_checks": actual_failed_checks,
        "auto_post_rows": auto_post_rows,
        "incorrect_auto_post_rows": incorrect_auto_post_rows,
        "field_errors": field_errors,
        "character_confusion": confusion,
    }


def _aggregate(case_scores):
    totals = Counter()
    confusion = [[0] * (len(CLASSES) + 1) for _ in CLASSES]
    route_counts = Counter()
    field_metrics = defaultdict(Counter)
    for case in case_scores:
        for name in (
            "character_count", "correct_characters", "digit_count", "correct_digits",
            "capital_letter_count", "correct_capital_letters", "field_count", "exact_fields",
            "trip_rows", "exact_rows", "extra_predicted_rows", "cell_extraction_successes",
            "auto_post_rows", "incorrect_auto_post_rows",
        ):
            totals[name] += case[name]
        for label, row in zip(CLASSES, case["character_confusion"]):
            index = CLASSES.index(label)
            confusion[index] = [a + b for a, b in zip(confusion[index], row)]
        for route, count in case["row_route_counts"].items():
            route_counts[route] += count
        for error in case["field_errors"]:
            field_metrics[error["field"].split(".")[-1]]["errors"] += 1

    # Field totals include exact and incorrect fields, so derive them from each
    # case's character-independent count and retain error counts separately.
    for case in case_scores:
        for field_type, count in case.get("field_type_counts", {}).items():
            field_metrics[field_type]["total"] += count
    for field_type, counts in field_metrics.items():
        counts["exact"] = counts["total"] - counts["errors"]

    char_count = totals["character_count"]
    digit_count = totals["digit_count"]
    letter_count = totals["capital_letter_count"]
    field_count = totals["field_count"]
    trip_rows = totals["trip_rows"]
    return {
        "images": len(case_scores),
        "registered_images": sum(case["registration_ok"] for case in case_scores),
        "registration_success_rate": (
            sum(case["registration_ok"] for case in case_scores) / len(case_scores)
            if case_scores else None
        ),
        "characters": char_count,
        "correct_characters": totals["correct_characters"],
        "character_accuracy": totals["correct_characters"] / char_count if char_count else None,
        "digit_characters": digit_count,
        "correct_digits": totals["correct_digits"],
        "digit_accuracy": totals["correct_digits"] / digit_count if digit_count else None,
        "capital_letter_characters": letter_count,
        "correct_capital_letters": totals["correct_capital_letters"],
        "capital_letter_accuracy": totals["correct_capital_letters"] / letter_count if letter_count else None,
        "fields": field_count,
        "exact_fields": totals["exact_fields"],
        "field_exact_accuracy": totals["exact_fields"] / field_count if field_count else None,
        "trip_rows": trip_rows,
        "exact_rows": totals["exact_rows"],
        "row_exact_accuracy": totals["exact_rows"] / trip_rows if trip_rows else None,
        "auto_post_rows": totals["auto_post_rows"],
        "incorrect_auto_post_rows": totals["incorrect_auto_post_rows"],
        "auto_post_residual_error_rate": (
            totals["incorrect_auto_post_rows"] / totals["auto_post_rows"]
            if totals["auto_post_rows"] else None
        ),
        "extra_predicted_rows": totals["extra_predicted_rows"],
        "cell_extraction_successes": totals["cell_extraction_successes"],
        "cell_extraction_success_rate": totals["cell_extraction_successes"] / char_count if char_count else None,
        "row_validation_routes": dict(route_counts),
        "field_metrics": {
            name: {
                "exact": counts["exact"],
                "total": counts["total"],
                "exact_accuracy": counts["exact"] / counts["total"] if counts["total"] else None,
            }
            for name, counts in sorted(field_metrics.items())
        },
        "character_confusion": confusion,
    }


def _write_confusion(path, confusion):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["truth/prediction", *CLASSES, "?"])
        for label, row in zip(CLASSES, confusion):
            writer.writerow([label, *row])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    fixture_dir = REPO_ROOT / "examples/synthetic_forms"
    parser.add_argument("--manifest", type=Path, default=fixture_dir / "manifest.json")
    parser.add_argument("--ground-truth", type=Path, default=fixture_dir / "ground_truth.csv")
    parser.add_argument("--images-dir", type=Path, default=fixture_dir)
    parser.add_argument("--references", type=Path, default=fixture_dir / "reference_data.json")
    parser.add_argument("--policy", type=Path, default=CONFIG_DIR / "reader_policy.json")
    parser.add_argument("--model", type=Path, default=DATA_DIR / "models/emnist_mlp.npz")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "synthetic_forms_eval")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 10, 5))
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"No trained classifier found at {args.model}. Run scripts/train_model.py first.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with args.ground_truth.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = GROUND_TRUTH_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Ground truth is missing columns: {sorted(missing)}")
        grouped_rows = defaultdict(list)
        for row in reader:
            grouped_rows[row["FORM_ID"].strip()].append(row)
    case_ids = {item["form_id"] for item in manifest["cases"]}
    if case_ids != set(grouped_rows):
        raise SystemExit(
            "Manifest cases and ground-truth FORM_IDs differ: "
            f"manifest-only={sorted(case_ids - set(grouped_rows))}, "
            f"ground-truth-only={sorted(set(grouped_rows) - case_ids)}"
        )

    cases = []
    for item in manifest["cases"]:
        form_id = item["form_id"]
        image_path = args.images_dir / item["image"]
        if not image_path.is_file():
            raise SystemExit(f"Missing fixture image: {image_path}")
        labels, row_labels = _labels_for_form(form_id, grouped_rows[form_id])
        prediction = read_form(
            image_path,
            model_path=args.model,
            references=args.references,
            policy=args.policy,
            today=args.as_of,
        )
        if prediction.get("registration", {}).get("ok") and not prediction.get("model_available"):
            raise SystemExit("The classifier could not be loaded; no scores were produced.")
        cases.append(
            {
                "form_id": form_id,
                "manifest_case": item,
                "labels": labels,
                "row_labels": row_labels,
                "prediction": prediction,
            }
        )

    scored = []
    for case in cases:
        score = _case_score(case)
        score["field_type_counts"] = Counter(key.split(".")[-1] for key in case["labels"])
        scored.append(score)
    valid_scores = [item for item in scored if item["case_type"] == "valid"]
    invalid_scores = [item for item in scored if item["case_type"] == "invalid"]
    valid_by_condition = defaultdict(list)
    valid_by_group = defaultdict(list)
    for score in valid_scores:
        valid_by_condition[score["condition"]].append(score)
        valid_by_group[score["related_form_group"]].append(score)

    valid_metrics = _aggregate(valid_scores)
    valid_metrics.pop("character_confusion", None)
    valid_metrics["underlying_forms"] = len(valid_by_group)
    valid_metrics["image_variants"] = len(valid_scores)
    valid_metrics["by_condition"] = {
        condition: _aggregate(items) | {"underlying_forms": len(valid_by_group)}
        for condition, items in sorted(valid_by_condition.items())
    }
    for condition_metrics in valid_metrics["by_condition"].values():
        condition_metrics.pop("character_confusion", None)
    valid_metrics["by_form_group"] = {
        group: _aggregate(items)
        for group, items in sorted(valid_by_group.items())
    }
    for group_metrics in valid_metrics["by_form_group"].values():
        group_metrics.pop("character_confusion", None)

    expected_count = sum(len(item["expected_failed_checks"]) for item in invalid_scores)
    detected_count = sum(item["expected_failed_checks_detected"] for item in invalid_scores)
    invalid_metrics = {
        "images": len(invalid_scores),
        "registered_images": sum(item["registration_ok"] for item in invalid_scores),
        "registration_success_rate": (
            sum(item["registration_ok"] for item in invalid_scores) / len(invalid_scores)
            if invalid_scores else None
        ),
        "expected_failed_checks": expected_count,
        "expected_failed_checks_detected": detected_count,
        "expected_failed_check_detection_rate": detected_count / expected_count if expected_count else None,
        "cases_with_all_expected_failures_detected": sum(
            item["all_expected_failed_checks_detected"] for item in invalid_scores
        ),
        "case_results": [
            {
                "form_id": item["form_id"],
                "expected_failed_checks": item["expected_failed_checks"],
                "expected_failed_checks_detected": item["expected_failed_checks_detected"],
                "all_expected_failed_checks_detected": item["all_expected_failed_checks_detected"],
                "character_accuracy": item["character_accuracy"],
                "field_exact_accuracy": item["field_exact_accuracy"],
                "row_exact_accuracy": item["row_exact_accuracy"],
                "failed_check_results": item["failed_check_results"],
                "actual_failed_checks": item["actual_failed_checks"],
            }
            for item in invalid_scores
        ],
    }

    metrics = {
        "dataset": "pre-rendered synthetic ML-7 QA fixtures",
        "as_of_date_for_validation": args.as_of.isoformat(),
        "model": str(args.model),
        "references": str(args.references),
        "policy": str(args.policy),
        "manifest_image_count": manifest["image_count"],
        "valid_image_count": manifest["valid_images"],
        "distinct_valid_forms": manifest["distinct_valid_forms"],
        "invalid_image_count": manifest["invalid_images"],
        "valid_form_metrics": valid_metrics,
        "invalid_control_metrics": invalid_metrics,
        "interpretation": (
            "The 15 valid images are three paired capture variants of each of five forms; "
            "condition scores are robustness checks, not independent-form estimates. "
            "Invalid-control results measure detection of the manifest's expected failed checks."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    case_field_errors = [error for item in scored for error in item["field_errors"]]
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = [
            "form_id", "case_type", "condition", "related_form_group", "registration_ok",
            "character_count", "correct_characters", "character_accuracy", "field_count",
            "exact_fields", "field_exact_accuracy", "trip_rows", "exact_rows",
            "row_exact_accuracy", "predicted_rows", "extra_predicted_rows",
            "expected_failed_checks", "expected_failed_checks_detected",
            "all_expected_failed_checks_detected", "actual_failed_checks",
            "auto_post_rows", "incorrect_auto_post_rows",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in scored:
            writer.writerow({
                key: json.dumps(item[key]) if isinstance(item.get(key), (list, dict)) else item.get(key)
                for key in columns
            })
    with (args.output_dir / "field_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["form_id", "field", "truth", "prediction"])
        writer.writeheader()
        writer.writerows(case_field_errors)
    valid_confusion = _aggregate(valid_scores)["character_confusion"]
    _write_confusion(args.output_dir / "valid_character_confusion.csv", valid_confusion)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
