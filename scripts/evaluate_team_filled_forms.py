"""Score the independent team-filled ML-7 forms against their hand labels."""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import DATA_DIR, OUTPUT_DIR, REPO_ROOT  # noqa: E402
from mlreader.emnist import CLASSES  # noqa: E402
from mlreader.pipeline import read_form  # noqa: E402
from mlreader.registration import PREPROCESSING_VERSION  # noqa: E402


def _single_value(rows, column, form_id):
    values = {row[column].strip() for row in rows}
    if len(values) != 1:
        raise ValueError(f"{form_id} has inconsistent {column} labels: {sorted(values)}")
    return values.pop()


def _fixed_width(value, width, column, form_id, row_number=None):
    value = value.strip()
    location = f"{form_id} row {row_number}" if row_number is not None else form_id
    if not value.isdigit() or len(value) != width:
        raise ValueError(
            f"{location} {column} label must contain exactly {width} digits: {value!r}"
        )
    return value


def _right_aligned_cells(value, width, column, form_id):
    value = value.strip()
    if not value.isdigit() or not 1 <= len(value) <= width:
        raise ValueError(
            f"{form_id} {column} label must contain 1 to {width} digits: {value!r}"
        )
    return [None] * (width - len(value)) + list(value)


def _label_fields(form_id, rows):
    employee_id = _single_value(rows, "EMPLOYEE_ID", form_id).upper()
    if len(employee_id) != 6 or not employee_id.isalnum():
        raise ValueError(f"{form_id} EMPLOYEE_ID label must contain six alphanumeric characters.")
    week_ending = datetime.strptime(
        _single_value(rows, "WEEK_ENDING", form_id), "%m/%d/%y"
    ).strftime("%m%d%y")
    total_miles = _right_aligned_cells(
        _single_value(rows, "TOTAL_MILES", form_id), 4, "TOTAL_MILES", form_id
    )

    fields = [
        ("header.employee_id", list(employee_id)),
        ("header.week_ending", list(week_ending)),
    ]
    ordered_rows = sorted(rows, key=lambda row: int(row["ROW_NUMBER"]))
    row_labels = {}
    for row in ordered_rows:
        row_number = int(row["ROW_NUMBER"])
        client = row["CLIENT"].strip().upper()
        if len(client) != 3 or not client.isalpha():
            raise ValueError(f"{form_id} row {row_number} CLIENT label must be three letters.")
        values = {
            "date": list(_fixed_width(row["DATE_MMDD"], 4, "DATE_MMDD", form_id, row_number)),
            "client": list(client),
            "odometer_start": list(
                _fixed_width(row["ODOMETER_START"], 6, "ODOMETER_START", form_id, row_number)
            ),
            "odometer_end": list(
                _fixed_width(row["ODOMETER_END"], 6, "ODOMETER_END", form_id, row_number)
            ),
            "miles": list(_fixed_width(row["MILES"], 3, "MILES", form_id, row_number)),
        }
        row_labels[row_number] = values
        for name, value in values.items():
            fields.append((f"rows.{row_number}.{name}", value))
    fields.append(("footer.total_miles", total_miles))
    return fields, row_labels


def _read_field(document, key):
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


def _prediction_cells(field, width):
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


def _field_matches(field, expected_cells):
    predicted, statuses = _prediction_cells(field, len(expected_cells))
    return all(
        status == "empty" if truth is None else status == "ok" and guess == truth
        for truth, guess, status in zip(expected_cells, predicted, statuses)
    )


def _display_cells(cells):
    return "".join("∅" if character is None else character for character in cells)


def _row_validation(document, row_number):
    row = next(
        (item for item in document.get("rows", []) if item["row_number"] == row_number),
        None,
    )
    return row.get("validation", {}) if row else {}


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--forms-dir", type=Path, default=REPO_ROOT / "examples/team_filled_forms"
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=REPO_ROOT / "examples/team_filled_forms/ground_truth.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR / "team_filled_eval"
    )
    parser.add_argument("--model", type=Path, help="Override the default reader checkpoint")
    parser.add_argument(
        "--dataset-name", default="team-filled handwritten ML-7 forms"
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 10, 5))
    args = parser.parse_args()

    with args.ground_truth.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "FORM_ID", "ROW_NUMBER", "EMPLOYEE_ID", "WEEK_ENDING", "DATE_MMDD",
            "CLIENT", "ODOMETER_START", "ODOMETER_END", "MILES", "TOTAL_MILES",
        }
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"Ground truth is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        grouped = defaultdict(list)
        for row in reader:
            grouped[row["FORM_ID"].strip()].append(row)

    if not grouped:
        raise SystemExit("No labeled forms were found.")
    model_path = args.model or DATA_DIR / "models" / "emnist_mlp.npz"
    if not model_path.is_file():
        raise SystemExit("No trained classifier is available. Run scripts/train_model.py first.")

    class_index = {label: index for index, label in enumerate(CLASSES)}
    confusion = np.zeros((len(CLASSES), len(CLASSES) + 1), dtype=np.int64)
    usable_confusion = np.zeros_like(confusion)
    field_totals = defaultdict(Counter)
    field_errors = []
    form_results = []
    form_count = 0
    registration_successes = 0
    verified_alignments = preprocessing_warning_cells = 0
    border_recovered_cells = 0
    lighting_adjusted_cells = 0
    expected_cells = usable_cells = 0
    correct_on_usable_cells = 0
    expected_blank_cells = correctly_empty_cells = 0
    correct_characters = exact_fields = total_fields = 0
    correct_digits = total_digits = correct_letters = total_letters = 0
    exact_rows = total_rows = wrong_auto_post_rows = 0
    route_counts = Counter()
    extra_predicted_rows = 0

    for form_id in sorted(grouped):
        rows = grouped[form_id]
        image_path = args.forms_dir / f"{form_id}.png"
        if not image_path.is_file():
            raise SystemExit(f"Missing labeled image: {image_path}")
        row_numbers = sorted(int(row["ROW_NUMBER"]) for row in rows)
        if row_numbers != list(range(1, len(row_numbers) + 1)):
            raise SystemExit(f"{form_id} row labels must be unique and contiguous from 1.")

        labels, row_labels = _label_fields(form_id, rows)
        document = read_form(image_path, model_path=args.model, today=args.as_of)
        if document.get("registration", {}).get("ok") and not document.get("model_available"):
            raise SystemExit("No trained classifier is available. Run scripts/train_model.py first.")
        form_count += 1
        registered = bool(document.get("registration", {}).get("ok"))
        registration_successes += registered
        alignment_verified = bool(document.get("registration", {}).get("template_alignment", {}).get("verified"))
        verified_alignments += alignment_verified
        warning_cells = document.get("preprocessing", {}).get("cells_requiring_review", 0)
        preprocessing_warning_cells += warning_cells
        recovered_cells = document.get("preprocessing", {}).get("border_recovered_cells", 0)
        border_recovered_cells += recovered_cells
        adjusted_cells = document.get("preprocessing", {}).get("lighting_adjusted_cells", 0)
        lighting_adjusted_cells += adjusted_cells
        predicted_rows = {item["row_number"]: item for item in document.get("rows", [])}
        extra_rows = len(set(predicted_rows) - set(row_labels))
        extra_predicted_rows += extra_rows

        form_exact_fields = 0
        form_total_fields = 0
        form_exact_rows = 0
        form_routes = Counter()
        row_exact_by_number = {}
        for row_number, truth_values in row_labels.items():
            field_matches = []
            for name, expected in truth_values.items():
                key = f"rows.{row_number}.{name}"
                field_matches.append(
                    _field_matches(_read_field(document, key), expected)
                )
            row_exact = all(field_matches)
            row_exact_by_number[row_number] = row_exact
            total_rows += 1
            exact_rows += row_exact
            validation = _row_validation(document, row_number)
            route = validation.get("route", "not_read")
            route_counts[route] += 1
            form_routes[route] += 1
            if route == "auto_post" and not row_exact:
                wrong_auto_post_rows += 1

        for key, expected in labels:
            actual_field = _read_field(document, key)
            predicted, statuses = _prediction_cells(actual_field, len(expected))
            matched = _field_matches(actual_field, expected)
            total_fields += 1
            form_total_fields += 1
            exact_fields += matched
            form_exact_fields += matched
            field_type = key.split(".")[-1]
            field_totals[field_type]["total"] += 1
            field_totals[field_type]["exact"] += matched
            if key.startswith("rows."):
                _, row_text, _ = key.split(".")
                row_number = int(row_text)
            else:
                row_number = ""
            if not matched:
                field_errors.append(
                    {
                        "form_id": form_id,
                        "row_number": row_number,
                        "field": key.split(".")[-1],
                        "truth": _display_cells(expected),
                        "prediction": "".join(predicted),
                    }
                )

            for position, truth_char in enumerate(expected):
                status = statuses[position]
                predicted_char = predicted[position]
                if truth_char is None:
                    expected_blank_cells += 1
                    correctly_empty_cells += status == "empty"
                    continue
                if status != "ok":
                    predicted_char = "?"
                is_correct = truth_char == predicted_char
                correct_characters += is_correct
                if truth_char.isdigit():
                    total_digits += 1
                    correct_digits += is_correct
                else:
                    total_letters += 1
                    correct_letters += is_correct
                predicted_index = class_index.get(predicted_char, len(CLASSES))
                confusion[class_index[truth_char], predicted_index] += 1

                expected_cells += 1
                usable_cells += status == "ok"
                if status == "ok":
                    correct_on_usable_cells += is_correct
                    usable_confusion[class_index[truth_char], predicted_index] += 1

        for row_number in row_labels:
            form_exact_rows += row_exact_by_number[row_number]
        form_results.append(
            {
                "form_id": form_id,
                "registration_ok": registered,
                "template_alignment_verified": alignment_verified,
                "preprocessing_warning_cells": warning_cells,
                "border_recovered_cells": recovered_cells,
                "lighting_adjusted_cells": adjusted_cells,
                "expected_rows": len(row_labels),
                "predicted_rows": len(predicted_rows),
                "extra_predicted_rows": extra_rows,
                "exact_rows": form_exact_rows,
                "row_exact_accuracy": _rate(form_exact_rows, len(row_labels)),
                "exact_fields": form_exact_fields,
                "fields": form_total_fields,
                "field_exact_accuracy": _rate(form_exact_fields, form_total_fields),
                "validation_outcome": document.get("validation", {}).get("outcome", "not_read"),
                "auto_post_rows": form_routes["auto_post"],
                "needs_review_rows": form_routes["needs_review"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": args.dataset_name,
        "model": str(args.model or DATA_DIR / "models/emnist_mlp.npz"),
        "preprocessing_version": PREPROCESSING_VERSION,
        "template_alignment_verified_forms": verified_alignments,
        "preprocessing_warning_cells": preprocessing_warning_cells,
        "border_recovered_cells": border_recovered_cells,
        "lighting_adjusted_cells": lighting_adjusted_cells,
        "forms": form_count,
        "as_of_date_for_validation": args.as_of.isoformat(),
        "registration_successes": registration_successes,
        "registration_success_rate": _rate(registration_successes, form_count),
        "trip_rows": total_rows,
        "characters": expected_cells,
        "correct_characters": correct_characters,
        "character_accuracy": _rate(correct_characters, expected_cells),
        "digit_characters": total_digits,
        "correct_digit_characters": correct_digits,
        "digit_accuracy": _rate(correct_digits, total_digits),
        "capital_letter_characters": total_letters,
        "correct_capital_letter_characters": correct_letters,
        "capital_letter_accuracy": _rate(correct_letters, total_letters),
        "fields": total_fields,
        "exact_fields": exact_fields,
        "field_exact_accuracy": _rate(exact_fields, total_fields),
        "exact_rows": exact_rows,
        "row_exact_accuracy": _rate(exact_rows, total_rows),
        "exact_forms": sum(item["field_exact_accuracy"] == 1 for item in form_results),
        "cell_extraction_successes": usable_cells,
        "expected_cells": expected_cells,
        "cell_extraction_success_rate": _rate(usable_cells, expected_cells),
        "cell_extraction_failures": expected_cells - usable_cells,
        "expected_blank_cells": expected_blank_cells,
        "correctly_empty_blank_cells": correctly_empty_cells,
        "correct_characters_on_usable_cells": correct_on_usable_cells,
        "character_accuracy_on_usable_cells": _rate(correct_on_usable_cells, usable_cells),
        "row_validation_routes": dict(route_counts),
        "incorrect_auto_post_rows": wrong_auto_post_rows,
        "extra_predicted_rows": extra_predicted_rows,
        "field_metrics": {
            name: {
                "exact": totals["exact"],
                "total": totals["total"],
                "exact_accuracy": _rate(totals["exact"], totals["total"]),
            }
            for name, totals in sorted(field_totals.items())
        },
        "top_character_errors_on_usable_cells": [],
    }

    errors = []
    for truth_index, row in enumerate(usable_confusion):
        for predicted_index, count in enumerate(row):
            if count and predicted_index != truth_index:
                guessed = CLASSES[predicted_index] if predicted_index < len(CLASSES) else "?"
                errors.append((int(count), CLASSES[truth_index], guessed))
    metrics["top_character_errors_on_usable_cells"] = [
        {"truth": truth, "prediction": predicted, "count": count}
        for count, truth, predicted in sorted(errors, reverse=True)[:10]
    ]

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.output_dir / "per_form.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(form_results[0]))
        writer.writeheader()
        writer.writerows(form_results)
    with (args.output_dir / "field_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["form_id", "row_number", "field", "truth", "prediction"]
        )
        writer.writeheader()
        writer.writerows(field_errors)
    with (args.output_dir / "character_confusion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["truth/prediction", *CLASSES, "?"])
        for label, row in zip(CLASSES, confusion):
            writer.writerow([label, *row.tolist()])
    with (args.output_dir / "usable_character_confusion.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["truth/prediction", *CLASSES, "?"])
        for label, row in zip(CLASSES, usable_confusion):
            writer.writerow([label, *row.tolist()])

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
