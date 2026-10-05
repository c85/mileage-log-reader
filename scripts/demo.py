"""Run the Form ML-7 reader on an image supplied now or on the bundled example."""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import CONFIG_DIR, OUTPUT_DIR  # noqa: E402
from mlreader.pipeline import read_form  # noqa: E402
from mlreader.review import append_correction_audit, apply_correction  # noqa: E402
from mlreader.validation import validate_document  # noqa: E402


def _parse_correction(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Corrections use FIELD.PATH=VALUE, for example rows.5.miles=047")
    field, corrected = value.split("=", 1)
    if not field or not corrected:
        raise argparse.ArgumentTypeError("Both a field path and corrected value are required.")
    return field, corrected


def _show_field(field):
    confidence = field.get("confidence")
    suffix = f" (min confidence {confidence:.3f})" if confidence is not None else " (confidence unavailable)"
    if field.get("correction"):
        suffix += f" [corrected from {field['correction']['original_value']}]"
    return f"{field.get('raw_value', '')} -> {field.get('value') or 'UNREADABLE'}{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", type=Path, default=Path("examples/figure1_clean_scan.png"))
    parser.add_argument("--model", type=Path, help="Override data/models/emnist_mlp.npz")
    parser.add_argument("--policy", type=Path, default=CONFIG_DIR / "reader_policy.json")
    parser.add_argument("--references", type=Path, default=CONFIG_DIR / "reference_data.json")
    parser.add_argument("--output-dir", type=Path, help="Evidence and result folder")
    parser.add_argument("--as-of", type=date.fromisoformat, help="Validation date YYYY-MM-DD")
    parser.add_argument("--correction", action="append", type=_parse_correction, default=[])
    parser.add_argument("--reason", help="Required reason for each --correction")
    args = parser.parse_args()
    if args.correction and not args.reason:
        parser.error("--reason is required whenever --correction is used.")

    run_dir = args.output_dir or OUTPUT_DIR / "demo" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    references = json.loads(args.references.read_text())
    policy = json.loads(args.policy.read_text())
    document = read_form(
        args.image,
        model_path=args.model,
        references=references,
        policy=policy,
        evidence_dir=run_dir / "crops",
        today=args.as_of,
    )
    corrections = []
    for field_path, corrected_value in args.correction:
        corrections.append(apply_correction(document, field_path, corrected_value, args.reason))
    if corrections and document.get("registration", {}).get("ok"):
        validate_document(document, references, policy, today=args.as_of)
        for correction in corrections:
            append_correction_audit(run_dir / "correction_audit.jsonl", correction, document, correction["field_path"])
        document["status"] = document["validation"]["outcome"]

    output_path = run_dir / "result.json"
    output_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"Image: {args.image}")
    print(f"Registration: {document.get('registration', {}).get('method')} ({document.get('registration', {}).get('ok')})")
    if not document.get("registration", {}).get("ok"):
        print(f"Needs review: {document.get('review_reason')}")
        print(f"Evidence: {output_path}")
        return 0
    print(f"Classifier available: {document.get('model_available')}")
    if not document.get("model_available"):
        print("Train the project model with scripts/download_emnist.py and scripts/train_model.py to read characters.")
    header = document["header"]
    print(f"Employee ID: {_show_field(header['employee_id'])}")
    print(f"Week ending: {_show_field(header['week_ending'])} -> {header['week_ending'].get('normalized_value')}")
    print("\nRows")
    for row in document["rows"]:
        fields = row["fields"]
        validation = row.get("validation", {})
        confidences = [
            field.get("confidence")
            for field in fields.values()
            if field.get("confidence") is not None
        ]
        min_confidence = f"{min(confidences):.3f}" if confidences else "unavailable"
        checks = ", ".join(name for name, passed in validation.get("checks", {}).items() if not passed) or "all business checks passed"
        reasons = ", ".join(validation.get("route_reasons", [])) or checks
        print(
            f"{row['row_number']:>2}  date={fields['date'].get('value') or fields['date']['raw_value']} "
            f"client={fields['client'].get('value') or fields['client']['raw_value']} "
            f"odo={fields['odometer_start'].get('value') or fields['odometer_start']['raw_value']}->"
            f"{fields['odometer_end'].get('value') or fields['odometer_end']['raw_value']} "
            f"miles={fields['miles'].get('value') or fields['miles']['raw_value']} "
            f"min-confidence={min_confidence} "
            f"calculated-not-posted=${validation.get('reimbursement') or 'n/a'} "
            f"route={validation.get('route', 'needs_review')} ({reasons})"
        )
    total = document["validation"]["weekly_total"]
    print(f"\nWeekly total: written={total['written_total']} sum-of-rows={total['sum_of_written_row_miles']} passed={total['passed']}")
    print(f"Overall: {document['validation']['outcome']}; extraction failures={len(document.get('cell_extraction_failures', []))}; recognition failures={len(document.get('recognition_failures', []))}")
    if corrections:
        print(f"Correction audit: {run_dir / 'correction_audit.jsonl'}")
    print(f"Evidence: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
