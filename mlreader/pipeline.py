"""End-to-end photo read: registration, extraction, prediction, assembly, checks."""

import json
from pathlib import Path

import cv2

from mlreader import CONFIG_DIR, DATA_DIR
from mlreader.classifier import EMNISTMLP, ModelNotAvailable
from mlreader.fields import assemble_field
from mlreader.layout import FIELDS, MAX_ROWS
from mlreader.registration import PREPROCESSING_VERSION, extract_cells, register_page
from mlreader.validation import validate_document


def _load_json(source, default_path):
    if source is None:
        source = default_path
    if isinstance(source, dict):
        return source
    return json.loads(Path(source).read_text())


def _predict_cell(cell, model):
    evidence = {
        "source_rect": list(cell.rect),
        "crop_rect": list(cell.crop_rect) if cell.crop_rect else None,
        "border_recovery": cell.border_recovery,
        "ink_pixels": cell.ink_pixels,
        "preprocessing_issues": list(cell.preprocessing_issues),
    }
    if cell.extraction_status != "ok":
        return {
            **evidence,
            "character": None,
            "confidence": None,
            "class_probabilities": {},
            "top_alternatives": [],
            "extraction_status": cell.extraction_status,
            "failure_kind": "cell_extraction",
        }
    if model is None:
        return {
            **evidence,
            "character": None,
            "confidence": None,
            "class_probabilities": {},
            "top_alternatives": [],
            "extraction_status": "ok",
            "failure_kind": "recognition_model_unavailable",
        }
    prediction = model.predict(cell.normalized, cell.allowed)
    prediction.update(evidence)
    prediction["extraction_status"] = "ok"
    prediction["failure_kind"] = None if prediction.get("character") else "recognition_failure"
    return prediction


def _save_crop(cell, evidence_dir):
    if evidence_dir is None:
        return None
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    safe = cell.field.replace(".", "_")
    path = evidence_dir / f"{safe}_cell_{cell.position + 1}.png"
    cv2.imwrite(str(path), cell.image)
    return str(path)


def read_form(
    image_source,
    model_path=None,
    references=None,
    policy=None,
    evidence_dir=None,
    today=None,
):
    """Read one supplied log image. Unsupported images are returned for review."""
    registration = register_page(image_source)
    if not registration.ok:
        return {
            "registration": {
                "ok": False,
                "method": registration.method,
                "corners": registration.corners,
                "reason": registration.reason,
            },
            "model_available": False,
            "field_extraction_status": "not_run",
            "fields": {},
            "rows": [],
            "status": "needs_review",
            "review_reason": registration.reason,
        }

    if model_path is None:
        model_path = DATA_DIR / "models" / "emnist_mlp.npz"
    try:
        model = EMNISTMLP(model_path)
    except ModelNotAvailable:
        model = None

    references = _load_json(references, CONFIG_DIR / "reference_data.json")
    policy = _load_json(policy, CONFIG_DIR / "reader_policy.json")
    crops = extract_cells(registration.image)
    field_reads = {}
    field_predictions = {}
    evidence = {}
    for field_name, field_crops in crops.items():
        predictions = []
        for cell in field_crops:
            predictions.append(_predict_cell(cell, model))
            evidence[f"{field_name}.{cell.position}"] = _save_crop(cell, evidence_dir)
        field_predictions[field_name] = predictions
        field_reads[field_name] = assemble_field(field_name, predictions)

    header = {
        "employee_id": field_reads["header.employee_id"],
        "week_ending": field_reads["header.week_ending"],
    }
    rows = []
    active_row_numbers = set()
    for row_number in range(1, MAX_ROWS + 1):
        names = ("date", "client", "odometer_start", "odometer_end", "miles")
        row_fields = {name: field_reads[f"rows.{row_number}.{name}"] for name in names}
        row_crops = [crops[f"rows.{row_number}.{name}"] for name in names]
        if all(cell.extraction_status == "empty" for field_crops in row_crops for cell in field_crops):
            continue
        active_row_numbers.add(row_number)
        rows.append({"row_number": row_number, "fields": row_fields})
    expected_fields = {"header.employee_id", "header.week_ending", "footer.total_miles"}
    for row_number in active_row_numbers:
        expected_fields.update(f"rows.{row_number}.{name}" for name in ("date", "client", "odometer_start", "odometer_end", "miles"))
    extraction_failures = []
    recognition_failures = []
    for field_name in sorted(expected_fields):
        for cell, prediction in zip(crops[field_name], field_predictions[field_name]):
            if cell.extraction_status != "ok":
                extraction_failures.append(
                    {
                        "field": field_name,
                        "position": cell.position,
                        "status": cell.extraction_status,
                        "reason": cell.extraction_reason,
                    }
                )
            elif prediction.get("failure_kind"):
                recognition_failures.append(
                    {
                        "field": field_name,
                        "position": cell.position,
                        "failure_kind": prediction["failure_kind"],
                    }
                )
    footer = {"total_miles": field_reads["footer.total_miles"]}
    document = {
        "registration": {
            "ok": True,
            "method": registration.method,
            "corners": registration.corners,
            "reason": None,
            "template_alignment": registration.template_alignment,
        },
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "cells_requiring_review": sum(
                bool(cell.preprocessing_issues) for field_name in expected_fields for cell in crops[field_name]
            ),
            "border_recovered_cells": sum(
                bool(cell.border_recovery and cell.border_recovery.get("applied"))
                for field_name in expected_fields for cell in crops[field_name]
            ),
            "review_reasons": (
                [] if registration.template_alignment and registration.template_alignment.get("verified")
                else ["Printed form alignment could not be verified."]
            ),
        },
        "model_available": model is not None,
        "model_path": str(model_path),
        "field_extraction_status": "ok" if not extraction_failures else "partial",
        "cell_extraction_failures": extraction_failures,
        "recognition_failures": recognition_failures,
        "evidence": evidence,
        "header": header,
        "rows": rows,
        "footer": footer,
    }
    validate_document(document, references, policy, today=today)
    document["status"] = document["validation"]["outcome"]
    if model is None:
        document["model_note"] = "No trained model checkpoint was found; non-empty cells are routed as recognition failures."
    return document
