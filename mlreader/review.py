"""Reviewer correction helpers with an append-only JSONL audit record."""

import json
from datetime import datetime, timezone
from pathlib import Path


def _locate_field(document, field_path):
    parts = field_path.split(".")
    if len(parts) == 2 and parts[0] == "header" and parts[1] in document["header"]:
        return document["header"][parts[1]]
    if len(parts) == 2 and parts[0] == "footer" and parts[1] in document["footer"]:
        return document["footer"][parts[1]]
    if len(parts) == 3 and parts[0] == "rows":
        row_number = int(parts[1])
        field = parts[2]
        row = next((row for row in document["rows"] if row["row_number"] == row_number), None)
        if row is not None and field in row["fields"]:
            return row["fields"][field]
    raise ValueError(f"Unknown field path: {field_path}")


def apply_correction(document, field_path, corrected_value, reason):
    """Replace the working value while retaining the original recognition."""
    if not reason or not reason.strip():
        raise ValueError("A correction reason is required for the audit trail.")
    field = _locate_field(document, field_path)
    original = field.get("value") if field.get("value") is not None else field.get("raw_value")
    field["correction"] = {
        "original_value": original,
        "corrected_value": str(corrected_value),
        "reason": reason.strip(),
        "applied_at": datetime.now(timezone.utc).isoformat(),
    }
    field["value"] = str(corrected_value)
    field["complete"] = True
    field["normalization_error"] = None
    return {"field_path": field_path, **field["correction"]}


def append_correction_audit(path, correction, document, field_path):
    """Append the correction and its post-correction validation evidence."""
    parts = field_path.split(".")
    row_result = None
    field = _locate_field(document, field_path)
    if len(parts) == 3 and parts[0] == "rows":
        row = next(row for row in document["rows"] if row["row_number"] == int(parts[1]))
        row_result = row.get("validation")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "field_path": field_path,
        "original_value": correction["original_value"],
        "corrected_value": correction["corrected_value"],
        "reason": correction["reason"],
        "review_outcome": "reviewed_ready_for_manual_entry" if row_result and not row_result.get("issues") else "needs_review",
        "row_validation": row_result,
        "source_crops": [
            document.get("evidence", {}).get(f"{field_path}.{index}")
            for index in range(len(field.get("characters", [])))
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
