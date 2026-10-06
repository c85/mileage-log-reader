"""Field assembly from per-cell predictions, preserving raw evidence."""

from mlreader.layout import FIELD_KIND, FIELD_LENGTHS


def assemble_field(field_name, predictions):
    expected = FIELD_LENGTHS[field_name]
    if len(predictions) != expected:
        raise ValueError(f"{field_name} expects {expected} cells, got {len(predictions)}")
    characters = []
    for position, prediction in enumerate(predictions):
        prediction = prediction or {}
        char = prediction.get("character") or "?"
        characters.append(
            {
                "position": position,
                "character": char,
                "confidence": prediction.get("confidence"),
                "class_probabilities": prediction.get("class_probabilities", {}),
                "top_alternatives": prediction.get("top_alternatives", []),
                "extraction_status": prediction.get("extraction_status", "ok"),
                "failure_kind": prediction.get("failure_kind"),
                "source_rect": prediction.get("source_rect"),
                "crop_rect": prediction.get("crop_rect"),
                "border_recovery": prediction.get("border_recovery"),
                "ink_pixels": prediction.get("ink_pixels"),
                "preprocessing_issues": prediction.get("preprocessing_issues", []),
            }
        )
    raw = "".join(character["character"] for character in characters)
    complete = "?" not in raw
    confidences = [c["confidence"] for c in characters if c["confidence"] is not None]
    confidence = min(confidences) if complete and len(confidences) == expected else None
    kind = FIELD_KIND[field_name]
    normalized = raw if complete else None
    normalization_error = None
    if complete and kind == "week_ending":
        if not raw.isdigit() or len(raw) != 6:
            normalized, normalization_error = None, "Week ending must contain six digits (MMDDYY)."
    elif complete and kind == "trip_date":
        if not raw.isdigit() or len(raw) != 4:
            normalized, normalization_error = None, "Trip date must contain four digits (MMDD)."
    elif complete and kind in {"odometer", "miles", "total_miles"}:
        if not raw.isdigit():
            normalized, normalization_error = None, "This field accepts digits only."
    elif complete and kind == "client_code":
        if not raw.isalpha() or not raw.isupper():
            normalized, normalization_error = None, "Client code must contain capital letters only."
    elif complete and kind == "employee_id":
        if not raw[:2].isalpha() or not raw[:2].isupper() or not raw[2:].isdigit():
            normalized, normalization_error = None, "Employee ID must use two capital letters followed by four digits."

    return {
        "field": field_name,
        "kind": kind,
        "raw_value": raw,
        "value": normalized,
        "normalized_value": None,
        "expected_characters": expected,
        "characters": characters,
        "confidence": confidence,
        "complete": complete,
        "normalization_error": normalization_error,
        "correction": None,
    }
