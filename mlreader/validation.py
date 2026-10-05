"""Deterministic ML-7 validation and post-or-review routing."""

from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


def _value(field):
    return field.get("value") if field else None


def _read_iso_week_ending(raw):
    if not raw or len(raw) != 6 or not raw.isdigit():
        return None
    try:
        return datetime.strptime(raw, "%m%d%y").date()
    except ValueError:
        return None


def _read_trip_date(raw, week_ending):
    if not raw or len(raw) != 4 or not raw.isdigit() or week_ending is None:
        return None
    start = week_ending - timedelta(days=6)
    candidates = []
    for year in (week_ending.year - 1, week_ending.year, week_ending.year + 1):
        try:
            candidate = date(year, int(raw[:2]), int(raw[2:]))
        except ValueError:
            continue
        if start <= candidate <= week_ending:
            candidates.append(candidate)
    return candidates[0] if len(candidates) == 1 else None


def _confidence_check(field, minimum, critical_minimum=None, critical_positions=()):
    if field and field.get("correction"):
        return True, "manually corrected by reviewer"
    if not field or not field.get("complete"):
        return False, "field is incomplete"
    characters = field.get("characters", [])
    for character in characters:
        confidence = character.get("confidence")
        position = character.get("position")
        threshold = critical_minimum if position in critical_positions and critical_minimum is not None else minimum
        if confidence is None:
            return False, f"character {position + 1} has no model confidence"
        if confidence < threshold:
            return False, f"character {position + 1} confidence {confidence:.3f} is below {threshold:.3f}"
    return True, None


def _money(miles, rate):
    try:
        amount = Decimal(int(miles)) * Decimal(str(rate))
    except (TypeError, ValueError, InvalidOperation):
        return None
    return str(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def validate_document(document, references, policy, today=None):
    """Attach field checks and a conservative route to every non-empty row."""
    today = today or date.today()
    header = document["header"]
    employee_field = header["employee_id"]
    week_field = header["week_ending"]
    employee = _value(employee_field)
    week_raw = _value(week_field)
    week_date = _read_iso_week_ending(week_raw)
    employees = set(references.get("employees", []))
    header_checks = {
        "employee_id_format": bool(employee and len(employee) == 6 and employee[:2].isalpha() and employee[:2].isupper() and employee[2:].isdigit()),
        "employee_exists": employee in employees,
        "week_ending_format": week_date is not None,
        "week_ending_is_sunday": week_date is not None and week_date.weekday() == 6,
        "week_ending_within_last_60_days": week_date is not None and today - timedelta(days=60) <= week_date <= today,
    }
    if week_date is not None:
        week_field["normalized_value"] = week_date.isoformat()
    else:
        week_field["normalized_value"] = None

    global_issues = [name for name, passed in header_checks.items() if not passed]
    minimum = float(policy.get("min_character_confidence", 1.01))
    critical_minimum = float(policy.get("critical_odometer_digit_min_confidence", minimum))
    critical_positions = set(policy.get("critical_odometer_positions", [0, 1, 2]))
    header_confidence = {}
    for name, field in (("employee_id", employee_field), ("week_ending", week_field)):
        ok, reason = _confidence_check(field, minimum)
        header_confidence[name] = {"passed": ok, "reason": reason}
        if not ok:
            global_issues.append(f"{name}_confidence")

    previous_end = None
    row_results = []
    total_values = []
    for row in document.get("rows", []):
        fields = row["fields"]
        row_number = row["row_number"]
        trip_field = fields["date"]
        client_field = fields["client"]
        start_field = fields["odometer_start"]
        end_field = fields["odometer_end"]
        miles_field = fields["miles"]
        raw_trip_date = _value(trip_field)
        trip_date = _read_trip_date(raw_trip_date, week_date)
        if trip_date is not None:
            trip_field["normalized_value"] = trip_date.isoformat()

        employee_schedule = references.get("visit_schedule", {}).get(employee or "", {})
        allowed_clients = employee_schedule.get(trip_date.isoformat(), []) if trip_date else []
        try:
            start_odo = int(_value(start_field)) if _value(start_field) else None
            end_odo = int(_value(end_field)) if _value(end_field) else None
            miles = int(_value(miles_field)) if _value(miles_field) else None
        except ValueError:
            start_odo = end_odo = miles = None
        if miles is not None:
            total_values.append(miles)

        checks = {
            "trip_date_format_and_in_week": trip_date is not None,
            "client_on_visit_schedule": _value(client_field) in allowed_clients,
            "odometer_end_after_start": start_odo is not None and end_odo is not None and end_odo > start_odo,
            "odometer_continuity": start_odo is not None and (previous_end is None or start_odo >= previous_end),
            "miles_equals_odometer_difference": start_odo is not None and end_odo is not None and miles is not None and miles == end_odo - start_odo,
        }
        confidence = {}
        for name, field in (
            ("date", trip_field),
            ("client", client_field),
            ("odometer_start", start_field),
            ("odometer_end", end_field),
            ("miles", miles_field),
        ):
            ok, reason = _confidence_check(
                field,
                minimum,
                critical_minimum if name in {"odometer_start", "odometer_end"} else None,
                critical_positions,
            )
            confidence[name] = {"passed": ok, "reason": reason}

        issues = [name for name, passed in checks.items() if not passed]
        if policy.get("route_log_when_header_invalid", True):
            issues.extend(f"header_{name}" for name, passed in header_checks.items() if not passed)
            issues.extend(name for name, result in header_confidence.items() if not result["passed"])
        issues.extend(f"{name}_confidence" for name, result in confidence.items() if not result["passed"])
        row_result = {
            "row_number": row_number,
            "checks": checks,
            "confidence_checks": confidence,
            "issues": issues,
            "miles_value": miles,
            "reimbursement": _money(miles, policy.get("reimbursement_rate", 0.62)),
            "previous_end_odometer": previous_end,
        }
        row_results.append(row_result)
        row["validation"] = row_result
        if end_odo is not None:
            previous_end = end_odo

    total_field = document["footer"]["total_miles"]
    total_raw = _value(total_field)
    try:
        written_total = int(total_raw) if total_raw else None
    except ValueError:
        written_total = None
    calculated_total = sum(total_values) if len(total_values) == len(row_results) and row_results else None
    total_check = written_total is not None and calculated_total is not None and written_total == calculated_total
    total_field["normalized_value"] = written_total
    total_confidence_ok, total_confidence_reason = _confidence_check(total_field, minimum)
    total_check_result = {
        "passed": total_check,
        "written_total": written_total,
        "sum_of_written_row_miles": calculated_total,
        "reason": None if total_check else "Total miles must equal the sum of all active row miles.",
    }
    if not total_confidence_ok:
        total_check_result["confidence_issue"] = total_confidence_reason
    if not row_results:
        global_issues.append("no_trip_rows_found")
    if policy.get("route_rows_when_total_mismatch", True) and not total_check:
        for row_result in row_results:
            row_result["issues"].append("weekly_total_matches")
    header_or_total_corrected = bool(employee_field.get("correction") or week_field.get("correction") or total_field.get("correction"))
    for row_result in row_results:
        if not total_confidence_ok:
            row_result["issues"].append("total_miles_confidence")
        row_result["route"] = (
            "auto_post"
            if policy.get("auto_post_enabled", False) and not row_result["issues"]
            else "needs_review"
        )
        row_number = row_result["row_number"]
        corrected_row = any(
            field.get("correction")
            for row in document.get("rows", [])
            if row["row_number"] == row_number
            for field in row.get("fields", {}).values()
        )
        if not row_result["issues"] and (header_or_total_corrected or corrected_row):
            row_result["route"] = "reviewed"
        row_result["route_reasons"] = row_result["issues"][:]
    for row in document.get("rows", []):
        match = next(result for result in row_results if result["row_number"] == row["row_number"])
        row["validation"] = match
    document["validation"] = {
        "header_checks": header_checks,
        "header_confidence_checks": header_confidence,
        "global_issues": global_issues,
        "weekly_total": total_check_result,
        "outcome": "auto_post" if row_results and all(r["route"] == "auto_post" for r in row_results) else "needs_review",
        "policy_name": policy.get("policy_name", "unnamed"),
        "reimbursement_rate": policy.get("reimbursement_rate", 0.62),
    }
    return document
