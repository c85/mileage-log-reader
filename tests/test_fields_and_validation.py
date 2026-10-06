"""Regression coverage for field assembly, checks, and routing (synthetic data only)."""

import unittest
from datetime import date

from mlreader.fields import assemble_field
from mlreader.review import apply_correction
from mlreader.registration import BORDER_RECOVERY_WARNING
from mlreader.validation import validate_document


def read_field(name, value, confidence=0.99):
    return assemble_field(
        name,
        [
            {"character": char, "confidence": confidence, "top_alternatives": [{"character": char, "confidence": confidence}], "extraction_status": "ok"}
            for char in value
        ],
    )


def valid_document(miles="025", client="KMR", employee="RC5107", week="092726", start="048213", end="048238", total="0025", confidence=0.99, trip_date="0921"):
    return {
        "header": {
            "employee_id": read_field("header.employee_id", employee, confidence),
            "week_ending": read_field("header.week_ending", week, confidence),
        },
        "rows": [
            {
                "row_number": 1,
                "fields": {
                    "date": read_field("rows.1.date", trip_date, confidence),
                    "client": read_field("rows.1.client", client, confidence),
                    "odometer_start": read_field("rows.1.odometer_start", start, confidence),
                    "odometer_end": read_field("rows.1.odometer_end", end, confidence),
                    "miles": read_field("rows.1.miles", miles, confidence),
                },
            }
        ],
        "footer": {"total_miles": read_field("footer.total_miles", total, confidence)},
    }


REFERENCES = {
    "employees": ["RC5107"],
    "visit_schedule": {"RC5107": {"2026-09-21": ["KMR"]}},
}
POLICY = {
    "policy_name": "test-policy",
    "auto_post_enabled": True,
    "min_character_confidence": 0.80,
    "critical_odometer_digit_min_confidence": 0.95,
    "critical_odometer_positions": [0, 1, 2],
    "reimbursement_rate": 0.62,
    "route_log_when_header_invalid": True,
    "route_rows_when_total_mismatch": True,
}


class FieldAssemblyTests(unittest.TestCase):
    def test_assembles_and_preserves_character_confidence(self):
        field = read_field("rows.1.odometer_start", "048213")
        self.assertEqual(field["raw_value"], "048213")
        self.assertEqual(field["value"], "048213")
        self.assertEqual(field["confidence"], 0.99)
        self.assertEqual(len(field["characters"]), 6)

    def test_extraction_and_recognition_failures_remain_distinct(self):
        field = assemble_field(
            "rows.1.odometer_start",
            [
                {"character": None, "confidence": None, "extraction_status": "empty", "failure_kind": "cell_extraction"},
                {"character": None, "confidence": None, "extraction_status": "ok", "failure_kind": "recognition_failure"},
                *[{"character": c, "confidence": 0.9, "extraction_status": "ok"} for c in "8213"],
            ],
        )
        self.assertEqual(field["characters"][0]["failure_kind"], "cell_extraction")
        self.assertEqual(field["characters"][1]["failure_kind"], "recognition_failure")
        self.assertEqual(field["raw_value"], "??8213")

    def test_field_length_must_match_the_form_map(self):
        with self.assertRaises(ValueError):
            assemble_field("rows.1.odometer_start", [{"character": "0"}] * 5)


class ReimbursementRuleTests(unittest.TestCase):
    def validate(self, document, policy=None):
        return validate_document(document, REFERENCES, policy or POLICY, today=date(2026, 9, 28))

    def test_valid_synthetic_row_can_auto_post(self):
        document = self.validate(valid_document())
        row = document["rows"][0]["validation"]
        self.assertEqual(row["route"], "auto_post")
        self.assertEqual(row["reimbursement"], "15.50")

    def test_miles_disagreement_routes_to_review(self):
        document = self.validate(valid_document(miles="024", total="0024"))
        row = document["rows"][0]["validation"]
        self.assertFalse(row["checks"]["miles_equals_odometer_difference"])
        self.assertEqual(row["route"], "needs_review")

    def test_bad_week_day_routes_to_review(self):
        document = self.validate(valid_document(week="092626"))
        self.assertFalse(document["validation"]["header_checks"]["week_ending_is_sunday"])
        self.assertEqual(document["rows"][0]["validation"]["route"], "needs_review")

    def test_employee_pattern_and_trip_date_window_are_enforced(self):
        bad_employee = self.validate(valid_document(employee="R15107"))
        self.assertFalse(bad_employee["validation"]["header_checks"]["employee_id_format"])
        outside_week = self.validate(valid_document(trip_date="0930"))
        self.assertFalse(outside_week["rows"][0]["validation"]["checks"]["trip_date_format_and_in_week"])

    def test_unknown_employee_and_unscheduled_client_route_to_review(self):
        document = self.validate(valid_document(employee="ZZ9999", client="AVT"))
        self.assertFalse(document["validation"]["header_checks"]["employee_exists"])
        self.assertFalse(document["rows"][0]["validation"]["checks"]["client_on_visit_schedule"])
        self.assertEqual(document["rows"][0]["validation"]["route"], "needs_review")

    def test_odometer_continuity_and_week_total_are_checked(self):
        document = valid_document(start="048212", end="048238", miles="026", total="0026")
        second = valid_document(start="048237", end="048250", miles="013", total="0039")["rows"][0]
        second["row_number"] = 2
        for field_name, field in second["fields"].items():
            field["field"] = f"rows.2.{field_name}"
        document["rows"].append(second)
        document["footer"]["total_miles"] = read_field("footer.total_miles", "0038")
        document = self.validate(document)
        self.assertFalse(document["rows"][1]["validation"]["checks"]["odometer_continuity"])
        self.assertFalse(document["validation"]["weekly_total"]["passed"])

    def test_end_odometer_must_exceed_start(self):
        document = self.validate(valid_document(start="048238", end="048213", miles="000", total="0000"))
        checks = document["rows"][0]["validation"]["checks"]
        self.assertFalse(checks["odometer_end_after_start"])
        self.assertFalse(checks["miles_equals_odometer_difference"])

    def test_critical_odometer_confidence_uses_configured_threshold(self):
        document = self.validate(valid_document(confidence=0.90))
        row = document["rows"][0]["validation"]
        self.assertFalse(row["confidence_checks"]["odometer_start"]["passed"])
        self.assertEqual(row["route"], "needs_review")

    def test_correction_keeps_original_read_and_reason(self):
        document = valid_document(miles="024", total="0024")
        correction = apply_correction(document, "rows.1.miles", "025", "Compared with the two odometer reads")
        self.assertEqual(correction["original_value"], "024")
        self.assertEqual(document["rows"][0]["fields"]["miles"]["value"], "025")
        self.assertEqual(document["rows"][0]["fields"]["miles"]["raw_value"], "024")

    def test_high_confidence_does_not_override_uncertain_preprocessing(self):
        document = valid_document()
        character = document["rows"][0]["fields"]["miles"]["characters"][0]
        character["preprocessing_issues"] = ["Printed cell borders could not all be verified."]
        document = self.validate(document)
        confidence = document["rows"][0]["validation"]["confidence_checks"]["miles"]
        self.assertFalse(confidence["passed"])
        self.assertIn("borders", confidence["reason"])
        self.assertEqual(document["rows"][0]["validation"]["route"], "needs_review")

    def test_high_confidence_does_not_override_failed_extraction(self):
        document = valid_document()
        document["rows"][0]["fields"]["miles"]["characters"][0]["extraction_status"] = "unreadable"
        document = self.validate(document)
        self.assertEqual(document["rows"][0]["validation"]["route"], "needs_review")

    def test_recovered_border_requires_review_even_at_high_confidence(self):
        document = valid_document()
        character = document["rows"][0]["fields"]["miles"]["characters"][0]
        character["border_recovery"] = {"applied": True, "added_ink_pixels": 25}
        character["preprocessing_issues"] = [BORDER_RECOVERY_WARNING]
        document = self.validate(document)
        self.assertEqual(document["rows"][0]["validation"]["route"], "needs_review")
        self.assertIn(BORDER_RECOVERY_WARNING, document["rows"][0]["validation"]["confidence_checks"]["miles"]["reason"])

    def test_unverified_template_routes_even_when_business_checks_pass(self):
        document = valid_document()
        document["preprocessing"] = {"review_reasons": ["Printed form alignment could not be verified."]}
        document = self.validate(document)
        self.assertEqual(document["rows"][0]["validation"]["route"], "needs_review")
        self.assertIn("Printed form alignment could not be verified.", document["rows"][0]["validation"]["route_reasons"])

    def test_reviewed_correction_can_resolve_a_crop_warning(self):
        document = valid_document()
        document["rows"][0]["fields"]["miles"]["characters"][0]["preprocessing_issues"] = ["Unverified crop"]
        apply_correction(document, "rows.1.miles", "025", "Verified the mileage on the source form")
        document = self.validate(document)
        self.assertEqual(document["rows"][0]["validation"]["route"], "reviewed")


if __name__ == "__main__":
    unittest.main()
