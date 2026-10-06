"""Check box alignment, ink preservation, and rejection of uncertain crops."""

import unittest
from dataclasses import replace

import cv2
import numpy as np

from mlreader import REPO_ROOT
from mlreader.layout import CANVAS_HEIGHT, CANVAS_WIDTH, FIELDS
from mlreader.fields import assemble_field
from mlreader.pipeline import _predict_cell
from mlreader.registration import (
    BORDER_RECOVERY_WARNING, CellCrop, _recover_border_ink,
    extract_cells, normalize_cell, register_page,
)


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = cv2.imread(str(REPO_ROOT / "assets/form_ml7_blank.png"))

    def test_specks_and_small_edge_fragments_are_empty(self):
        crop = np.full((52, 52), 255, dtype=np.uint8)
        crop[2:23, 0] = 0
        crop[15, 20] = crop[30, 40] = 0
        normalized, status, ink, _ = normalize_cell(crop)
        self.assertEqual(status, "empty")
        self.assertIsNone(normalized)
        self.assertEqual(ink, 0)

    def test_large_handwritten_stroke_at_edge_is_preserved(self):
        crop = np.full((52, 52), 255, dtype=np.uint8)
        crop[8:44, 0:5] = 0
        normalized, status, ink, _ = normalize_cell(crop)
        self.assertEqual(status, "ok")
        self.assertEqual(ink, 180)
        self.assertGreater(np.count_nonzero(normalized), 0)

    def test_missing_pixels_and_dark_crops_are_unreadable(self):
        for crop in (np.empty((0, 0), dtype=np.uint8), np.zeros((52, 52), dtype=np.uint8)):
            self.assertEqual(normalize_cell(crop)[1], "unreadable")

    def test_printing_offset_is_corrected_by_template_alignment(self):
        shifted = cv2.warpAffine(
            self.template, np.float32([[1, 0, 25], [0, 1, 35]]),
            (CANVAS_WIDTH, CANVAS_HEIGHT), borderValue=(255, 255, 255),
        )
        result = register_page(shifted)
        self.assertTrue(result.ok)
        self.assertTrue(result.template_alignment["verified"])
        transform = np.asarray(result.template_alignment["transform"])
        self.assertAlmostEqual(transform[0, 2], -25, delta=3)
        self.assertAlmostEqual(transform[1, 2], -35, delta=3)

    def test_local_box_detection_preserves_empty_rows_and_cell_count(self):
        shifted = cv2.warpAffine(
            self.template, np.float32([[1, 0, 15], [0, 1, 12]]),
            (CANVAS_WIDTH, CANVAS_HEIGHT), borderValue=(255, 255, 255),
        )
        crops = extract_cells(shifted)
        self.assertEqual(len(crops["rows.1.odometer_start"]), 6)
        nominal = FIELDS["rows.1.odometer_start"][0].rect
        actual = crops["rows.1.odometer_start"][0].rect
        self.assertLessEqual(abs(actual[0] - nominal[0] - 15), 2)
        self.assertLessEqual(abs(actual[1] - nominal[1] - 12), 2)
        self.assertTrue(all(cell.extraction_status == "empty" for cells in crops.values() for cell in cells))

    def test_unrecognized_page_does_not_claim_template_alignment(self):
        result = register_page(np.full_like(self.template, 255))
        self.assertTrue(result.ok)
        self.assertFalse(result.template_alignment["verified"])

    def test_missing_frame_keeps_readable_ink_and_requests_review(self):
        image = np.full_like(self.template, 255)
        left, top, _, _ = FIELDS["header.employee_id"][0].rect
        cv2.putText(image, "B", (left + 12, top + 46), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
        cell = extract_cells(image)["header.employee_id"][0]
        self.assertEqual(cell.extraction_status, "ok")
        self.assertTrue(cell.preprocessing_issues)
        self.assertIsNotNone(cell.normalized)

    def test_handwriting_touching_crop_edge_requests_review(self):
        image = self.template.copy()
        left, top, _, _ = FIELDS["header.employee_id"][0].rect
        cv2.rectangle(image, (left + 5, top + 10), (left + 10, top + 50), (0, 0, 0), -1)
        cell = extract_cells(image)["header.employee_id"][0]
        self.assertEqual(cell.extraction_status, "ok")
        self.assertIn("Writing touches the crop boundary and may be clipped.", cell.preprocessing_issues)


class BorderRecoveryTests(unittest.TestCase):
    def make_cell(self, image, issues=()):
        crop = image[15:67, 15:67]
        normalized, status, pixels, reason = normalize_cell(crop)
        return CellCrop(
            "header.employee_id", 0, "letters", None, (10, 10, 72, 72),
            crop, normalized, status, pixels, reason, issues, (15, 15, 67, 67),
        )

    def stroke_image(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[25:55, 13:20] = 0
        return image

    def test_connected_border_stroke_is_retained_inside_own_box(self):
        image = self.stroke_image()
        cell = self.make_cell(image)
        recovered = _recover_border_ink(image, cell)
        self.assertEqual(cell.ink_pixels, 150)
        self.assertEqual(recovered.ink_pixels, 210)
        self.assertEqual(recovered.border_recovery["added_ink_pixels"], 60)
        self.assertEqual(recovered.border_recovery["original_crop_rect"], [15, 15, 67, 67])
        self.assertEqual(recovered.crop_rect, (12, 12, 70, 70))
        np.testing.assert_array_equal(recovered.image, image[12:70, 12:70])
        self.assertIn(BORDER_RECOVERY_WARNING, recovered.preprocessing_issues)
        # A neighboring character beyond this frame cannot affect the crop.
        image[20:60, 73:85] = 0
        neighbor = _recover_border_ink(image, self.make_cell(image))
        np.testing.assert_array_equal(neighbor.normalized, recovered.normalized)

    def test_disconnected_marks_do_not_join_the_character(self):
        image = self.stroke_image()
        recovered = _recover_border_ink(image, self.make_cell(image))
        image[60:65, 13:15] = 0
        with_speck = _recover_border_ink(image, self.make_cell(image))
        self.assertEqual(with_speck.ink_pixels, recovered.ink_pixels)
        np.testing.assert_array_equal(with_speck.normalized, recovered.normalized)

    def test_recovery_evidence_survives_prediction_and_field_assembly(self):
        image = self.stroke_image()
        recovered = _recover_border_ink(image, self.make_cell(image))
        prediction = _predict_cell(recovered, None)
        field = assemble_field("header.employee_id", [prediction] * 6)
        character = field["characters"][0]
        self.assertEqual(character["source_rect"], [10, 10, 72, 72])
        self.assertEqual(character["crop_rect"], [12, 12, 70, 70])
        self.assertEqual(character["border_recovery"]["original_crop_rect"], [15, 15, 67, 67])
        self.assertEqual(character["border_recovery"]["added_ink_pixels"], 60)
        self.assertIn(BORDER_RECOVERY_WARNING, character["preprocessing_issues"])

    def test_printed_line_touching_glyph_is_removed_only_outside_original_crop(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[12:40, 30:38] = 0
        cell = self.make_cell(image)
        image[12:14, 12:70] = 0
        recovered = _recover_border_ink(image, cell)
        self.assertEqual(cell.ink_pixels, 200)
        self.assertEqual(recovered.ink_pixels, 208)
        self.assertGreater(recovered.border_recovery["removed_frame_pixels"], 100)

    def test_empty_unreadable_and_unverified_boxes_are_not_expanded(self):
        blank = np.full((100, 100, 3), 255, dtype=np.uint8)
        image = self.stroke_image()
        for cell, source in (
            (self.make_cell(blank), blank),
            (replace(self.make_cell(image), extraction_status="unreadable"), image),
            (self.make_cell(image, ("Printed field frame could not be located.",)), image),
        ):
            self.assertIs(_recover_border_ink(source, cell), cell)

    def test_missing_added_ink_keeps_original_crop(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[25:55, 25:30] = 0
        cell = self.make_cell(image)
        self.assertIs(_recover_border_ink(image, cell), cell)

    def test_excessively_dark_recovery_keeps_original_and_requests_review(self):
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        image[12:70, 12:70] = 0
        # Original inset is just below the existing density limit; the exposed
        # margin contains dark connected ink beyond that limit.
        image[15:67, 15:67] = 255
        image[15:67, 15:40] = 0
        image[15:18, 15:67] = 0
        image[64:67, 15:67] = 0
        for position in range(12, 70, 10):
            image[12:15, position] = 255
            image[67:70, position] = 255
            image[position, 12:15] = 255
            image[position, 67:70] = 255
        cell = self.make_cell(image)
        self.assertEqual(cell.extraction_status, "ok")
        recovered = _recover_border_ink(image, cell)
        self.assertIsNone(recovered.border_recovery)
        np.testing.assert_array_equal(recovered.normalized, cell.normalized)
        self.assertIn("Border ink could not be recovered reliably.", recovered.preprocessing_issues)


if __name__ == "__main__":
    unittest.main()
