"""Check box alignment, ink preservation, and rejection of uncertain crops."""

import unittest

import cv2
import numpy as np

from mlreader import REPO_ROOT
from mlreader.layout import CANVAS_HEIGHT, CANVAS_WIDTH, FIELDS
from mlreader.registration import extract_cells, normalize_cell, register_page


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


if __name__ == "__main__":
    unittest.main()
