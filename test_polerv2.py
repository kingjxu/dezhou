import unittest

import numpy as np

from app.layouts.polerv2 import LayoutPolerV2
from app.recognizer import (
    TableRecognizer,
    _classify_v2_marker_digit,
    _should_use_v2_tight_hero_rank,
)


class PolerV2Test(unittest.TestCase):
    def test_v2_marker_digit_supports_distinct_blind_values(self):
        self.assertEqual(_classify_v2_marker_digit(4, 20), 1)
        self.assertEqual(_classify_v2_marker_digit(14, 28), 2)
        self.assertEqual(_classify_v2_marker_digit(22, 28), 4)

    def test_v2_button_positions_include_hero_side_marker(self):
        self.assertIn((0.65, 0.78), LayoutPolerV2().button_positions[1])

    def test_v2_marker_roi_covers_left_upper_seat(self):
        self.assertEqual(
            LayoutPolerV2().seat_bet_fallback_rois[4],
            (0.120, 0.270, 0.280, 0.380),
        )

    def test_v2_rank_crop_keeps_two_character_rank(self):
        self.assertEqual(LayoutPolerV2().hero_card_rank_rx, (0.00, 0.80))
        self.assertEqual(LayoutPolerV2().hero_card_rank_ry, (0.02, 0.60))

    def test_v2_tight_rank_overrides_artwork_polluted_face_or_suit(self):
        self.assertTrue(_should_use_v2_tight_hero_rank("Q", "7", 1.0))
        self.assertTrue(_should_use_v2_tight_hero_rank("2", "J", 0.999))
        self.assertTrue(_should_use_v2_tight_hero_rank("9", "K", 0.999))

    def test_v2_tight_rank_does_not_override_similar_numeral_noise(self):
        self.assertFalse(_should_use_v2_tight_hero_rank("5", "6", 0.999))
        self.assertFalse(_should_use_v2_tight_hero_rank("8", "A", 0.999))
        self.assertFalse(_should_use_v2_tight_hero_rank("Q", "7", 0.97))

    def _normalizer(self):
        recognizer = TableRecognizer.__new__(TableRecognizer)
        recognizer._layout = LayoutPolerV2()
        return recognizer

    def test_normalize_canvas_keeps_full_normal_screenshot(self):
        image = np.full((2400, 1176, 3), 20, dtype=np.uint8)
        normalized = self._normalizer()._normalize_canvas(image)
        self.assertEqual(normalized.shape, image.shape)

    def test_normalize_canvas_crops_black_letterbox(self):
        image = np.zeros((2160, 1080, 3), dtype=np.uint8)
        image[80:2080, 90:990] = 20
        normalized = self._normalizer()._normalize_canvas(image)
        self.assertEqual(normalized.shape, (2400, 1080, 3))

    def test_normalize_canvas_keeps_black_ui_regions_inside_screenshot(self):
        image = np.full((2400, 1176, 3), 20, dtype=np.uint8)
        image[300:1000, 300:900] = 0
        image[1800:2200, 100:1100] = 0
        normalized = self._normalizer()._normalize_canvas(image)
        self.assertEqual(normalized.shape, image.shape)

    def test_normalize_canvas_keeps_non_reference_single_axis_black_border(self):
        image = np.full((2400, 1176, 3), 20, dtype=np.uint8)
        image[:80] = 0
        normalized = self._normalizer()._normalize_canvas(image)
        self.assertEqual(normalized.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
