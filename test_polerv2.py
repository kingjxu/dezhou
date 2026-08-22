import unittest
from pathlib import Path

import numpy as np

from app.engines import get_recognizer
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
        self.assertTrue(_should_use_v2_tight_hero_rank("Q", 0.84, "7", 1.0))
        self.assertTrue(_should_use_v2_tight_hero_rank("Q", 1.0, "10", 0.999))
        self.assertTrue(_should_use_v2_tight_hero_rank("2", 0.99, "J", 0.999))
        self.assertTrue(_should_use_v2_tight_hero_rank("9", 0.99, "K", 0.999))
        self.assertTrue(_should_use_v2_tight_hero_rank("J", 0.55, "A", 0.999))
        self.assertTrue(_should_use_v2_tight_hero_rank("2", 0.99, "3", 0.999))

    def test_v2_tight_rank_does_not_override_similar_numeral_noise(self):
        self.assertFalse(_should_use_v2_tight_hero_rank("5", 1.0, "6", 0.999))
        self.assertFalse(_should_use_v2_tight_hero_rank("8", 1.0, "A", 0.999))
        self.assertFalse(_should_use_v2_tight_hero_rank("Q", 0.84, "7", 0.97))
        self.assertFalse(_should_use_v2_tight_hero_rank("J", 0.90, "A", 0.999))

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

    def test_v2_seven_image_regression(self):
        image_dir = Path(__file__).parent / "reports" / "images-seven"
        cases = [
            ("01.jpg", ["A 黑桃", "J 红桃"], 24, "preflop", 60, []),
            ("02.jpg", ["K 黑桃", "10 黑桃"], 42, "preflop", 21, []),
            ("03.jpg", ["K 方块", "A 梅花"], 6, "preflop", 31, []),
            ("04.jpg", ["6 红桃", "9 方块"], 18, "flop", 78, ["Q 红桃", "6 方块", "8 梅花"]),
            ("05.jpg", ["J 红桃", "10 梅花"], 24, "preflop", 25, []),
            ("06.jpg", ["3 黑桃", "3 红桃"], 6, "preflop", 21, []),
            ("07.jpg", ["9 方块", "8 方块"], 90, "preflop", 94, []),
        ]
        if not image_dir.exists():
            self.skipTest("seven-image regression fixtures are not available")
        recognizer = get_recognizer("polerv2")
        import base64
        for filename, hero_cards, mushroom, stage, pot, community in cases:
            with self.subTest(filename=filename):
                image_path = image_dir / filename
                result = recognizer.recognize(
                    base64.b64encode(image_path.read_bytes()).decode(),
                    parse_all=True,
                )
                table_info = result["table_info"]
                self.assertEqual(result["hero_info"]["hero_cards"], hero_cards)
                self.assertEqual(table_info["mushroom_count"], mushroom)
                self.assertEqual(table_info["stage"], stage)
                self.assertEqual(table_info["main_pot"], pot)
                self.assertEqual(table_info["community_cards"], community)
                self.assertEqual(table_info["blind_size"], "1/2(2)")


if __name__ == "__main__":
    unittest.main()
