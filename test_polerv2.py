import unittest

import numpy as np

from app.layouts.polerv2 import LayoutPolerV2
from app.recognizer import TableRecognizer, _confirmed_marker_bets


class PolerV2Test(unittest.TestCase):
    def test_marker_majority_discards_isolated_value(self):
        self.assertEqual(
            _confirmed_marker_bets({2: 4, 5: 4, 7: 4, 8: 1}),
            {2: 4, 5: 4, 7: 4},
        )

    def test_marker_majority_requires_single_value(self):
        self.assertEqual(_confirmed_marker_bets({2: 4, 5: 1}), {})

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
        image[88:2072, 54:1026] = 20
        normalized = self._normalizer()._normalize_canvas(image)
        self.assertEqual(normalized.shape, (1984, 972, 3))

    def test_normalize_canvas_keeps_black_ui_regions_inside_screenshot(self):
        image = np.full((2400, 1176, 3), 20, dtype=np.uint8)
        image[300:1000, 300:900] = 0
        image[1800:2200, 100:1100] = 0
        normalized = self._normalizer()._normalize_canvas(image)
        self.assertEqual(normalized.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
