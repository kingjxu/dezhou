"""Real-image coverage of active cyan check buttons and gray pre-actions."""
from pathlib import Path
import unittest

import cv2

from app.layouts.polerv2 import LayoutPolerV2
from app.recognizer import TableRecognizer


class TurnGateImagesTest(unittest.TestCase):
    def setUp(self):
        self.recognizer = TableRecognizer.__new__(TableRecognizer)
        self.recognizer._layout = LayoutPolerV2()

    def test_real_images_at_different_sizes(self):
        cases = {
            'check-action.webp': True,
            'nine-seat-check.jpg': True,
            'no-actions.webp': False,
            '2026-09-14 23:48:51-hero.jpg': True,
            '2026-09-15 00:00:05-hero.jpg': False,
            '2026-09-15 00:16:14-hero.jpg': False,
        }
        for name, expected in cases.items():
            original = cv2.imread(str(Path(__file__).parent / 'fixtures/turn-gate' / name))
            self.assertIsNotNone(original, name)
            half = cv2.resize(original, (original.shape[1] // 2, original.shape[0] // 2))
            _, jpeg = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 50])
            variants = {
                'original': original,
                'half': half,
                'letterbox': cv2.copyMakeBorder(half, 40, 40, 30, 30,
                                                cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                'jpeg50': cv2.imdecode(jpeg, cv2.IMREAD_COLOR),
            }
            for variant, image in variants.items():
                with self.subTest(image=name, variant=variant):
                    normalized = self.recognizer._normalize_canvas(image)
                    self.assertEqual(self.recognizer._detect_hero_turn(normalized), expected)
                    self.assertEqual(self.recognizer._detect_turn_fast(normalized), expected)

    def test_cyan_button_needs_fold_button_and_icon(self):
        image = cv2.imread(str(Path(__file__).parent / 'fixtures/turn-gate/check-action.webp'))
        image = self.recognizer._normalize_canvas(image)
        h, w = image.shape[:2]
        # Remove the fold button: table-colored highlights alone are not a turn.
        no_fold = image.copy()
        no_fold[int(h*.70):int(h*.78), int(w*.24):int(w*.40)] = (0, 0, 0)
        self.assertFalse(self.recognizer._detect_hero_turn(no_fold))
        # A bright cyan patch without a white action icon is also insufficient.
        image[int(h*.70):int(h*.78), int(w*.61):int(w*.76)] = (200, 220, 0)
        self.assertFalse(self.recognizer._detect_hero_turn(image))


if __name__ == '__main__':
    unittest.main()
