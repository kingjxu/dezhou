"""Observed nine-seat UI, numbering, and request-local layout selection."""
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2

from app.engines import get_recognizer
from app.layouts.polerv2 import LayoutPolerV2, nine_seat_layout
from app.recognizer import TableRecognizer

FIXTURES = Path(__file__).parent / 'fixtures/turn-gate'


class NineSeatTest(unittest.TestCase):
    def test_layout_evidence_and_missing_cross(self):
        r = TableRecognizer.__new__(TableRecognizer)
        r._layout = LayoutPolerV2()
        for name, expected in [('nine-seat-check.jpg', True), ('no-actions.webp', True),
                               ('check-action.webp', False),
                               ('2026-09-15 00:00:05-hero.jpg', False)]:
            image = cv2.imread(str(FIXTURES / name))
            _, compressed = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            for variant in (image, cv2.resize(image, None, fx=.5, fy=.5),
                            cv2.copyMakeBorder(image, 30, 30, 30, 30, cv2.BORDER_CONSTANT),
                            cv2.imdecode(compressed, cv2.IMREAD_COLOR)):
                with self.subTest(name=name, shape=variant.shape):
                    self.assertEqual(r._has_nine_seat_placeholders(r._normalize_canvas(variant)), expected)
        image = r._normalize_canvas(cv2.imread(str(FIXTURES / 'nine-seat-check.jpg')))
        h, w = image.shape[:2]
        image[int(h*.17):int(h*.22), int(w*.28):int(w*.38)] = 0
        self.assertFalse(r._has_nine_seat_placeholders(image))

    def test_numbering_and_concurrent_layout_selection(self):
        layout = nine_seat_layout()
        self.assertEqual([s for s, _ in layout.seat_anchors], list(range(2, 10)))
        self.assertEqual(layout.seat_positions[8], LayoutPolerV2().seat_positions[7])
        self.assertEqual(layout.button_positions[8], LayoutPolerV2().button_positions[7])
        r = TableRecognizer.__new__(TableRecognizer)
        r._layout = LayoutPolerV2()
        original = r._layout
        def selected(instance, image, parse_all):
            return tuple(s for s, _ in instance._layout.seat_anchors)
        images = [base64.b64encode((FIXTURES / name).read_bytes()).decode()
                  for name in ('nine-seat-check.jpg', 'check-action.webp')]
        # Only layout routing runs concurrently; never share live OCR calls.
        with patch.object(TableRecognizer, '_recognize_normalized', selected):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(r.recognize, images * 4))
        self.assertEqual(results, [tuple(range(2, 10)), tuple(range(2, 9))] * 4)
        self.assertIs(r._layout, original)

    def test_real_nine_then_eight_results(self):
        r = get_recognizer('polerv2')
        for name, button, seats in [('nine-seat-check.jpg', 8, 8), ('check-action.webp', 3, 7)]:
            encoded = base64.b64encode((FIXTURES / name).read_bytes()).decode()
            result = r.recognize(encoded, parse_all=True)
            self.assertEqual(result['table_info']['button_seat'], button)
            self.assertEqual(len(result['villains_info']), seats)
            if seats == 8:
                players = {p['seat']: p for p in result['villains_info']}
                self.assertEqual(players[8]['stack'], 836)
                self.assertTrue(all(p['status'] == 'empty' for s,p in players.items() if s != 8))
        self.assertEqual(len(r._layout.seat_anchors), 7)
