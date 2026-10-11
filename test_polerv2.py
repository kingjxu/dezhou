import math
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import app.recognizer as recognizer_module

from app.engines import get_recognizer
from app.layouts.polerv2 import LayoutPolerV2
from app.layouts.polerv2 import nine_seat_layout
from app.layouts.poler import LayoutPoler
from app.recognizer import OcrItem
from app.recognizer import (
    TableRecognizer,
    _classify_v2_marker_digit,
    _should_skip_v2_wide_hero_akq,
    _should_use_v2_tight_hero_rank,
)


class PolerV2Test(unittest.TestCase):
    def test_v2_turn_requires_fold_button_with_bright_right_control(self):
        recognizer = self._normalizer()
        hsv = np.zeros((1000, 1000, 3), dtype=np.uint8)
        hsv[:, :] = (85, 180, 90)
        # A bright green right-side control also exists on the hand-end screen.
        hsv[700:780, 610:760] = (60, 255, 220)
        hand_end = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertFalse(recognizer._detect_hero_turn(hand_end))

        # The same right control is actionable only when the red fold button is
        # simultaneously present in its dedicated left ROI.
        hsv[700:780, 240:400] = (0, 255, 220)
        actionable = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertTrue(recognizer._detect_hero_turn(actionable))

    def test_v2_turn_rejects_launcher_icons_without_table_surface(self):
        recognizer = self._normalizer()
        hsv = np.zeros((1000, 490, 3), dtype=np.uint8)
        hsv[:, :] = (170, 120, 220)
        hsv[700:780, 118:196] = (0, 255, 220)
        hsv[700:780, 299:373] = (60, 255, 220)

        launcher = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.assertFalse(recognizer._detect_hero_turn(launcher))

    def test_v2_turn_rejects_history_sheet_over_action_buttons(self):
        recognizer = self._normalizer()
        hsv = np.zeros((1000, 490, 3), dtype=np.uint8)
        hsv[:, :] = (85, 180, 90)
        hsv[300:650, 147:343] = (0, 0, 35)
        hsv[700:780, 118:196] = (0, 255, 220)
        hsv[700:780, 299:373] = (60, 255, 220)

        history = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.assertFalse(recognizer._detect_hero_turn(history))

    def test_v2_turn_accepts_gray_fold_x_for_all_in_option(self):
        recognizer = self._normalizer()
        hsv = np.zeros((1000, 1000, 3), dtype=np.uint8)
        hsv[:, :] = (85, 180, 90)
        # Some all-in states replace the red fold face with a gray circle and
        # a large white X while keeping an actionable right-side control.
        cv2.circle(hsv, (320, 740), 55, (0, 20, 140), -1)
        cv2.line(hsv, (285, 705), (355, 775), (0, 0, 255), 12)
        cv2.line(hsv, (355, 705), (285, 775), (0, 0, 255), 12)
        hsv[700:780, 610:760] = (0, 255, 220)

        actionable = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.assertTrue(recognizer._detect_hero_turn(actionable))

    def test_v2_button_accepts_small_hero_side_badge(self):
        recognizer = self._normalizer()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        center = (614, 799)
        points = []
        for index in range(16):
            angle = math.pi * index / 8
            radius = 35 if index % 2 == 0 else 15
            points.append([
                round(center[0] + radius * math.cos(angle)),
                round(center[1] + radius * math.sin(angle)),
            ])
        cv2.fillPoly(image, [np.array(points, dtype=np.int32)], (255, 255, 255))

        self.assertEqual(recognizer._detect_button_seat(image), 1)

    def test_v2_low_confidence_board_ten_uses_ocr_verification(self):
        recognizer = get_recognizer("polerv2")
        image = cv2.imread(str(
            Path(__file__).parent
            / "fixtures/field-regression/board-ten-transition.jpg"))

        ranks = recognizer._find_ranks(
            image, [], recognizer._layout.board_rois[:3])

        self.assertEqual(ranks, {0: "7", 1: "6", 2: "10"})

    def test_empty_placeholder_uses_explicit_plus_center(self):
        recognizer = self._normalizer()
        hsv = np.zeros((2400, 1080, 3), dtype=np.uint8)
        cx, cy = int(.92 * 1080), int(.61 * 2400)
        cv2.line(hsv, (cx - 18, cy), (cx + 18, cy), (85, 200, 180), 12)
        cv2.line(hsv, (cx, cy - 18), (cx, cy + 18), (85, 200, 180), 12)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.assertTrue(recognizer._has_v2_empty_placeholder(image, 8))
        self.assertFalse(recognizer._has_v2_empty_placeholder(image, 2))

    def test_middle_right_empty_placeholder_uses_inward_center(self):
        recognizer = self._normalizer()
        hsv = np.zeros((2400, 1080, 3), dtype=np.uint8)
        cx, cy = int(.92 * 1080), int(.47 * 2400)
        cv2.line(hsv, (cx - 18, cy), (cx + 18, cy), (85, 200, 180), 12)
        cv2.line(hsv, (cx, cy - 18), (cx, cy + 18), (85, 200, 180), 12)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.assertTrue(recognizer._has_v2_empty_placeholder(image, 7))

    def test_nine_seat_detection_accepts_two_top_balances_only(self):
        recognizer = self._normalizer()
        image = np.zeros((2400, 1080, 3), dtype=np.uint8)

        def fill(roi, pixels):
            x0, y0, x1, y1 = roi
            xa, ya = int(x0 * 1080), int(y0 * 2400)
            xb, yb = int(x1 * 1080), int(y1 * 2400)
            image[ya:yb, xa:xb] = 0
            image[ya:yb, xa:min(xb, xa + pixels)] = 255

        left = (.265, .205, .395, .250)
        right = (.605, .205, .735, .250)
        center = (.435, .205, .565, .250)
        fill(left, 2)
        fill(right, 2)
        self.assertTrue(recognizer._has_nine_seat_top_balances(image))

        fill(center, 4)
        self.assertFalse(recognizer._has_nine_seat_top_balances(image))

        image[:] = 0
        fill(left, 2)
        self.assertFalse(recognizer._has_nine_seat_top_balances(image))

    def test_v2_marker_digit_supports_distinct_blind_values(self):
        self.assertIsNone(_classify_v2_marker_digit(2, 32))
        self.assertEqual(_classify_v2_marker_digit(7, 20, 0.28), 1)
        self.assertEqual(_classify_v2_marker_digit(14, 28), 2)
        self.assertEqual(_classify_v2_marker_digit(22, 28), 4)

    def test_v2_marker_digit_rejects_sparse_card_back_strokes(self):
        self.assertIsNone(_classify_v2_marker_digit(9, 21, 0.17))
        self.assertIsNone(_classify_v2_marker_digit(8, 14, 0.22))

    def test_v2_marker_ocr_uncertainty_preserves_digit_count(self):
        class OCRStub:
            def __init__(self, mode, text):
                self.mode, self.text = mode, text

            def text_rec(self, images):
                if self.mode == "exception":
                    raise TypeError("unsupported recognizer signature")
                return [(self.text, 0.99 if self.mode == "confident" else 0.2)], 0.0

            def __call__(self, image, **kwargs):
                return [], None

        for text, mode, expected in (
            ("33", "confident", 33),
            (".9", "confident", 9),
            (". 21", "confident", 21),
            ("24.6", "confident", None),
            ("156", "confident", 156),
            ("33", "uncertain", None),
            ("156", "uncertain", None),
            ("33", "exception", None),
            ("4", "uncertain", 4),
            ("", "uncertain", None),
        ):
            with self.subTest(text=text, mode=mode):
                # 真实颜色/轮廓路径，只替换不可控的 OCR 外部依赖。
                image = np.full((100, 140, 3), (40, 40, 160), dtype=np.uint8)
                cv2.putText(image, text, (20, 57), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 3, cv2.LINE_8)
                recognizer = self._normalizer()
                recognizer._ocr = OCRStub(mode, text)
                with patch.dict(recognizer_module._OCR_ENGINE_BACKENDS,
                                {id(recognizer._ocr): "rapid"}):
                    with patch.dict(recognizer_module._RAPID_CALL_KWARGS, {}, clear=True):
                        actual = recognizer._detect_bet_chip_digit(
                            image, (0, 0, 1, 1), ("red",))
                self.assertEqual(actual, expected)

    def test_blinds_preserve_v2_structure_without_changing_legacy(self):
        r = self._normalizer()
        for text, expected in (("德州1/2/4（2）", "1/2/4(2)"),
                               ("德州10/20/40 (20)", "10/20/40(20)"),
                               ("德州172/4(2)", "1/2/4(2)"),
                               ("1/2(2)", "1/2(2)")):
            items = [OcrItem(text=text, cx=.5, cy=.58, score=.99)]
            self.assertEqual(r._find_blind_size(items), expected)
        r._layout = LayoutPoler()
        self.assertEqual(r._find_blind_size([
            OcrItem(text="1/2/4(2)", cx=.5, cy=.58, score=.99)]), "1/2(2)")

    def test_v2_wager_images_keep_edge_digits_and_ignore_overlapping_labels(self):
        r = get_recognizer("polerv2")
        for filename, seat, expected in (("bet-22.jpg", 4, 22),
                                         ("bet-21.jpg", 4, 21),
                                         ("bet-21.jpg", 6, 21),
                                         ("bet-9.jpg", 3, 9),
                                         ("bet-4.jpg", 6, 4)):
            with self.subTest(filename=filename, seat=seat):
                image = cv2.imread(str(Path(__file__).parent / "fixtures/field-regression" / filename))
                image = r._normalize_canvas(image)
                self.assertEqual(r._detect_bet_chip_digit(
                    image, r._layout.seat_bet_fallback_rois[seat], ("red",)), expected)

    def test_v2_blue_wager_cards_and_hero_bet_roi(self):
        r = get_recognizer("polerv2")
        fixtures = Path(__file__).parent / "fixtures/field-regression"

        blue_wagers = r._normalize_canvas(cv2.imread(str(fixtures / "blue-wagers.jpg")))
        self.assertEqual(r._read_v2_hero_bet(blue_wagers), 1)
        for seat, expected in ((2, 2), (3, 4), (6, 4)):
            with self.subTest(seat=seat):
                self.assertEqual(r._detect_bet_chip_digit(
                    blue_wagers,
                    r._layout.seat_bet_fallback_rois[seat],
                    r._layout.seat_bet_fallback_chip_colors[seat],
                ), expected)

        hero_bet = r._normalize_canvas(cv2.imread(str(fixtures / "hero-bet-2.jpg")))
        self.assertEqual(r._read_v2_hero_bet(hero_bet), 2)

    def test_v2_blue_marker_support_does_not_create_wagers(self):
        r = get_recognizer("polerv2")
        image = cv2.imread(str(
            Path(__file__).parent / "fixtures/turn-gate/no-actions.webp"))
        image = r._normalize_canvas(image)

        self.assertIsNone(r._read_v2_hero_bet(image))
        for seat, roi in r._layout.seat_bet_fallback_rois.items():
            with self.subTest(seat=seat):
                self.assertIsNone(r._detect_bet_chip_digit(
                    image, roi, r._layout.seat_bet_fallback_chip_colors[seat]))

    def test_v2_visible_hero_bet_33_from_decided_frame(self):
        r = get_recognizer("polerv2")
        image = cv2.imread(str(
            Path(__file__).parent
            / "fixtures/field-regression/hero-bet-33-decided.jpg"))
        image = r._normalize_canvas(image)

        self.assertEqual(r._read_v2_hero_bet(image), 33)

    def test_confident_ocr_cannot_drop_a_visible_wager_digit(self):
        r = self._normalizer()
        image = cv2.imread(str(Path(__file__).parent / "fixtures/field-regression/bet-21.jpg"))
        image = r._normalize_canvas(image)
        class TruncatedOCR:
            def text_rec(self, images):
                return [("2", .999)], 0.0
        r._ocr = TruncatedOCR()
        with patch.dict(recognizer_module._OCR_ENGINE_BACKENDS, {id(r._ocr): "rapid"}), patch.object(
                recognizer_module, "_ocr_run", return_value=[(None, "2", .999)]):
            self.assertIsNone(r._detect_bet_chip_digit(
                image, r._layout.seat_bet_fallback_rois[4], ("red",)))

    def test_right_wager_roi_keeps_leading_three_digit_value(self):
        r = self._normalizer()
        image = np.zeros((2400, 1080, 3), dtype=np.uint8)
        cv2.putText(image, "102", (710, 790), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.rectangle(image, (785, 720), (835, 820), (40, 40, 150), -1)

        class ThreeDigitOCR:
            def text_rec(self, images):
                return [("102", .999) for _ in images], 0.0

        r._ocr = ThreeDigitOCR()
        with patch.dict(recognizer_module._OCR_ENGINE_BACKENDS,
                        {id(r._ocr): "rapid"}):
            self.assertEqual(r._detect_bet_chip_digit(
                image, r._layout.seat_bet_fallback_rois[6], ("red",)), 102)

    def test_real_right_wager_keeps_255_leading_digit(self):
        r = get_recognizer("polerv2")
        image = cv2.imread(str(
            Path(__file__).parent / "fixtures/field-regression/wager-255-crop.jpg"))

        self.assertEqual(r._detect_bet_chip_digit(
            image, (0, 0, 1, 1), ("red", "blue")), 255)

    def test_wider_wager_row_does_not_absorb_nearby_status_text(self):
        r = get_recognizer("polerv2")
        image = cv2.imread(str(
            Path(__file__).parent
            / "fixtures/field-regression/wager-20-nearby-text-crop.jpg"))

        self.assertEqual(r._detect_bet_chip_digit(
            image, (0, 0, 1, 1), ("red", "blue")), 20)

    def test_right_wager_crop_is_not_limited_to_three_digits(self):
        r = self._normalizer()
        image = np.zeros((240, 172, 3), dtype=np.uint8)
        cv2.rectangle(image, (88, 124), (153, 198), (40, 40, 150), -1)
        cv2.putText(image, "1255", (18, 170), cv2.FONT_HERSHEY_SIMPLEX,
                    .95, (255, 255, 255), 2, cv2.LINE_AA)

        class FourDigitOCR:
            def text_rec(self, images):
                return [("1255", .999) for _ in images], 0.0

        r._ocr = FourDigitOCR()
        with patch.dict(recognizer_module._OCR_ENGINE_BACKENDS,
                        {id(r._ocr): "rapid"}):
            self.assertEqual(r._detect_bet_chip_digit(
                image, (0, 0, 1, 1), ("red",)), 1255)

    def test_wager_validation_accepts_table_max_four_digits(self):
        self.assertEqual(
            self._normalizer()._valid_v2_digit_read("1255", .99, 4),
            1255,
        )
        self.assertIsNone(
            self._normalizer()._valid_v2_digit_read("12550", .99, 5),
        )

    def test_stack_glyph_count_detects_truncated_high_confidence_read(self):
        r = self._normalizer()
        image = cv2.imread(str(Path(__file__).parent / "fixtures/field-regression/balance-191.jpg"))
        for image in (image, cv2.resize(image, None, fx=.5, fy=.5)):
            self.assertEqual(r._v2_stack_digit_count(r._normalize_canvas(image), 2), 3)
            self.assertNotEqual(r._v2_stack_digit_count(r._normalize_canvas(image), 2), len("19"))

    def test_wide_canvas_uses_alternate_top_stack_band(self):
        r = self._normalizer()
        wide = np.zeros((2048, 1004, 3), dtype=np.uint8)
        reference = np.zeros((2400, 1080, 3), dtype=np.uint8)

        self.assertEqual(
            r._v2_stack_roi(wide, 5),
            r._layout.wide_canvas_seat_stack_rois[5],
        )
        self.assertEqual(
            r._v2_stack_roi(reference, 5),
            r._layout.seat_stack_rois[5],
        )

    def test_stack_batch_is_single_call_and_does_not_mix_rows(self):
        r = self._normalizer()
        r._ocr = object()
        def box(y0, y1):
            return [[20, y0], [60, y0], [60, y1], [20, y1]]
        crop = np.zeros((40, 100, 3), dtype=np.uint8)
        for reads, expected in (
            ([(box(20, 50), "191", .99), (box(75, 105), "1156", .99)], {2: 191, 3: 1156}),
            ([(box(50, 80), "1911156", .99)], {}),
            ([(box(20, 50), "191", .5)], {}),
            ([(box(20, 30), "191", .99), (box(35, 50), "92", .99)], {}),
        ):
            with patch.object(r, "_roi", return_value=crop), patch.object(
                    recognizer_module, "_ocr_run", return_value=reads) as ocr:
                self.assertEqual(r._read_v2_stacks(crop, [2, 3]), expected)
                self.assertEqual(ocr.call_count, 1)
        with patch.object(recognizer_module, "_ocr_run") as ocr:
            self.assertEqual(r._read_v2_stacks(crop, []), {})
            ocr.assert_not_called()

    def test_small_mushroom_uses_ocr_and_never_guesses_six(self):
        r = self._normalizer()
        r._ocr = object()
        image = cv2.imread(str(Path(__file__).parent / "fixtures/turn-gate/no-actions.webp"))
        image = r._normalize_canvas(image)
        for reads, expected in (([(None, "0", .9)], 0),
                                ([(None, "6", .9)], 6),
                                ([(None, "0", .4)], None),
                                ([], None)):
            with patch.object(recognizer_module, "_ocr_run", return_value=reads):
                self.assertEqual(r._detect_small_mushroom_count(image), expected)

    def test_top_overlay_text_is_not_a_player_status(self):
        r = self._normalizer()
        for text in ("等待反馈", "弃牌"):
            items = [OcrItem(text=text, cx=.5, cy=.07, score=.99),
                     OcrItem(text="三好姐姐", cx=.5, cy=.143, score=.99),
                     OcrItem(text="154", cx=.5, cy=.222, score=.99)]
            self.assertEqual(r._seat_status(items, 5), "active")
        self.assertEqual(r._seat_status([
            OcrItem(text="留座174秒", cx=.5, cy=.18, score=.99)], 5), "waiting")

    def test_nine_top_statuses_share_one_ocr_call_and_stay_isolated(self):
        r = self._normalizer()
        r._layout = nine_seat_layout()
        r._ocr = object()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)

        def box(cx, cy):
            return [[cx - 10, cy - 5], [cx + 10, cy - 5],
                    [cx + 10, cy + 5], [cx - 10, cy + 5]]

        # The packed crops start at x=24 and x=173. These centers map back
        # exactly to the two nine-seat status anchors at x=.44 and x=.56.
        reads = [(box(89, 69), "弃牌", .99),
                 (box(233, 69), "留座20秒", .99)]
        with patch.object(recognizer_module, "_ocr_run", return_value=reads) as ocr:
            self.assertEqual(r._read_v2_nine_top_statuses(image), {
                5: "folded",
                6: "waiting",
            })
            ocr.assert_called_once()

    def test_eight_seat_skips_nine_top_status_ocr(self):
        r = self._normalizer()
        r._ocr = object()
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        with patch.object(recognizer_module, "_ocr_run") as ocr:
            self.assertEqual(r._read_v2_nine_top_statuses(image), {})
            ocr.assert_not_called()

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
        self.assertTrue(_should_use_v2_tight_hero_rank("2", 0.98, "3", 0.999))

    def test_v2_tight_rank_does_not_override_similar_numeral_noise(self):
        self.assertFalse(_should_use_v2_tight_hero_rank("5", 1.0, "6", 0.999))
        self.assertFalse(_should_use_v2_tight_hero_rank("8", 1.0, "A", 0.999))
        self.assertFalse(_should_use_v2_tight_hero_rank("Q", 0.84, "7", 0.97))
        self.assertFalse(_should_use_v2_tight_hero_rank("J", 0.90, "A", 0.999))
        self.assertFalse(_should_use_v2_tight_hero_rank("2", 1.0, "3", 0.986))

    def test_v2_wide_hero_akq_exception_is_k_only(self):
        shape = (20, 16)
        self.assertTrue(_should_skip_v2_wide_hero_akq(
            "K", True, "polerv2", shape, 0.99))
        self.assertFalse(_should_skip_v2_wide_hero_akq(
            "A", True, "polerv2", shape, 0.99))
        self.assertFalse(_should_skip_v2_wide_hero_akq(
            "Q", True, "polerv2", shape, 0.99))
        self.assertFalse(_should_skip_v2_wide_hero_akq(
            "K", True, "polerv2", shape, 0.94))
        self.assertFalse(_should_skip_v2_wide_hero_akq(
            "K", True, "polerv2", (20, 12), 0.99))

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
                self.assertEqual(table_info["blind_size"], "1/2/4(2)")

    def test_v2_remote_hero_image_regression(self):
        image_dir = Path(__file__).parent / "reports" / "images-remote-hero"
        cases = [
            ("154714-hero.jpg", ["Q 方块", "2 方块"], None, "preflop", 21, []),
            ("155317-hero.jpg", ["7 梅花", "2 梅花"], None, "preflop", 25, []),
            ("155457-hero.jpg", ["J 方块", "8 梅花"], None, "preflop", 23, []),
            ("2026-09-21-22-02-29-hero.jpg", ["6 红桃", "K 黑桃"], None, "preflop", 36, []),
        ]
        if not image_dir.exists():
            self.skipTest("remote hero regression fixtures are not available")
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
                self.assertEqual(table_info["blind_size"], "1/2/4(2)")


if __name__ == "__main__":
    unittest.main()
