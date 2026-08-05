import unittest

import app.recognizer as recognizer


class FakeRapidOCR:
    def __init__(self):
        self.calls = []

    def __call__(self, _image, **kwargs):
        self.calls.append(kwargs)
        return [[[None, "12", 0.9]], None]


class OCRIsolationTest(unittest.TestCase):
    def setUp(self):
        recognizer._OCR_ENGINE_BACKENDS.clear()
        recognizer._RAPID_CALL_KWARGS.clear()
        recognizer._RAPID_PREFERRED_KWARGS.clear()
        recognizer._OCR_BACKEND = "rapid"

    def test_poler_then_dpzx_keeps_call_kwargs_isolated(self):
        poler = FakeRapidOCR()
        dpzx = FakeRapidOCR()
        recognizer._OCR_ENGINE_BACKENDS[id(poler)] = "rapid"
        recognizer._OCR_ENGINE_BACKENDS[id(dpzx)] = "rapid"
        recognizer._RAPID_PREFERRED_KWARGS[id(poler)] = {"use_cls": False}
        recognizer._RAPID_PREFERRED_KWARGS[id(dpzx)] = {}

        recognizer._ocr_run(poler, object())
        recognizer._ocr_run(dpzx, object())

        self.assertEqual(poler.calls, [{"use_cls": False}])
        self.assertEqual(dpzx.calls, [{}])

    def test_dpzx_then_poler_keeps_call_kwargs_isolated(self):
        poler = FakeRapidOCR()
        dpzx = FakeRapidOCR()
        recognizer._OCR_ENGINE_BACKENDS[id(poler)] = "rapid"
        recognizer._OCR_ENGINE_BACKENDS[id(dpzx)] = "rapid"
        recognizer._RAPID_PREFERRED_KWARGS[id(poler)] = {"use_cls": False}
        recognizer._RAPID_PREFERRED_KWARGS[id(dpzx)] = {}

        recognizer._ocr_run(dpzx, object())
        recognizer._ocr_run(poler, object())

        self.assertEqual(dpzx.calls, [{}])
        self.assertEqual(poler.calls, [{"use_cls": False}])
