import json
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from app.main import recognize
from app.recognizer import _b64_to_bgr
from app.schemas import RecognizeRequest


class RecognitionLogTest(unittest.TestCase):
    def test_success_logs_result_and_latency_without_image(self):
        payload = {'table_info': {'main_pot': 42},
                   'hero_info': {'hero_cards': ['A 黑桃', 'K 红桃']}, 'villains_info': []}
        engine = Mock(spec=["recognize"])
        engine.recognize.return_value = payload
        request = RecognizeRequest(app='polerv2', image_base64='PRIVATE_IMAGE_CONTENT', parse_all=True)
        with patch('app.main.get_recognizer', return_value=engine):
            with self.assertLogs('dezhou.recognition', level='INFO') as logs:
                result = recognize(request, x_recognition_source='validation')
        self.assertEqual(len(logs.records), 1)
        event = json.loads(logs.records[0].getMessage())
        self.assertEqual(event['result'], payload)
        self.assertEqual(event['elapsed_ms'], result['elapsed_ms'])
        self.assertEqual(event['source'], 'validation')
        self.assertEqual(event['http_status'], 200)
        self.assertTrue(event['parse_all'])
        self.assertEqual(len(event['request_id']), 32)
        self.assertNotIn('PRIVATE_IMAGE_CONTENT', logs.records[0].getMessage())

    def test_empty_decoded_image_is_a_client_error(self):
        for value in ("", "data:image/png;base64,", "%%%%"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Invalid image data"):
                    _b64_to_bgr(value)

    def test_bad_image_logs_400_and_preserves_error(self):
        engine = Mock(spec=["recognize"])
        engine.recognize.side_effect = ValueError('bad image')
        with patch('app.main.get_recognizer', return_value=engine):
            with self.assertLogs('dezhou.recognition', level='INFO') as logs:
                with self.assertRaises(HTTPException) as caught:
                    recognize(RecognizeRequest(image_base64='private'))
        self.assertEqual(caught.exception.status_code, 400)
        event = json.loads(logs.records[0].getMessage())
        self.assertEqual(event['http_status'], 400)
        self.assertEqual(event['error_type'], 'ValueError')
        self.assertNotIn('result', event)
        self.assertEqual(event['source'], 'live')
        self.assertGreaterEqual(event['elapsed_ms'], 0)

    def test_runtime_failure_logs_500_without_exception_payload(self):
        with patch('app.main.get_recognizer', side_effect=RuntimeError('PRIVATE_ERROR_CONTENT')):
            with self.assertLogs('dezhou.recognition', level='INFO') as logs:
                with self.assertRaises(HTTPException) as caught:
                    recognize(RecognizeRequest(image_base64='private'))
        self.assertEqual(caught.exception.status_code, 500)
        event = json.loads(logs.records[0].getMessage())
        self.assertEqual(event['http_status'], 500)
        self.assertEqual(event['error_type'], 'RuntimeError')
        self.assertNotIn('PRIVATE_ERROR_CONTENT', logs.records[0].getMessage())


if __name__ == '__main__':
    unittest.main()
