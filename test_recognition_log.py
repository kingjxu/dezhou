import copy
import json
import os
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from app.main import app
from app.recognizer import _b64_to_bgr


class RecognitionLogTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        self.payload = {'table_info': {'main_pot': 42},
                        'hero_info': {'seat': 1, 'hero_cards': ['A 黑桃', 'K 红桃']},
                        'villains_info': []}
        self.engine = Mock(spec=['recognize'])
        self.engine.recognize.return_value = self.payload

    def request(self, headers=None):
        with patch('app.main.get_recognizer', return_value=self.engine):
            with self.assertLogs('dezhou.recognition', level='INFO') as logs:
                response = self.client.post('/recognize', headers=headers or {}, json={
                    'app': 'polerv2', 'image_base64': 'PRIVATE_IMAGE_CONTENT', 'parse_all': True,
                })
        self.assertEqual(len(logs.records), 1)
        self.assertNotIn('PRIVATE_IMAGE_CONTENT', logs.records[0].getMessage())
        return response, json.loads(logs.records[0].getMessage())

    def test_success_logs_validated_result_and_latency(self):
        response, event = self.request()
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(event['result'], {key: body[key] for key in self.payload})
        self.assertEqual(event['elapsed_ms'], body['elapsed_ms'])
        self.assertEqual(event['source'], 'live')
        self.assertEqual(event['http_status'], 200)
        self.assertTrue(event['parse_all'])
        self.assertEqual(len(event['request_id']), 32)

    def test_validation_requires_configured_server_secret(self):
        for configured, source, token, expected in (
            ('', 'validation', '', 'live'),
            ('', 'validation', 'secret', 'live'),
            ('secret', 'validation', '', 'live'),
            ('secret', 'validation', 'wrong', 'live'),
            ('secret', 'live', 'secret', 'live'),
            ('secret', 'validation', 'secret', 'validation'),
        ):
            with self.subTest(configured=bool(configured), source=source, expected=expected):
                with patch.dict(os.environ, {'DEZHOU_VALIDATION_TOKEN': configured}):
                    response, event = self.request({
                        'X-Recognition-Source': source, 'X-Recognition-Token': token,
                    })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(event['source'], expected)
                self.assertNotIn('secret', json.dumps(event))

    def test_invalid_engine_results_log_actual_http_500(self):
        bad_payloads = [
            {},
            {**self.payload, 'table_info': {'stage': 'illegal'}},
            {**self.payload, 'hero_info': {'seat': 1, 'status': 'illegal'}},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                self.engine.recognize.return_value = copy.deepcopy(payload)
                response, event = self.request()
                self.assertEqual(response.status_code, 500)
                self.assertEqual(event['http_status'], response.status_code)
                self.assertEqual(event['error_type'], 'ValidationError')
                self.assertNotIn('result', event)

    def test_empty_decoded_image_is_a_client_error(self):
        for value in ('', 'data:image/png;base64,', '%%%%'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'Invalid image data'):
                    _b64_to_bgr(value)

    def test_bad_image_logs_400_and_preserves_error(self):
        self.engine.recognize.side_effect = ValueError('bad image')
        response, event = self.request()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(event['http_status'], 400)
        self.assertEqual(event['error_type'], 'ValueError')
        self.assertNotIn('result', event)
        self.assertEqual(event['source'], 'live')
        self.assertGreaterEqual(event['elapsed_ms'], 0)

    def test_runtime_failure_logs_500_without_exception_payload(self):
        self.engine.recognize.side_effect = RuntimeError('PRIVATE_ERROR_CONTENT')
        response, event = self.request()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(event['http_status'], 500)
        self.assertEqual(event['error_type'], 'RuntimeError')
        self.assertNotIn('PRIVATE_ERROR_CONTENT', json.dumps(event))


if __name__ == '__main__':
    unittest.main()
