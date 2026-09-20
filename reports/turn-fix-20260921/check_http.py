"""Validate full and gated responses for both reported screenshots, in process."""
import base64
import json
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from app.main import app

rows = []
with TestClient(app) as client:
    for name, expected in [('check-action.webp', True), ('no-actions.webp', False)]:
        encoded = base64.b64encode((ROOT / 'fixtures/turn-gate' / name).read_bytes()).decode()
        for parse_all in (False, True):
            started = perf_counter()
            response = client.post('/recognize', json={
                'app': 'polerv2', 'parse_all': parse_all, 'image_base64': encoded})
            seconds = perf_counter() - started
            response.raise_for_status()
            result = response.json()
            assert result['app'] == 'polerv2', result
            assert isinstance(result['elapsed_ms'], int) and result['elapsed_ms'] >= 0, result
            assert datetime.fromisoformat(result['received_at']) <= datetime.fromisoformat(result['parsed_at']), result
            assert result['hero_info']['is_hero_turn'] is expected, result
            if expected:
                assert result['hero_info']['hero_cards'] == ['A 方块', '4 红桃'], result
                assert result['table_info']['community_cards'] == ['K 方块', 'J 方块', '9 梅花'], result
            elif not parse_all:
                assert result['hero_info']['hero_cards'] == [], result
            rows.append(dict(file=name, parse_all=parse_all, http_status=response.status_code,
                             seconds=seconds, response=result))
print(json.dumps(rows, ensure_ascii=False, indent=2))
