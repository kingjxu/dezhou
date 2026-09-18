"""Check turn gating on original, half-size and letterboxed screenshots."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.engines import get_recognizer
from app.main import recognize
from app.schemas import RecognizeRequest, RecognizeResponse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'reports/hero-regression/manifest.json').read_text())
    cases = {case['index']: case for case in manifest['cases']}
    recognizer = get_recognizer('polerv2')
    checks = []
    # #1 has real action buttons. #34/#86 have gray pre-action controls.
    for index in (1, 34, 86):
        case = cases[index]
        path = args.images / case['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != case['sha256']:
            raise ValueError(f'Image hash mismatch: {path.name}')
        original = cv2.imread(str(path))
        for variant in ('original', 'half', 'letterbox'):
            image = original if variant == 'original' else cv2.resize(original, (540, 1200))
            if variant == 'letterbox':
                image = cv2.copyMakeBorder(image, 40, 40, 30, 30,
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
            _, data = cv2.imencode('.png', image)
            encoded = base64.b64encode(data).decode()
            started = time.perf_counter()
            if index == 1:
                actual = recognizer._detect_turn_fast(recognizer._normalize_canvas(image))
                assert actual is True, (index, variant, actual)
            else:
                result = recognize(RecognizeRequest(app='polerv2', image_base64=encoded,
                                                     parse_all=False))
                RecognizeResponse(**result)
                assert result['hero_info']['is_hero_turn'] is False, (index, variant, result)
                assert result['hero_info']['hero_cards'] == [], (index, variant, result)
                assert result['table_info']['stage'] == 'unknown', (index, variant, result)
            checks.append(dict(index=index, variant=variant,
                               seconds=time.perf_counter()-started, passed=True))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
