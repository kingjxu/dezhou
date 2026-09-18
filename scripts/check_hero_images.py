"""Serial, labeled image regression and core latency measurement (no HTTP).

Run from the repository: python scripts/check_hero_images.py --images DIR --output DIR
Image hashes are checked before execution. Expected values are manually reviewed,
not generated from the recognizer. Output preserves every response and mismatch.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.engines import get_recognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'reports/hero-regression/manifest.json')
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())['cases']
    # Fail before the run if an input is missing, changed or duplicated.
    seen = set()
    for case in cases:
        name = case['file']
        if Path(name).name != name or name in seen:
            raise ValueError(f'Invalid or duplicate filename: {name}')
        seen.add(name)
        if hashlib.sha256((args.images / name).read_bytes()).hexdigest() != case['sha256']:
            raise ValueError(f'Image hash mismatch: {name}')
    if not cases:
        raise ValueError('Empty manifest')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    recognizer = get_recognizer('polerv2')
    init_seconds = time.perf_counter() - started
    timings, failures = [], []
    with (args.output / 'results.jsonl').open('w') as output:
        for case in cases:
            encoded = base64.b64encode((args.images / case['file']).read_bytes()).decode()
            started = time.perf_counter()
            result = recognizer.recognize(encoded, parse_all=True)
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            errors = []
            for section, expected in case['expected'].items():
                actual = result[section]
                if section == 'villains_info':
                    actual = {str(v['seat']): v for v in actual}
                    for seat, fields in expected.items():
                        for key, value in fields.items():
                            got = actual.get(seat, {}).get(key)
                            if got != value:
                                errors.append([f'{section}.{seat}.{key}', value, got])
                else:
                    for key, value in expected.items():
                        if actual.get(key) != value:
                            errors.append([f'{section}.{key}', value, actual.get(key)])
            row = dict(index=case['index'], file=case['file'], seconds=elapsed,
                       errors=errors, result=result)
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            if errors:
                failures.append(dict(index=case['index'], errors=errors))
            print(case['index'], round(elapsed, 3), errors, flush=True)
    summary = dict(images=len(cases), passed=len(cases)-len(failures),
                   initialization_seconds=init_seconds,
                   mean_seconds=statistics.mean(timings),
                   median_seconds=statistics.median(timings),
                   p95_seconds=sorted(timings)[math.ceil(len(timings)*.95)-1],
                   max_seconds=max(timings), failures=failures)
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
