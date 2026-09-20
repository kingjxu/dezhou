"""Sequential, identical-runner A/B on the same 95 images; no production writes."""
import json
from pathlib import Path
import subprocess
import sys

STAGING = Path('/home/ubuntu/poker/dezhou-turn-20260921')
REPORT = STAGING / 'reports/turn-fix-20260921'
results = {}
for label, root in [('baseline', Path('/home/ubuntu/poker/dezhou')),
                    ('optimized', STAGING)]:
    output = REPORT / f'ab-{label}'
    with (REPORT / f'ab-{label}.log').open('w') as log:
        process = subprocess.run([
            sys.executable, str(STAGING / 'scripts/check_hero_images.py'),
            '--code-root', str(root), '--images', '/usr/local/webserver/images',
            '--manifest', str(REPORT / 'eight-seat-manifest.json'),
            '--output', str(output),
        ], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode not in (0, 1) or not (output / 'summary.json').exists():
        raise RuntimeError(f'{label} failed to finish: exit={process.returncode}')
    results[label] = json.loads((output / 'summary.json').read_text())
    print(label, results[label]['mean_seconds'], results[label]['passed'], flush=True)
a, b = results['baseline'], results['optimized']
assert a['run_finished_at'] <= b['run_started_at']
results['mean_change_percent'] = (b['mean_seconds']/a['mean_seconds']-1)*100
results['p95_change_percent'] = (b['p95_seconds']/a['p95_seconds']-1)*100
(REPORT / 'ab-summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in results.items() if k.endswith('percent')}, indent=2))
raise SystemExit(bool(b['failures']))
