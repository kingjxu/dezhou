"""Exercise the CLI with fake external SSH/botmux processes, no notifications."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class WatchRecognitionTest(unittest.TestCase):
    def run_watcher(self, failures):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [dict(event='recognition', source='live', elapsed_ms=100,
                           http_status=200, request_id='first'),
                      dict(event='recognition', source='validation', elapsed_ms=10,
                           http_status=200),
                      dict(event='recognition', source='live', elapsed_ms=4000,
                           http_status=200, request_id='slow'),
                      dict(event='recognition', source='live', elapsed_ms=50,
                           http_status=500, request_id='error')]
            ssh = root / 'ssh'
            ssh.write_text('#!' + sys.executable + '\nimport time\n'
                           + 'print(' + repr('\n'.join(map(json.dumps, events)))
                           + ', flush=True)\ntime.sleep(10)\n')
            botmux = root / 'botmux'
            botmux.write_text('#!' + sys.executable + '\n'
                              'import os, pathlib, sys\n'
                              "p = pathlib.Path(os.environ['NOTIFY_ATTEMPTS'])\n"
                              "n = int(p.read_text()) if p.exists() else 0\n"
                              'p.write_text(str(n+1))\n'
                              'sys.exit(7 if n < ' + str(failures) + ' else 0)\n')
            ssh.chmod(0o755)
            botmux.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       NOTIFY_ATTEMPTS=str(root / 'attempts'))
            result = subprocess.run([
                sys.executable, str(Path(__file__).parent / 'scripts/watch_recognition.py'),
                '--ssh', 'fake-host', '--session-id', 'fake-session', '--duration', '1',
            ], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            lines = [json.loads(line) for line in result.stdout.splitlines()]
            summary = lines[-1]['notification'] if failures < 100 else next(
                item['notification'] for item in reversed(lines) if 'notification' in item)
            self.assertIn('真实请求 3，错误 1', summary)
            self.assertIn('平均 1383ms', summary)
            return lines, int((root / 'attempts').read_text())

    def test_first_failure_retries_and_later_events_continue(self):
        lines, attempts = self.run_watcher(1)
        self.assertEqual(attempts, 5)  # First + retry, slow, error, final summary.
        self.assertEqual(sum('notification_error' in item for item in lines), 1)
        self.assertFalse(any('notification_dropped' in item for item in lines))

    def test_exhausted_first_notification_recovers_on_later_events(self):
        lines, attempts = self.run_watcher(3)
        self.assertEqual(attempts, 6)
        self.assertEqual(sum('notification_dropped' in item for item in lines), 1)

    def test_persistent_failure_does_not_abort_or_recurse(self):
        lines, attempts = self.run_watcher(100)
        self.assertEqual(attempts, 12)  # Three tries per notification, including exit.
        self.assertEqual(sum('notification_dropped' in item for item in lines), 4)


if __name__ == '__main__':
    unittest.main()
