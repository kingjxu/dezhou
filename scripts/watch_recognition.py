"""Watch JSONL recognition logs over SSH and notify the requested botmux session.

No image payloads or SSH passwords are stored. Use an existing SSH key/agent or
ControlMaster socket. Offline replay requires --dry-run and sends no messages.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import selectors
import shlex
import subprocess
import time


def event_lines(stream, deadline):
    selector = selectors.DefaultSelector()
    selector.register(stream, selectors.EVENT_READ)
    pending = b''
    try:
        while time.monotonic() < deadline:
            if not selector.select(timeout=min(1, max(0, deadline-time.monotonic()))):
                yield None
                continue
            chunk = stream.read1(65536)
            if not chunk:
                raise ConnectionError('SSH 日志连接已关闭')
            pending += chunk
            while b'\n' in pending:
                line, pending = pending.split(b'\n', 1)
                yield line
            if len(pending) > 1024 * 1024:
                pending = b''  # Ignore malformed oversized log lines.
    finally:
        selector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh', help='SSH user@host')
    parser.add_argument('--control-path')
    parser.add_argument('--remote-log', default='/home/ubuntu/poker/dezhou/logs/recognition.jsonl')
    parser.add_argument('--session-id')
    parser.add_argument('--duration', type=int, default=1800)
    parser.add_argument('--summary-interval', type=int, default=300)
    parser.add_argument('--slow-ms', type=int, default=3000)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--input-file', type=Path, help='Offline replay (requires --dry-run)')
    args = parser.parse_args()
    if min(args.duration, args.summary_interval, args.slow_ms) <= 0:
        parser.error('duration, summary-interval and slow-ms must be positive')
    if args.input_file and not args.dry_run:
        parser.error('--input-file requires --dry-run')
    if not args.input_file and not args.ssh:
        parser.error('--ssh is required for live monitoring')
    if not args.dry_run and not args.session_id:
        parser.error('--session-id is required for notifications')

    def notify(message):
        print(json.dumps({'notification': message}, ensure_ascii=False), flush=True)
        if args.dry_run:
            return
        # Notification delivery is independent of the SSH reader. A failed
        # channel must not kill monitoring or recursively notify its own failure.
        for attempt in range(1, 4):
            try:
                subprocess.run(['botmux', 'send', '--session-id', args.session_id,
                                '--no-mention', '--text', message], check=True, timeout=5)
                return
            except (OSError, subprocess.SubprocessError) as error:
                print(json.dumps({'notification_error': type(error).__name__,
                                  'attempt': attempt}, ensure_ascii=False), flush=True)
        print(json.dumps({'notification_dropped': message}, ensure_ascii=False), flush=True)

    process = None
    started = time.monotonic()
    deadline = started + args.duration
    count = errors = slow = 0
    elapsed_sum = 0.0
    recent = deque(maxlen=1000)
    last_summary = started
    summarized_count = 0
    last_alert = {'error': -float('inf'), 'slow': -float('inf')}

    def summary(reason):
        if not count:
            return f'dezhou 监控{reason}：未收到真实识别请求（验证流量不计入）。'
        p95 = sorted(recent)[math.ceil(len(recent) * .95)-1]
        return (f'dezhou 监控{reason}：真实请求 {count}，错误 {errors}，'
                f'耗时≥{args.slow_ms}ms 的请求 {slow}；平均 {elapsed_sum/count:.0f}ms，'
                f'最近 {len(recent)} 条 P95 {p95:.0f}ms。')

    try:
        if args.input_file:
            lines = args.input_file.read_bytes().splitlines()
        else:
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']
            if args.control_path:
                command += ['-S', args.control_path]
            command += [args.ssh, 'tail -n 0 --sleep-interval=1 --max-unchanged-stats=1 -F -- ' + shlex.quote(args.remote_log)]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines = event_lines(process.stdout, deadline)
        for line in lines:
            now = time.monotonic()
            if line is not None:
                try:
                    event = json.loads(line)
                    if event.get('event') != 'recognition' or event.get('source') == 'validation':
                        continue
                    elapsed = float(event['elapsed_ms'])
                    status = int(event['http_status'])
                    if not math.isfinite(elapsed) or elapsed < 0:
                        continue
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
                count += 1
                elapsed_sum += elapsed
                recent.append(elapsed)
                errors += status >= 400
                slow += elapsed >= args.slow_ms
                kind = 'error' if status >= 400 else 'slow' if elapsed >= args.slow_ms else None
                if count == 1 or (kind and now-last_alert[kind] >= 60):
                    result = event.get('result') or {}
                    hero = result.get('hero_info', {})
                    table = result.get('table_info', {})
                    prefix = '收到首笔真实识别流量' if count == 1 else '识别错误' if kind == 'error' else '慢请求'
                    notify(f'dezhou {prefix}：{elapsed:.0f}ms，HTTP {status}，app={event.get("app")}。'
                           f'手牌={hero.get("hero_cards")}，公共牌={table.get("community_cards")}，'
                           f'筹码={hero.get("stack")}，下注={hero.get("current_bet")}。'
                           f'request_id={event.get("request_id")}')
                    if kind:
                        last_alert[kind] = now
            if now-last_summary >= args.summary_interval and count != summarized_count:
                notify(summary('阶段汇总'))
                last_summary, summarized_count = now, count
        notify(summary('结束'))
    except (ConnectionError, subprocess.SubprocessError) as error:
        notify(summary('中断') + f' 原因：{type(error).__name__}；监控已停止。')
        raise
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    main()
