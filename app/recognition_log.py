"""Bounded JSONL recognition logs, separate from server access logs."""
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

LOG_PATH = Path(os.environ.get(
    'DEZHOU_RECOGNITION_LOG',
    str(Path(__file__).resolve().parents[1] / 'logs' / 'recognition.jsonl'),
))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger('dezhou.recognition')
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024,
                                  backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)


def record(event: dict) -> None:
    # Keep one JSON record per line; never include request images or base64.
    logger.info(json.dumps(event, ensure_ascii=False, separators=(',', ':'), default=str))
