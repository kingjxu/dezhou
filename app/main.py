from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException

from app.engines import get_recognizer, supported_apps
from app.schemas import RecognizeRequest, RecognizeResponse
from app.recognition_log import record


app = FastAPI(title="德州扑克牌图片元素识别后端", version="2.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "supported_apps": supported_apps()}


@app.post("/recognize", response_model=RecognizeResponse)
def recognize(req: RecognizeRequest,
              x_recognition_source: str = Header(default="live")) -> dict:
    received_at = datetime.now(timezone.utc)
    t0 = perf_counter()
    payload = None
    elapsed_ms = None
    http_status = 500
    error_type = None
    try:
        recognizer = get_recognizer(req.app)
        payload = recognizer.recognize(req.image_base64, parse_all=req.parse_all)
        parsed_at = datetime.now(timezone.utc)
        elapsed_ms = int(round((perf_counter() - t0) * 1000))
        http_status = 200
        return {
            "received_at": received_at,
            "parsed_at": parsed_at,
            "elapsed_ms": elapsed_ms,
            "app": req.app,
            **payload,
        }
    except ValueError as e:
        http_status = 400
        error_type = type(e).__name__
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        http_status = 500
        error_type = type(e).__name__
        raise HTTPException(status_code=500, detail=f"recognize failed: {e}") from e
    finally:
        event = {
            "event": "recognition",
            "request_id": uuid4().hex,
            "received_at": received_at.isoformat(),
            "app": req.app[:64],
            "parse_all": req.parse_all,
            "source": "validation" if x_recognition_source == "validation" else "live",
            "input_chars": len(req.image_base64),
            "http_status": http_status,
            "elapsed_ms": (elapsed_ms if elapsed_ms is not None
                           else round((perf_counter() - t0) * 1000)),
        }
        if http_status == 200:
            event["result"] = payload
        else:
            event["error_type"] = error_type
        record(event)
