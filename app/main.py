from __future__ import annotations

from datetime import datetime, timezone
import hmac
import os
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import ValidationError

from app.engines import get_recognizer, supported_apps
from app.schemas import RecognizeRequest, RecognizeResponse
from app.recognition_log import record


app = FastAPI(title="德州扑克牌图片元素识别后端", version="2.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "supported_apps": supported_apps()}


@app.post("/recognize", response_model=RecognizeResponse)
def recognize(req: RecognizeRequest,
              x_recognition_source: str = Header(default="live"),
              x_recognition_token: str = Header(default="")) -> dict:
    received_at = datetime.now(timezone.utc)
    t0 = perf_counter()
    payload = None
    elapsed_ms = None
    http_status = 500
    error_type = None
    validation_token = os.environ.get("DEZHOU_VALIDATION_TOKEN", "")
    is_validation = (
        x_recognition_source == "validation"
        and bool(validation_token)
        and isinstance(x_recognition_token, str)
        and hmac.compare_digest(x_recognition_token.encode(), validation_token.encode())
    )
    try:
        recognizer = get_recognizer(req.app)
        payload = recognizer.recognize(req.image_base64, parse_all=req.parse_all)
        parsed_at = datetime.now(timezone.utc)
        elapsed_ms = int(round((perf_counter() - t0) * 1000))
        response = RecognizeResponse(**{
            "received_at": received_at,
            "parsed_at": parsed_at,
            "elapsed_ms": elapsed_ms,
            "app": req.app,
            **payload,
        }).model_dump(mode="json")
        payload = {key: response[key] for key in
                   ("table_info", "hero_info", "villains_info")}
        http_status = 200
        return response
    except ValidationError as e:
        # A malformed engine result is a server error, not bad user input.
        http_status = 500
        error_type = type(e).__name__
        raise HTTPException(status_code=500, detail="invalid recognition result") from e
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
            "source": "validation" if is_validation else "live",
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
