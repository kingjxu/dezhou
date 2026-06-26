from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

from fastapi import FastAPI, HTTPException

from app.engines import get_recognizer, supported_apps
from app.schemas import RecognizeRequest, RecognizeResponse


app = FastAPI(title="德州扑克牌图片元素识别后端", version="2.0.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "supported_apps": supported_apps()}


@app.post("/recognize", response_model=RecognizeResponse)
def recognize(req: RecognizeRequest) -> dict:
    received_at = datetime.now(timezone.utc)
    t0 = perf_counter()
    try:
        recognizer = get_recognizer(req.app)
        payload = recognizer.recognize(req.image_base64, parse_all=req.parse_all)
        parsed_at = datetime.now(timezone.utc)
        elapsed_ms = int(round((perf_counter() - t0) * 1000))
        return {
            "received_at": received_at,
            "parsed_at": parsed_at,
            "elapsed_ms": elapsed_ms,
            "app": req.app,
            **payload,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"recognize failed: {e}") from e
