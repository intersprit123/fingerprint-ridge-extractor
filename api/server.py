"""Small HTTP bridge for external tools.

POST /process with multipart field `image`, optional `mode`, `zoom_scale`, and `prediction`.
The response contains a downloadable result URL plus the usage/quality report.
"""
from __future__ import annotations

import base64
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from service import process_bytes, report_json

app = FastAPI(title="Fingerprint Ridge Extractor API", version="3.0.0")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./api_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))


def _save(job_id: str, suffix: str, payload: bytes) -> str:
    path = OUTPUT_DIR / f"{job_id}_{suffix}"
    path.write_bytes(payload)
    return f"/files/{path.name}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "fingerprint-ridge-extractor"}


@app.post("/process")
async def process(
    image: UploadFile = File(...),
    mode: int = Form(1),
    zoom_scale: int = Form(2),
    prediction: bool = Form(False),
):
    if mode not in (1, 2):
        raise HTTPException(400, "mode must be 1 or 2")
    if zoom_scale not in (2, 4):
        raise HTTPException(400, "zoom_scale must be 2 or 4")
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"image exceeds {MAX_BYTES} bytes")

    job_id = f"{int(time.time())}_{secrets.token_hex(4)}"
    try:
        result = process_bytes(data, image.content_type or "image/jpeg", mode, zoom_scale, prediction)
    except Exception as exc:
        raise HTTPException(422, f"processing failed: {exc}") from exc

    final_url = _save(job_id, "fingerprint_ridges_final.png", result["stages"]["stage4"])
    report_url = _save(job_id, "report.json", report_json(result["report"]))
    return {
        "status": "success",
        "job_id": job_id,
        "download": final_url,
        "report": report_url,
        "quality": result["report"]["quality"],
        "usage": result["report"]["ai"]["usage"],
        "integrity": result["report"]["integrity"],
    }


@app.get("/files/{filename}")
def files(filename: str):
    path = (OUTPUT_DIR / filename).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path)
