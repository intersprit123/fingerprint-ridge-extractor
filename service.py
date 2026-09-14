"""Shared application service used by Streamlit, HTTP API, and Discord."""
from __future__ import annotations

import io
import json
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ai_pipeline import run_ai_review_stack
from processor import analyze_quality, process_pipeline


def process_bytes(image_bytes: bytes, mime_type: str = "image/jpeg", mode: int = 2, zoom_scale: int = 2, prediction: bool = False) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(image)
    quality = analyze_quality(rgb)
    pipeline = process_pipeline(rgb, mode=mode, zoom_scale=zoom_scale)

    stages: dict[str, bytes] = {}
    for name, value in (("stage1", pipeline["stage1"]), ("stage2", pipeline["stage2"]), ("stage3", pipeline["stage3"]), ("stage4", pipeline["stage4"])):
        if value.ndim == 3:
            value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", value)
        if ok:
            stages[name] = encoded.tobytes()

    realistic = stages["stage4"]
    ai = run_ai_review_stack(image_bytes, mime_type, realistic, run_prediction=prediction)
    report = {
        "quality": quality,
        "mode": mode,
        "zoom_scale": pipeline["zoom_scale"],
        "ridge_coverage_percent": float(np.count_nonzero(pipeline["stage4"] < 128)) / pipeline["stage4"].size * 100.0,
        "ai": ai,
        "integrity": {
            "realistic_result_is_local_cv": True,
            "prediction_branch_overwrites_realistic": False,
            "generated_ridge_detail": False,
        },
    }
    return {"stages": stages, "report": report}


def report_json(report: dict[str, Any]) -> bytes:
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
