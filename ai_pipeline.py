"""AI orchestration for the fingerprint visualization project.

Gemini is the visual-analysis provider for the realistic branch. Hugging Face/Qwen
is an optional, strictly separate prediction/review branch. Neither AI is allowed
to synthesize or overwrite the realistic ridge image.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AIUsage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    status: str = "not-run"
    error: str | None = None


@dataclass
class AIResult:
    provider: str
    model: str
    text: str
    usage: AIUsage
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "text": self.text, "usage": asdict(self.usage)}


def _data_url(image_bytes: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def gemini_vision(image_bytes: bytes, mime_type: str, prompt: str) -> AIResult:
    """Use Gemini as the visual/preprocessing specialist."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    usage = AIUsage("gemini", model, status="failed")
    if not key:
        usage.error = "GEMINI_API_KEY is not configured"
        raise RuntimeError(usage.error)

    payload = {"contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}},
    ]}], "generationConfig": {"temperature": 0.1}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:700]
        raise RuntimeError(f"Gemini API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini connection error: {exc.reason}") from exc

    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned no guidance text")
    meta = data.get("usageMetadata", {})
    usage.input_tokens = int(meta.get("promptTokenCount", 0) or 0)
    usage.output_tokens = int(meta.get("candidatesTokenCount", 0) or 0)
    usage.total_tokens = int(meta.get("totalTokenCount", usage.input_tokens + usage.output_tokens) or 0)
    usage.status = "ok"
    return AIResult("gemini", model, text, usage, data)


def qwen_review(image_bytes: bytes, mime_type: str, prompt: str) -> AIResult:
    """Call Qwen through Hugging Face's OpenAI-compatible HTTP endpoint.

    The OpenAI SDK is NOT used here; Hugging Face is the provider.
    """
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct:fastest")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")

    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime_type)}},
        ]}],
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        "https://router.huggingface.co/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:700]
        raise RuntimeError(f"Hugging Face API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Hugging Face connection error: {exc.reason}") from exc

    choices = data.get("choices", [])
    text = ((choices[0].get("message", {}).get("content", "") if choices else "") or "").strip()
    if not text:
        raise RuntimeError("Hugging Face returned no review text")
    raw_usage = data.get("usage", {}) or {}
    inp = int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0)) or 0)
    out = int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0)) or 0)
    total = int(raw_usage.get("total_tokens", inp + out) or 0)
    usage = AIUsage("huggingface", model, inp, out, total, "ok")
    return AIResult("huggingface", model, text, usage, data)


REALISTIC_PROMPT = """
You are the visual-analysis stage of a fingerprint ridge visualization pipeline.
Analyze only the pixels actually present. Return concise JSON with finger_visible,
background_level, blur, ridge_visibility, recommended_zoom (2 or 4), and processing_notes.
Do not identify or name anyone, match fingerprints, infer identity, or invent/reconstruct missing ridge detail.
This is preprocessing guidance only; a local deterministic CV pipeline produces the realistic image.
""".strip()

PREDICTION_PROMPT = """
This is the separate experimental prediction + reality branch. Clearly separate direct observations from hypotheses
about weak, occluded, shadowed, or contaminated areas. Suggest only non-generative CV operations. Never claim a
predicted ridge is authentic. This output is advisory and MUST NOT overwrite the realistic result.
Do not identify or match a person.
""".strip()

REVIEW_PROMPT = """
Act as an independent quality reviewer of this image-processing result. Report ridge visibility, border contamination,
background artifacts, over-cleaning risk, and PASS or REVIEW. Do not identify anyone, match fingerprints, or invent
missing biometric detail. Keep observations and limitations explicit.
""".strip()


def run_ai_review_stack(image_bytes: bytes, mime_type: str, realistic_png: bytes, run_prediction: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"realistic": {}, "prediction": {}, "review": {}, "usage": []}
    try:
        r = gemini_vision(image_bytes, mime_type, REALISTIC_PROMPT)
        result["realistic"] = r.as_dict()
        result["usage"].append(asdict(r.usage))
    except Exception as exc:
        result["realistic"] = {"status": "unavailable", "error": str(exc)}

    if run_prediction and os.getenv("HF_TOKEN"):
        for key, prompt in (("prediction", PREDICTION_PROMPT), ("review", REVIEW_PROMPT)):
            try:
                r = qwen_review(realistic_png, "image/png", prompt)
                result[key] = r.as_dict()
                result["usage"].append(asdict(r.usage))
            except Exception as exc:
                result[key] = {"status": "unavailable", "error": str(exc)}
    return result
