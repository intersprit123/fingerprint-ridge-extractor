"""AI orchestration for the fingerprint visualization project.

AI models are used for visual analysis/review, never to synthesize missing ridge pixels.
The deterministic OpenCV pipeline remains the source of the realistic fingerprint image.
"""
from __future__ import annotations

import base64
import json
import os
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
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _usage_from_object(provider: str, model: str, usage: Any) -> AIUsage:
    def get(name: str) -> int:
        value = getattr(usage, name, 0) if usage is not None else 0
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    inp = get("input_tokens") or get("prompt_tokens")
    out = get("output_tokens") or get("completion_tokens")
    total = get("total_tokens") or inp + out
    return AIUsage(provider, model, inp, out, total, "ok")


def gemini_vision(image_bytes: bytes, mime_type: str, prompt: str) -> AIResult:
    """Ask Gemini for structured visual/preprocessing guidance only."""
    import urllib.error
    import urllib.request

    key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    usage = AIUsage("gemini", model, status="failed")
    if not key:
        usage.error = "GEMINI_API_KEY is not configured"
        raise RuntimeError(usage.error)

    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}},
        ]}],
        "generationConfig": {"temperature": 0.1},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:700]
        usage.error = f"HTTP {exc.code}: {detail}"
        raise RuntimeError(f"Gemini API error: {detail}") from exc
    except urllib.error.URLError as exc:
        usage.error = str(exc.reason)
        raise RuntimeError(f"Gemini connection error: {exc.reason}") from exc

    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts).strip()
    meta = data.get("usageMetadata", {})
    usage.input_tokens = int(meta.get("promptTokenCount", 0) or 0)
    usage.output_tokens = int(meta.get("candidatesTokenCount", 0) or 0)
    usage.total_tokens = int(meta.get("totalTokenCount", usage.input_tokens + usage.output_tokens) or 0)
    usage.status = "ok"
    return AIResult("gemini", model, text, usage, data)


def openai_vision(image_bytes: bytes, mime_type: str, prompt: str) -> AIResult:
    """Optional premium vision reviewer using the OpenAI Responses API."""
    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": _data_url(image_bytes, mime_type)},
            ],
        }],
    )
    text = (response.output_text or "").strip()
    usage = _usage_from_object("openai", model, getattr(response, "usage", None))
    return AIResult("openai", model, text, usage, {})


def qwen_review(image_bytes: bytes, mime_type: str, prompt: str) -> AIResult:
    """Run Qwen Vision through Hugging Face Inference Providers.

    This is a review/prediction branch. Its output is advisory and is never merged
    into the realistic ridge image.
    """
    from openai import OpenAI

    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct:fastest")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime_type)}},
            ],
        }],
        temperature=0.1,
    )
    text = (response.choices[0].message.content or "").strip()
    usage = _usage_from_object("huggingface", model, getattr(response, "usage", None))
    return AIResult("huggingface", model, text, usage, {})


REALISTIC_PROMPT = """
You are the visual-analysis stage of a fingerprint ridge visualization pipeline.
Analyze only the pixels actually present. Return concise JSON with:
- finger_visible (boolean)
- background_level (low/medium/high)
- blur (low/medium/high)
- ridge_visibility (low/medium/high)
- recommended_zoom (2 or 4)
- processing_notes (array of short notes)
Do not identify or name anyone, match fingerprints, infer identity, or invent/reconstruct missing ridge detail.
This is preprocessing guidance only; a local deterministic CV pipeline produces the final image.
""".strip()


PREDICTION_PROMPT = """
You are the separate experimental prediction/review stage. Inspect this fingerprint image and provide hypotheses
about where visible ridge evidence may be weak, occluded, shadowed, or contaminated. Clearly separate observations
from predictions. Return JSON with observations, predictions, risks, and suggested non-generative CV operations.
NEVER claim that predicted ridge detail is authentic. Do not identify or match a person.
This response is advisory only and MUST NOT overwrite the realistic result.
""".strip()


REVIEW_PROMPT = """
Act as an independent quality reviewer. Inspect this fingerprint visualization only as an image-processing result.
Report ridge visibility, border contamination, background artifacts, over-cleaning risk, and whether the result should
PASS or REVIEW. Do not identify anyone, match fingerprints, or invent missing biometric detail. Keep observations and
limitations explicit.
""".strip()


def run_ai_review_stack(image_bytes: bytes, mime_type: str, realistic_png: bytes, run_prediction: bool = False) -> dict[str, Any]:
    """Run optional AI stages while keeping realistic and predictive outputs separate."""
    result: dict[str, Any] = {"realistic": {}, "prediction": {}, "review": {}, "usage": []}

    try:
        r = gemini_vision(image_bytes, mime_type, REALISTIC_PROMPT)
        result["realistic"] = r.as_dict()
        result["usage"].append(asdict(r.usage))
    except Exception as exc:
        result["realistic"] = {"status": "unavailable", "error": str(exc)}

    if os.getenv("ENABLE_OPENAI_REVIEW", "0") == "1" and os.getenv("OPENAI_API_KEY"):
        try:
            r = openai_vision(image_bytes, mime_type, "Assess this image for conservative computer-vision preprocessing. " + REALISTIC_PROMPT)
            result["openai"] = r.as_dict()
            result["usage"].append(asdict(r.usage))
        except Exception as exc:
            result["openai"] = {"status": "unavailable", "error": str(exc)}

    if run_prediction and os.getenv("HF_TOKEN"):
        try:
            r = qwen_review(realistic_png, "image/png", PREDICTION_PROMPT)
            result["prediction"] = r.as_dict()
            result["usage"].append(asdict(r.usage))
        except Exception as exc:
            result["prediction"] = {"status": "unavailable", "error": str(exc)}

    if os.getenv("HF_TOKEN"):
        try:
            r = qwen_review(realistic_png, "image/png", REVIEW_PROMPT)
            result["review"] = r.as_dict()
            result["usage"].append(asdict(r.usage))
        except Exception as exc:
            result["review"] = {"status": "unavailable", "error": str(exc)}

    return result
