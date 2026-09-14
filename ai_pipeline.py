"""AI orchestration. AI reviews pixels; deterministic OpenCV owns the realistic output."""
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

def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

def _usage(provider: str, model: str, obj: Any) -> AIUsage:
    def n(name: str) -> int:
        try: return int(getattr(obj, name, 0) or 0)
        except (TypeError, ValueError): return 0
    inp = n("input_tokens") or n("prompt_tokens")
    out = n("output_tokens") or n("completion_tokens")
    return AIUsage(provider, model, inp, out, n("total_tokens") or inp + out, "ok")

def gemini_vision(image_bytes: bytes, mime_type: str, prompt: str) -> AIResult:
    import urllib.error
    import urllib.request
    key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    if not key: raise RuntimeError("GEMINI_API_KEY is not configured")
    payload = {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}}]}], "generationConfig": {"temperature": 0.1}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as response: data = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Gemini API error: {exc.read().decode(errors='replace')[:700]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini connection error: {exc.reason}") from exc
    text = "".join(p.get("text", "") for p in data.get("candidates", [{}])[0].get("content", {}).get("parts", [])).strip()
    meta = data.get("usageMetadata", {})
    usage = AIUsage("gemini", model, int(meta.get("promptTokenCount", 0) or 0), int(meta.get("candidatesTokenCount", 0) or 0), int(meta.get("totalTokenCount", 0) or 0), "ok")
    return AIResult("gemini", model, text, usage, data)

def openai_vision(image_bytes: bytes, mime_type: str, prompt: str) -> AIResult:
    from openai import OpenAI
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(model=model, input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": _data_url(image_bytes, mime_type)}]}])
    return AIResult("openai", model, (response.output_text or "").strip(), _usage("openai", model, getattr(response, "usage", None)), {})

def qwen_review(image_bytes: bytes, mime_type: str, prompt: str) -> AIResult:
    from openai import OpenAI
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct:fastest")
    if not token: raise RuntimeError("HF_TOKEN is not configured")
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=token)
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime_type)}}]}], temperature=0.1)
    return AIResult("huggingface", model, (response.choices[0].message.content or "").strip(), _usage("huggingface", model, getattr(response, "usage", None)), {})

REALISTIC_PROMPT = """Analyze only the pixels actually present for conservative computer-vision preprocessing. Return JSON with finger_visible, background_level, blur, ridge_visibility, recommended_zoom (2 or 4), and processing_notes. Do not identify/name anyone, match fingerprints, infer identity, or reconstruct missing ridge detail. A deterministic local CV pipeline produces the final realistic image.""".strip()
PREDICTION_PROMPT = """Separate experimental review only. Inspect the supplied fingerprint visualization and clearly separate observations from hypotheses about weak, occluded, shadowed, or contaminated regions. Return JSON with observations, predictions, risks, and suggested non-generative CV operations. Predicted detail is NOT authentic ridge evidence. This branch MUST NOT overwrite the realistic result.""".strip()
REVIEW_PROMPT = """Independently review this image-processing result. Report ridge visibility, border contamination, background artifacts, over-cleaning risk, and PASS/REVIEW. Do not identify anyone, match fingerprints, or invent missing detail.""".strip()

def run_ai_review_stack(image_bytes: bytes, mime_type: str, realistic_png: bytes, run_prediction: bool = False, include_gemini: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"realistic": {}, "prediction": {}, "review": {}, "usage": []}
    if include_gemini and os.getenv("GEMINI_API_KEY"):
        try:
            r = gemini_vision(image_bytes, mime_type, REALISTIC_PROMPT); result["realistic"] = r.as_dict(); result["usage"].append(asdict(r.usage))
        except Exception as exc: result["realistic"] = {"status": "unavailable", "error": str(exc)}
    if os.getenv("ENABLE_OPENAI_REVIEW", "0") == "1" and os.getenv("OPENAI_API_KEY"):
        try:
            r = openai_vision(image_bytes, mime_type, "Conservative visual quality analysis. " + REALISTIC_PROMPT); result["openai"] = r.as_dict(); result["usage"].append(asdict(r.usage))
        except Exception as exc: result["openai"] = {"status": "unavailable", "error": str(exc)}
    if os.getenv("HF_TOKEN"):
        if run_prediction:
            try:
                r = qwen_review(realistic_png, "image/png", PREDICTION_PROMPT); result["prediction"] = r.as_dict(); result["usage"].append(asdict(r.usage))
            except Exception as exc: result["prediction"] = {"status": "unavailable", "error": str(exc)}
        try:
            r = qwen_review(realistic_png, "image/png", REVIEW_PROMPT); result["review"] = r.as_dict(); result["usage"].append(asdict(r.usage))
        except Exception as exc: result["review"] = {"status": "unavailable", "error": str(exc)}
    return result
