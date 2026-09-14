import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from processor import analyze_quality, extract_ridges, make_preview

load_dotenv()

st.set_page_config(page_title="Fingerprint Ridge Extractor", page_icon="🖐️", layout="wide")
st.title("🖐️ Fingerprint Ridge Extractor")
st.caption("Single-finger isolation, automatic zoom, and visible ridge visualization for images you are authorized to process.")

uploaded = st.file_uploader("Upload a fingertip photo", type=["jpg", "jpeg", "png", "webp"])
mode = st.radio("Processing mode", ["Mode 1 — Realistic", "Mode 2 — Realistic + Precision"], horizontal=True)
mode_number = 1 if mode.startswith("Mode 1") else 2
use_gemini = st.checkbox("Use Gemini for image-quality guidance + automatic zoom recommendation", value=True)


def gemini_quality_guidance(image_bytes: bytes, mime_type: str) -> str:
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = (
        "Assess this image only for computer-vision preprocessing. Report whether one finger is clearly visible, "
        "background contamination, blur, lighting, and recommend exactly 2x or 4x zoom. "
        "Return the recommendation as ZOOM=2x or ZOOM=4x somewhere in the response. "
        "Do not identify, name, or match any person and do not reconstruct missing biometric detail."
    )
    payload = {"contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}},
    ]}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Gemini API: {exc.reason}") from exc
    parts = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError("Gemini returned no guidance text")
    return text


def recommended_zoom(guidance: str | None, image: Image.Image) -> int:
    if guidance:
        match = re.search(r"ZOOM\s*=\s*(2|4)x", guidance, flags=re.I)
        if match:
            return int(match.group(1))
    # Conservative fallback: low-resolution images benefit more from the 4x processing canvas.
    return 4 if min(image.size) < 900 else 2


if uploaded:
    data = uploaded.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image_array = np.array(image)
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with right:
        quality = analyze_quality(image_array)
        st.subheader("Image quality")
        st.metric("Local quality", f"{quality['score']:.0f}/100")
        st.write(f"**{quality['label']}** — sharpness {quality['sharpness']:.0f}, contrast {quality['contrast']:.1f}")

    if st.button("Extract single-finger ridges", type="primary", use_container_width=True):
        guidance = None
        if use_gemini:
            try:
                guidance = gemini_quality_guidance(data, uploaded.type or "image/jpeg")
            except Exception as exc:
                st.warning(f"Gemini guidance unavailable; using local automatic zoom: {exc}")
        zoom_scale = recommended_zoom(guidance, image)

        with st.spinner(f"Isolating one finger, removing background, auto-zooming {zoom_scale}x, and extracting visible ridge structure…"):
            result, mask, zoomed, actual_scale = extract_ridges(image_array, mode=mode_number, zoom_scale=zoom_scale)
            preview = make_preview(image_array, mask)

        st.success(f"Automatic finger zoom: {actual_scale}x — processing is performed on the isolated finger crop.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"1. Isolated finger — {actual_scale}x zoom")
            st.image(zoomed, use_container_width=True)
            st.caption("Background is replaced by white before ridge processing. The upscale enlarges captured pixels; it does not invent missing ridge detail.")
        with c2:
            st.subheader(f"2. Ridge-only — {mode}")
            st.image(result, use_container_width=True)
            ok, encoded = cv2.imencode(".png", result)
            if ok:
                st.download_button("Download ridge PNG", encoded.tobytes(), "fingerprint_ridges.png", "image/png", use_container_width=True)

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Ridge pixel coverage", f"{float(np.count_nonzero(result)) / result.size * 100:.1f}%")
        with d2:
            st.metric("Finger-mask coverage", f"{float(np.count_nonzero(mask)) / mask.size * 100:.1f}%")
        with d3:
            st.metric("Auto zoom", f"{actual_scale}x")

        with st.expander("Show original-frame finger isolation"):
            st.image(preview, use_container_width=True)
        if guidance:
            st.subheader("Gemini preprocessing guidance")
            st.code(guidance)
else:
    st.info("Upload a photo to begin.")

st.warning("Use only fingerprints you own or are authorized to process. This tool visualizes visible ridge structure and does not identify people, perform fingerprint matching, or bypass biometric authentication.")
