import base64
import io
import json
import os
import urllib.error
import urllib.request

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from processor import analyze_quality, extract_ridges, make_preview

st.set_page_config(page_title="Fingerprint Ridge Extractor", page_icon="🖐️", layout="wide")

st.title("🖐️ Fingerprint Ridge Extractor")
st.caption("Automatic contactless ridge visualization for images you are authorized to process.")

uploaded = st.file_uploader("Upload a fingertip photo", type=["jpg", "jpeg", "png", "webp"])
use_gemini = st.checkbox("Use Gemini for image-quality guidance", value=True)


def gemini_quality_guidance(image_bytes: bytes) -> str:
    """Ask Gemini for image-processing advice only; ridge extraction stays local."""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = {
        "contents": [{
            "parts": [
                {
                    "text": (
                        "Assess this fingertip photo only for image-processing quality. "
                        "Return concise JSON with quality 0-100, whether a fingertip is clearly visible, "
                        "and preprocessing advice. Do not identify or match the person. "
                        "Do not infer identity or generate missing fingerprint detail."
                    )
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                },
            ]
        }]
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
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


if uploaded:
    data = uploaded.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    image_array = np.array(image)
    quality = analyze_quality(image_array)
    with right:
        st.subheader("Image quality")
        st.metric("Local quality", f"{quality['score']:.0f}/100")
        st.write(f"**{quality['label']}** — sharpness {quality['sharpness']:.0f}, contrast {quality['contrast']:.1f}")

    if st.button("Extract fingerprint ridges", type="primary", use_container_width=True):
        guidance = None

        if use_gemini:
            # Send a JPEG representation to keep the request compact and predictable.
            jpeg_buffer = io.BytesIO()
            image.save(jpeg_buffer, format="JPEG", quality=92)
            try:
                guidance = gemini_quality_guidance(jpeg_buffer.getvalue())
            except Exception as exc:
                st.warning(f"Gemini guidance unavailable; continuing locally: {exc}")

        with st.spinner("Detecting fingertip and extracting visible ridge structure…"):
            result, mask = extract_ridges(image_array)
            preview = make_preview(image_array, mask)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Automatic fingertip region")
            st.image(preview, use_container_width=True)
        with c2:
            st.subheader("Ridge-only result")
            st.image(result, use_container_width=True)
            ok, encoded = cv2.imencode(".png", result)
            if ok:
                st.download_button(
                    "Download PNG",
                    encoded.tobytes(),
                    "fingerprint_ridges.png",
                    "image/png",
                    use_container_width=True,
                )
            st.metric("Ridge coverage", f"{float(np.count_nonzero(result)) / result.size * 100:.1f}%")

        if guidance:
            st.subheader("Gemini quality guidance")
            st.code(guidance, language="json")
else:
    st.info("Upload a photo to begin.")

st.warning("This tool is for processing authorized images and visualization only. It does not identify people, perform identity matching, or bypass biometric authentication.")
