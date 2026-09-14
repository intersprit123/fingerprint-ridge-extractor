import base64
import io
import os

import streamlit as st
from openai import OpenAI
from PIL import Image

from processor import analyze_quality, extract_ridges, make_preview

st.set_page_config(page_title="Fingerprint Ridge Extractor", page_icon="🖐️", layout="wide")

st.title("🖐️ Fingerprint Ridge Extractor")
st.caption("Automatic contactless ridge visualization for images you are authorized to process.")

uploaded = st.file_uploader("Upload a fingertip photo", type=["jpg", "jpeg", "png", "webp"])
use_openai = st.checkbox("Use OpenAI for image-quality guidance", value=True)

if uploaded:
    data = uploaded.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")

    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)

    quality = analyze_quality(__import__("numpy").array(image))
    with right:
        st.subheader("Image quality")
        st.metric("Quality", f"{quality['score']:.0f}/100")
        st.write(f"**{quality['label']}** — sharpness {quality['sharpness']:.0f}, contrast {quality['contrast']:.1f}")

    if st.button("Extract fingerprint ridges", type="primary", use_container_width=True):
        guidance = None

        if use_openai:
            api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    client = OpenAI(api_key=api_key)
                    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
                    response = client.responses.create(
                        model=model,
                        input=[{
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Assess this fingertip photo only for image-processing quality. Return concise JSON with quality 0-100, whether a fingertip is clearly visible, and preprocessing advice. Do not identify or match the person."},
                                {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")},
                            ],
                        }],
                    )
                    guidance = response.output_text
                except Exception as exc:
                    st.warning(f"OpenAI guidance unavailable; continuing locally: {exc}")
            else:
                st.info("OPENAI_API_KEY is not configured, so extraction will run locally.")

        with st.spinner("Detecting fingertip and extracting visible ridge structure…"):
            result, mask = extract_ridges(__import__("numpy").array(image))
            preview = make_preview(__import__("numpy").array(image), mask)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Automatic fingertip region")
            st.image(preview, use_container_width=True)
        with c2:
            st.subheader("Ridge-only result")
            st.image(result, use_container_width=True)
            ok, encoded = __import__("cv2").imencode(".png", result)
            if ok:
                st.download_button("Download PNG", encoded.tobytes(), "fingerprint_ridges.png", "image/png", use_container_width=True)
            st.metric("Ridge coverage", f"{float(__import__('numpy').count_nonzero(result)) / result.size * 100:.1f}%")

        if guidance:
            st.subheader("OpenAI quality guidance")
            st.code(guidance, language="json")
else:
    st.info("Upload a photo to begin.")

st.warning("This tool is for processing authorized images and visualization only. It does not identify people, perform identity matching, or bypass biometric authentication.")
