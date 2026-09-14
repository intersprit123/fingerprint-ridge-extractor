from pathlib import Path
import tempfile

import streamlit as st
from openai import OpenAI

from processor import process_fingerprint

st.set_page_config(page_title="Fingerprint Ridge Extractor", page_icon="🖐️", layout="wide")

st.title("🖐️ Fingerprint Ridge Extractor")
st.caption("Local OpenCV processing with optional OpenAI image-quality analysis. Use only fingerprints you are authorized to process.")

uploaded = st.file_uploader("Upload a fingertip photo", type=["jpg", "jpeg", "png", "webp"])
use_openai = st.checkbox("Use OpenAI for image-quality / ROI guidance", value=True)

if uploaded:
    data = uploaded.getvalue()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(data, use_container_width=True)

    guidance = None
    if use_openai and st.button("Analyze + Extract", type="primary"):
        api_key = st.secrets.get("OPENAI_API_KEY") or __import__("os").getenv("OPENAI_API_KEY")
        if not api_key:
            st.warning("OPENAI_API_KEY is not configured. Running local extraction only.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                response = client.responses.create(
                    model="gpt-5.6-luna",
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Assess this fingertip photo for ridge-processing quality. Return concise JSON with quality (0-100), whether a fingertip region is clearly visible, and practical preprocessing advice. Do not identify the person or infer identity."},
                            {"type": "input_image", "image_url": "data:image/jpeg;base64," + __import__("base64").b64encode(data).decode("ascii")},
                        ],
                    }],
                )
                guidance = response.output_text
            except Exception as exc:
                st.warning(f"OpenAI analysis unavailable; continuing locally: {exc}")

        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as f:
            f.write(data)
            input_path = f.name
        result = process_fingerprint(input_path)

        with col2:
            st.subheader("Ridge-only result")
            st.image(result["image"], use_container_width=True)
            st.download_button("Download PNG", result["png_bytes"], "fingerprint_ridges.png", "image/png")
            st.metric("Ridge coverage", f"{result['ridge_coverage']:.1f}%")

        if guidance:
            st.subheader("OpenAI quality guidance")
            st.code(guidance, language="json")
else:
    st.info("Upload a photo to begin.")
