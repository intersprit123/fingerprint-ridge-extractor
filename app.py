from __future__ import annotations

import io
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from ai_pipeline import run_ai_review_stack
from processor import analyze_quality, make_preview, process_pipeline

load_dotenv()
st.set_page_config(page_title="Fingerprint Ridge Extractor", page_icon="🖐️", layout="wide")
st.title("🖐️ Fingerprint Ridge Extractor")
st.caption("Local-first, detail-preserving visualization with a strictly separated AI prediction branch.")

uploaded = st.file_uploader("Upload a fingertip photo", type=["jpg", "jpeg", "png", "webp"])
mode = st.radio("Realistic processing", ["Mode 1 — Realistic", "Mode 2 — Realistic + Precision"], index=1, horizontal=True)
mode_number = 1 if mode.startswith("Mode 1") else 2
c1, c2 = st.columns(2)
with c1: use_ai = st.checkbox("Run Gemini visual guidance", value=True)
with c2: prediction_branch = st.checkbox("Run separate Prediction + Reality review", value=False)
st.info("🟢 Realistic output is never overwritten by the prediction branch. Prediction is advisory/experimental only.")

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
        st.subheader("Local quality")
        st.metric("Quality score", f"{quality['score']:.0f}/100")
        st.write(f"**{quality['label']}** · sharpness {quality['sharpness']:.0f} · contrast {quality['contrast']:.1f}")

    if st.button("Run professional extraction", type="primary", use_container_width=True):
        zoom_scale = 4 if min(image.size) < 900 else 2
        gemini_text = None
        if use_ai:
            try:
                from ai_pipeline import REALISTIC_PROMPT, gemini_vision
                guidance = gemini_vision(data, uploaded.type or "image/jpeg", REALISTIC_PROMPT)
                gemini_text = guidance.text
                if "ZOOM=4" in gemini_text.upper(): zoom_scale = 4
                elif "ZOOM=2" in gemini_text.upper(): zoom_scale = 2
            except Exception as exc:
                st.warning(f"Gemini guidance unavailable; local zoom selection used. {exc}")

        with st.spinner(f"Running deterministic CV pipeline at {zoom_scale}x…"):
            pipeline = process_pipeline(image_array, mode=mode_number, zoom_scale=zoom_scale)
        stage1, stage2, stage3, stage4 = (pipeline[k] for k in ("stage1", "stage2", "stage3", "stage4"))
        mask, zoomed, actual_scale = pipeline["mask"], pipeline["zoomed"], pipeline["zoom_scale"]

        st.success(f"Realistic pipeline complete · Task 1 → 2 → 3 → 4 · {actual_scale}x")
        st.header("🟢 Realistic pipeline")
        a, b = st.columns(2)
        with a:
            st.subheader("Task 1 — Background removed")
            st.image(stage1, use_container_width=True)
            st.caption("One visible finger is isolated and the rest of the frame is removed.")
        with b:
            st.subheader("Task 2 — Extra marks suppressed")
            st.image(stage2, use_container_width=True)
            st.caption("Ridge-like evidence is extracted with local normalization and oriented filtering.")
        c, d = st.columns(2)
        with c:
            st.subheader("Task 3 — Finger border removed")
            st.image(stage3, use_container_width=True)
        with d:
            st.subheader("Task 4 — Final dark ridge representation")
            st.image(stage4, use_container_width=True)

        ok, encoded = cv2.imencode(".png", stage4)
        if ok:
            st.header("📥 Download realistic result")
            st.download_button("Download fingerprint_ridges_realistic.png", encoded.tobytes(), "fingerprint_ridges_realistic.png", "image/png", use_container_width=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Ridge coverage", f"{np.count_nonzero(stage4 < 128) / stage4.size * 100:.1f}%")
        m2.metric("Finger-mask coverage", f"{np.count_nonzero(mask) / mask.size * 100:.1f}%")
        m3.metric("Automatic zoom", f"{actual_scale}x")
        with st.expander("Selected finger / enlarged source"): st.image(zoomed, use_container_width=True)
        with st.expander("Original-frame isolation"): st.image(make_preview(image_array, mask), use_container_width=True)
        if gemini_text:
            with st.expander("Gemini visual guidance"): st.code(gemini_text)

        if prediction_branch and ok:
            st.header("🟠 Prediction + Reality — separate branch")
            st.warning("This branch is NOT merged into the realistic image. Predicted detail is hypothetical and must not be treated as authentic ridge evidence.")
            with st.spinner("Running Qwen Vision prediction and Hugging Face review…"):
                ai = run_ai_review_stack(data, uploaded.type or "image/jpeg", encoded.tobytes(), run_prediction=True, include_gemini=False)
            pred, review = ai.get("prediction", {}), ai.get("review", {})
            if pred:
                st.subheader("Qwen Vision prediction/reasoning")
                st.code(pred.get("text", "No prediction returned."))
            if review:
                st.subheader("Hugging Face final review")
                st.code(review.get("text", "No review returned."))
            report = json.dumps(ai, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button("Download AI review report", report, "fingerprint_ai_review.json", "application/json", use_container_width=True)
        else:
            st.caption("Prediction branch is off. The realistic output remains purely local CV processing.")
else:
    st.info("Upload a photo to begin.")

st.warning("Use only fingerprints you own or are authorized to process. This project visualizes visible ridge structure and does not identify people, perform fingerprint matching, or bypass biometric authentication.")
