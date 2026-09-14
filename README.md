# Fingerprint Ridge Extractor

A small, local-first computer-vision tool for extracting **visible fingerprint ridge structure** from an authorized fingertip photograph.

## What it does

1. Accepts a fingertip image.
2. Enhances local ridge/valley contrast with OpenCV.
3. Segments visible ridge structure.
4. Removes small isolated artifacts and image-edge artifacts.
5. Produces a black-background image containing the extracted visible ridge structures.
6. Optionally asks an OpenAI vision model for image-quality and preprocessing guidance.

The OpenAI step is advisory. Pixel extraction remains deterministic and local so the system does not ask a generative model to invent biometric ridge detail.

## Safety

Use this project only with fingerprints you own or have explicit permission to process. It is intended for image processing and research/education, not for identifying people or bypassing biometric authentication.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Optional OpenAI setup

Set `OPENAI_API_KEY` in your environment or Streamlit secrets. Never commit the key.

The app uses an OpenAI vision-capable model through the Responses API for quality guidance; see the OpenAI model documentation for current model availability.

## Roadmap

- Better automatic fingertip ROI detection
- Finger orientation normalization
- Ridge-frequency estimation
- Quality scoring from local image features
- Optional minutiae visualization for authorized research images
- Automated regression tests using synthetic ridge patterns
