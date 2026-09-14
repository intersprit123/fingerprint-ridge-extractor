# Fingerprint Ridge Extractor

A local-first computer-vision tool for visualizing **visible fingerprint ridge structure** from close-up fingertip photographs that the operator is authorized to process.

## V1 pipeline

1. Upload JPG, JPEG, PNG, or WebP.
2. Estimate the foreground/fingertip region automatically.
3. Crop to a conservative interior of the detected region.
4. Enhance local ridge/valley contrast with OpenCV.
5. Segment visible ridge structure.
6. Remove small artifacts and image-boundary noise.
7. Produce a black-background, white-ridge PNG.
8. Optionally use **Google Gemini** for image-quality/preprocessing guidance.

The Gemini step is advisory. Pixel extraction remains local and deterministic so a generative model is not asked to invent missing biometric ridge detail.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Gemini configuration

Create a Gemini API key through Google's AI Studio, then set `GEMINI_API_KEY` in your environment or Streamlit secrets. Optionally set `GEMINI_MODEL`; the default is `gemini-2.5-flash`. The app calls Gemini directly over HTTPS, so no OpenAI API key is required.

Linux/macOS:

```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

Never commit an API key or biometric images.

## Limitations

A normal phone photo may not contain enough information for reliable fingerprint capture. Enhancement cannot recover ridge information that was never recorded. Results are for visualization and image-processing experiments, not forensic identification or authentication.

## Acceptable use

Use only fingerprints you own or have explicit permission to process. This project intentionally does not implement person identification, fingerprint matching, authentication bypass, or identity inference.
