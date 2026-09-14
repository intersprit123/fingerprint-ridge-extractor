# Fingerprint Ridge Extractor

A local-first computer-vision tool for visualizing **visible fingerprint ridge structure** from close-up fingertip photographs that the operator is authorized to process.

## V2 pipeline

1. Upload JPG, JPEG, PNG, or WebP.
2. Build a conservative skin/finger foreground mask.
3. Select **one visible finger** using area and centrality rather than processing the whole frame.
4. Remove the rest of the background before ridge processing.
5. Fill the isolated finger onto a **white background**, so unrelated objects, text/characters, and background texture are excluded from the ridge stage.
6. Enhance local ridge/valley contrast with OpenCV.
7. Extract only ridge structure already present in the image.
8. Clean small disconnected artifacts and keep processing inside the finger mask.
9. Produce a black-background, white-ridge PNG.

### Processing modes

- **Mode 1 — Realistic:** conservative enhancement and adaptive thresholding, preserving more naturally visible ridge texture.
- **Mode 2 — Realistic + Precision:** Mode 1 plus multi-angle Gabor frequency filtering to suppress background-like texture and strengthen existing ridge structure.

The filtering is deterministic image processing. It does not synthesize missing biometric ridges.

## Gemini guidance

Optionally, Google Gemini analyzes the uploaded image only for preprocessing advice such as whether one finger is clearly visible, background contamination, blur, lighting, and which processing mode is likely to be more suitable. Gemini is **not** used to generate or reconstruct fingerprint ridges.

Create a Gemini API key through Google's AI Studio and keep it outside Git:

```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

The default model is `gemini-2.5-flash`; override it with `GEMINI_MODEL` if needed.

A local `.env` file is also supported:

```env
GEMINI_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL=gemini-2.5-flash
```

`.env` is ignored by Git. Never commit API keys or biometric images.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

A normal phone photo may not contain enough information for reliable fingerprint capture. Enhancement cannot recover ridge information that was never recorded. Background removal is heuristic and can fail on unusual lighting, skin-like backgrounds, gloves, or multiple overlapping fingers. Results are for visualization and image-processing experiments, not forensic identification or authentication.

## Acceptable use

Use only fingerprints you own or have explicit permission to process. This project intentionally does not implement person identification, fingerprint matching, authentication bypass, or identity inference.
