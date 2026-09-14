# Fingerprint Ridge Extractor

A local-first computer-vision tool for visualizing **visible fingerprint ridge structure** from close-up fingertip photographs that the operator is authorized to process.

## Architecture

The project has two deliberately isolated tracks:

### 🟢 Realistic track — source of truth

`Original → Gemini Vision guidance → OpenCV → realistic PNG`

Gemini is the visual-analysis specialist: finger visibility, background contamination, blur, ridge visibility, and zoom guidance. OpenCV performs the actual segmentation, normalization, ridge filtering, border removal, and final rendering. No AI-generated pixels are merged into the realistic result.

### 🟠 Prediction + Reality track — experimental only

`Realistic PNG → Qwen Vision through Hugging Face → prediction/reasoning + review`

This branch is never allowed to overwrite the realistic PNG. Predictions are clearly labeled as hypotheses and are not treated as authentic fingerprint detail.

## Four deterministic stages

1. **Background removed** — select one visible finger and put it on a clean white background.
2. **Extra marks suppressed** — local normalization, black-hat enhancement, adaptive thresholding, and multi-angle Gabor evidence.
3. **Finger border removed** — distance-transform inner ROI removes the silhouette.
4. **Final dark representation** — connected-component cleanup and clean ridge rendering.

The processor does not synthesize missing biometric detail.

## AI providers

- **Gemini**: the only AI used for the realistic visual-guidance stage.
- **Qwen through Hugging Face Inference Providers**: optional separate prediction/reasoning and independent review.
- **OpenAI Vision is not used by this project.**

The provider layer records input/output/total token usage when the provider returns usage metadata. The Discord bot keeps a local application-side ledger. That ledger is **not** a provider's remaining quota; provider quotas are controlled by the provider account.

## Configuration

Copy `.env.example` to `.env` and add your own secrets. Never commit `.env`, API keys, Discord tokens, or biometric images.

```env
GEMINI_API_KEY=YOUR_GEMINI_KEY
GEMINI_MODEL=gemini-2.5-flash
HF_TOKEN=YOUR_HF_TOKEN
HF_MODEL=Qwen/Qwen2.5-VL-32B-Instruct:fastest
DISCORD_BOT_TOKEN=YOUR_DISCORD_BOT_TOKEN
OUTPUT_DIR=./api_outputs
MAX_IMAGE_BYTES=12582912
```

## Install

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Streamlit

```bash
streamlit run app.py
```

The UI provides a direct download button for the realistic PNG and, when the prediction branch is enabled, a downloadable JSON AI review report.

## CLI

```bash
python cli.py photo.jpg --mode 2 --zoom 4
```

Outputs:

```text
outputs/fingerprint_ridges_realistic.png
outputs/fingerprint_report.json
```

For the separate prediction/review branch:

```bash
python cli.py photo.jpg --prediction
```

## HTTP API

Start the API:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/process \
  -F "image=@photo.jpg" \
  -F "mode=2" \
  -F "zoom_scale=4" \
  -F "prediction=false"
```

The JSON response contains `download` and `report` URLs. Open the returned download URL to retrieve the PNG.

## Discord

The repository contains an optional Discord interface. It requires your own Discord application and bot token; no token is stored in GitHub.

```bash
python -m discord_bot.bot
```

Slash commands:

```text
/fingerprint image:<attachment> prediction:<true|false>
/usage
```

A successful request returns the realistic PNG and JSON report as Discord attachments. `/usage` is local accounting only, not a provider remaining-quota value.

## Downloads

- **Source code:** use GitHub's Code → Download ZIP, or the main-branch ZIP URL.
- **Streamlit result:** click the Download button.
- **CLI result:** open the files in the `outputs/` directory.
- **HTTP API result:** open the `download` URL returned by `/process`.
- **Discord result:** download the PNG attachment from the bot response.

## Quality and limitations

A phone photograph may not contain enough information for reliable fingerprint capture. Enhancement cannot recover ridge information that was never recorded. Background segmentation is heuristic and can fail on unusual lighting, skin-like backgrounds, gloves, multiple overlapping fingers, or severe blur.

## Acceptable use

Use only fingerprints you own or have explicit permission to process. This project intentionally does not implement person identification, fingerprint matching, authentication bypass, or identity inference.
