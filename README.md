# Fingerprint Ridge Extractor

A local-first computer-vision tool for visualizing **visible fingerprint ridge structure** from close-up fingertip photographs that the operator is authorized to process.

## Architecture

The project has two deliberately isolated tracks:

### 🟢 Realistic track — source of truth

`Original → Gemini guidance → OpenCV → realistic PNG`

Gemini may recommend crop/zoom and preprocessing. OpenCV performs the actual segmentation, normalization, ridge filtering, border removal, and final rendering. No AI-generated pixels are merged into the realistic result.

### 🟠 Prediction + Reality track — experimental only

`Realistic PNG → Qwen Vision → prediction/reasoning → Hugging Face review`

This branch is never allowed to overwrite the realistic PNG. Predictions are clearly labeled as hypotheses and are not treated as authentic fingerprint detail.

## Four deterministic stages

1. **Background removed** — select one visible finger and put it on a clean white background.
2. **Extra marks suppressed** — local normalization, black-hat enhancement, adaptive thresholding, and multi-angle Gabor evidence.
3. **Finger border removed** — distance-transform inner ROI removes the silhouette.
4. **Final dark representation** — connected-component cleanup and clean ridge rendering.

The processor does not synthesize missing biometric detail.

## AI providers

- **Gemini**: first visual/preprocessing analysis.
- **OpenAI**: optional premium visual review. The OpenAI Responses API supports image inputs. See the official API quickstart. 
- **Qwen through Hugging Face Inference Providers**: separate prediction/reasoning and independent final review. Hugging Face provides an OpenAI-compatible VLM endpoint and supports models such as Qwen2.5-VL.

The provider layer records input/output/total token usage when the provider returns usage metadata. The Discord bot also keeps a local application-side ledger. That ledger is **not** a provider's remaining quota; provider quotas are controlled by the provider account.

## Discord

The bot supports:

```text
/fingerprint image:<attachment> prediction:<true|false>
/usage
```

A successful request returns:

- `fingerprint_ridges_final.png` — downloadable realistic result
- `fingerprint_report.json` — quality, AI usage, and integrity metadata
- Discord embed with provider/model usage and local ledger totals

## HTTP API

Run:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Then POST an image to `/process` as multipart form data. The response contains `download`, `report`, quality metrics, usage, and integrity flags. `/health` provides a simple health check.

## Configuration

Copy `.env.example` to `.env` and add your own secrets. Never commit `.env`, API keys, Discord tokens, or biometric images.

```env
GEMINI_API_KEY=YOUR_GEMINI_KEY
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=YOUR_OPENAI_KEY
OPENAI_MODEL=gpt-5.6-luna
ENABLE_OPENAI_REVIEW=0
HF_TOKEN=YOUR_HF_TOKEN
HF_MODEL=Qwen/Qwen2.5-VL-32B-Instruct:fastest
DISCORD_BOT_TOKEN=YOUR_DISCORD_BOT_TOKEN
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Quality and limitations

A phone photograph may not contain enough information for reliable fingerprint capture. Enhancement cannot recover ridge information that was never recorded. Background segmentation is heuristic and can fail on unusual lighting, skin-like backgrounds, gloves, multiple overlapping fingers, or severe blur.

## Acceptable use

Use only fingerprints you own or have explicit permission to process. This project intentionally does not implement person identification, fingerprint matching, authentication bypass, or identity inference.
