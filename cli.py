"""Command-line entry point for local fingerprint visualization."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from service import process_bytes, report_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Fingerprint Ridge Extractor — authorized image visualization")
    parser.add_argument("image", help="Input JPG/PNG/WebP")
    parser.add_argument("-o", "--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--mode", type=int, choices=(1, 2), default=2)
    parser.add_argument("--zoom", type=int, choices=(2, 4), default=2)
    parser.add_argument("--prediction", action="store_true", help="Run the separate Qwen/Hugging Face prediction/review branch")
    args = parser.parse_args()

    source = Path(args.image)
    if not source.is_file():
        raise SystemExit(f"Input not found: {source}")

    data = source.read_bytes()
    mime = Image.open(source).get_format()
    mime_type = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(mime or "", "image/jpeg")
    result = process_bytes(data, mime_type, args.mode, args.zoom, args.prediction)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fingerprint_ridges_realistic.png").write_bytes(result["stages"]["stage4"])
    (out / "fingerprint_report.json").write_bytes(report_json(result["report"]))

    print(f"Realistic result: {out / 'fingerprint_ridges_realistic.png'}")
    print(f"Report:           {out / 'fingerprint_report.json'}")
    print(f"Quality:          {result['report']['quality']['score']:.0f}/100")
    print(f"Zoom:             {result['report']['zoom_scale']}x")
    print(f"Prediction:       {'ON' if args.prediction else 'OFF'}")


if __name__ == "__main__":
    main()
