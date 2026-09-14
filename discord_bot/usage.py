"""Tiny local usage ledger for the Discord bot.

This is application-side accounting. It is not an OpenAI/Hugging Face remaining-quota
counter; provider quotas are controlled by each provider account.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

PATH = Path(__file__).with_name("usage.json")
LOCK = Lock()


def add(report: dict) -> dict:
    with LOCK:
        data = json.loads(PATH.read_text()) if PATH.exists() else {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        data["requests"] += 1
        for item in report.get("ai", {}).get("usage", []):
            data["input_tokens"] += int(item.get("input_tokens", 0) or 0)
            data["output_tokens"] += int(item.get("output_tokens", 0) or 0)
            data["total_tokens"] += int(item.get("total_tokens", 0) or 0)
        PATH.write_text(json.dumps(data, indent=2))
        return data


def current() -> dict:
    with LOCK:
        return json.loads(PATH.read_text()) if PATH.exists() else {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
