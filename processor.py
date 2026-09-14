from pathlib import Path

import cv2
import numpy as np


def process_fingerprint(path: str):
    """Extract visible ridge structure without inventing missing biometric detail."""
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not read the image")

    # Local contrast enhancement and mild denoising.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Adaptive threshold keeps the pipeline usable across uneven lighting.
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 4
    )
    ridges = 255 - binary

    # Remove tiny isolated artifacts.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ridges, 8)
    clean = np.zeros_like(ridges)
    min_area = max(12, (gray.shape[0] * gray.shape[1]) // 200000)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    # Suppress a thin image boundary where the finger silhouette can appear.
    margin = max(8, min(clean.shape) // 40)
    clean[:margin] = 0
    clean[-margin:] = 0
    clean[:, :margin] = 0
    clean[:, -margin:] = 0

    # Black background, white visible ridge structures.
    ok, encoded = cv2.imencode(".png", clean)
    if not ok:
        raise ValueError("Could not encode result")

    return {
        "image": clean,
        "png_bytes": encoded.tobytes(),
        "ridge_coverage": float(np.count_nonzero(clean)) / clean.size * 100.0,
    }
