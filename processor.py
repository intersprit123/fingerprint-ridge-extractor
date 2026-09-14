import cv2
import numpy as np


def _largest_skin_region(gray: np.ndarray) -> np.ndarray:
    """Estimate a finger region from color-independent intensity information."""
    # Otsu gives a useful first separation for a close-up finger photo.
    smooth = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Try both polarities; keep the one with the largest plausible central component.
    candidates = [mask, 255 - mask]
    h, w = gray.shape
    best = None
    best_score = -1
    for candidate in candidates:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k, iterations=2)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k, iterations=1)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(candidate, 8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 0.03 * h * w:
                continue
            x, y, ww, hh = stats[i, :4]
            cx, cy = cents[i]
            centrality = 1.0 - min(1.0, np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2))
            score = area * (0.5 + centrality)
            if score > best_score:
                best_score = score
                best = (labels == i).astype(np.uint8) * 255

    return best if best is not None else np.full_like(gray, 255)


def _crop_to_region(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crop to a conservative interior of the detected finger region."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return gray

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    pad_x = max(4, int((x1 - x0) * 0.03))
    pad_y = max(4, int((y1 - y0) * 0.03))
    x0, x1 = max(0, x0 + pad_x), min(gray.shape[1], x1 - pad_x)
    y0, y1 = max(0, y0 + pad_y), min(gray.shape[0], y1 - pad_y)
    return gray[y0:y1, x0:x1]


def extract_ridges(image: np.ndarray):
    """Extract visible ridge structure; never synthesize missing biometric detail."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    region = _largest_skin_region(gray)
    crop = _crop_to_region(gray, region)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(crop)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Adaptive thresholding handles uneven phone-camera illumination.
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 4
    )
    ridges = 255 - binary

    # Remove small components and thin border artifacts.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ridges, 8)
    clean = np.zeros_like(ridges)
    min_area = max(12, (crop.shape[0] * crop.shape[1]) // 200000)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    margin = max(6, min(clean.shape) // 45)
    clean[:margin] = 0
    clean[-margin:] = 0
    clean[:, :margin] = 0
    clean[:, -margin:] = 0

    # A small closing joins broken ridge fragments without adding new structure.
    clean = cv2.morphologyEx(
        clean, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    )

    return clean, region


def make_preview(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = image.copy()
    if preview.ndim == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
    overlay = preview.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    return cv2.addWeighted(preview, 0.75, overlay, 0.25, 0)


def analyze_quality(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    score = min(100.0, 50.0 * min(1.0, sharpness / 300.0) + 50.0 * min(1.0, contrast / 60.0))
    label = "Good" if score >= 70 else "Fair" if score >= 40 else "Poor"
    return {"sharpness": sharpness, "contrast": contrast, "score": score, "label": label}


def process_fingerprint(path: str):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Could not read the image")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result, region = extract_ridges(rgb)
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise ValueError("Could not encode result")
    return {
        "image": result,
        "png_bytes": encoded.tobytes(),
        "ridge_coverage": float(np.count_nonzero(result)) / result.size * 100.0,
        "region": region,
    }
