import cv2
import numpy as np


def _skin_prior(image: np.ndarray) -> np.ndarray:
    """Build a conservative skin-color prior used only to isolate the visible finger."""
    if image.ndim != 3:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    h, s, v = cv2.split(hsv)
    _, cr, cb = cv2.split(ycrcb)

    # Broad skin ranges; intentionally followed by connected-component filtering.
    skin_hsv = ((s > 20) & (v > 45) & (h < 35)).astype(np.uint8) * 255
    skin_ycc = ((cr > 125) & (cr < 190) & (cb > 75) & (cb < 145)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(skin_hsv, skin_ycc)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _single_finger_mask(image: np.ndarray) -> np.ndarray:
    """Isolate one visible finger and reject text, objects, and background texture."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    prior = _skin_prior(image)

    candidates = []
    for source in (prior,):
        n, labels, stats, cents = cv2.connectedComponentsWithStats(source, 8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < max(500, 0.01 * h * w):
                continue
            x, y, ww, hh = stats[i, :4]
            cx, cy = cents[i]
            centrality = 1.0 - min(1.0, np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2))
            elongation = max(ww, hh) / max(1, min(ww, hh))
            score = area * (0.55 + 0.45 * centrality) * (1.0 + min(1.5, elongation) * 0.15)
            candidates.append((score, (labels == i).astype(np.uint8) * 255))

    if candidates:
        mask = max(candidates, key=lambda x: x[0])[1]
    else:
        # Color may fail on monochrome/poorly lit images. Use a central GrabCut seed.
        work = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gc = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
        margin_x, margin_y = max(4, w // 12), max(4, h // 12)
        gc[margin_y:h - margin_y, margin_x:w - margin_x] = cv2.GC_PR_FGD
        rect = (max(1, w // 8), max(1, h // 8), max(2, 3 * w // 4), max(2, 3 * h // 4))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(work, gc, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
            mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            mask = np.zeros((h, w), dtype=np.uint8)

    # Keep only the strongest central component and smooth its boundary.
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        best_i, best_score = 0, -1.0
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 0.005 * h * w:
                continue
            cx, cy = cents[i]
            centrality = 1.0 - min(1.0, np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2))
            score = area * (0.45 + centrality)
            if score > best_score:
                best_score, best_i = score, i
        if best_i:
            mask = (labels == best_i).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill internal holes so printed letters/marks cannot become separate foreground objects.
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    mask = cv2.bitwise_or(mask, holes)
    return mask


def _crop_to_region(image: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return image, mask
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    pad = max(8, int(0.04 * max(x1 - x0, y1 - y0)))
    x0, x1 = max(0, x0 - pad), min(image.shape[1], x1 + pad)
    y0, y1 = max(0, y0 - pad), min(image.shape[0], y1 + pad)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def _gabor_precision(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Enhance existing ridge-like frequencies without inventing new pixels."""
    grayf = gray.astype(np.float32) / 255.0
    responses = []
    for theta in np.linspace(0, np.pi, 12, endpoint=False):
        kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 9.0, 0.55, 0, ktype=cv2.CV_32F)
        response = cv2.filter2D(grayf, cv2.CV_32F, kernel)
        responses.append(np.abs(response))
    response = np.max(np.stack(responses, axis=0), axis=0)
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    response[mask == 0] = 0
    return response


def _clean_ridges(ridges: np.ndarray, mask: np.ndarray, mode: int) -> np.ndarray:
    ridges = cv2.bitwise_and(ridges, mask)
    # Remove tiny components, with stronger cleanup in precision mode.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ridges, 8)
    clean = np.zeros_like(ridges)
    min_area = max(18 if mode == 1 else 28, (ridges.shape[0] * ridges.shape[1]) // (120000 if mode == 1 else 80000))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255

    k = (2, 2) if mode == 1 else (3, 3)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k))
    clean[mask == 0] = 0
    return clean


def extract_ridges(image: np.ndarray, mode: int = 1):
    """Return (ridge image, single-finger mask, isolated-finger-on-white preview)."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    finger_mask = _single_finger_mask(image)
    isolated = np.full_like(image, 255)
    isolated[finger_mask > 0] = image[finger_mask > 0]
    crop, crop_mask = _crop_to_region(isolated, finger_mask)
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.2 if mode == 1 else 2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(crop_gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31 if mode == 1 else 25, 4 if mode == 1 else 3
    )

    if mode == 2:
        gabor = _gabor_precision(enhanced, crop_mask)
        _, gabor_binary = cv2.threshold(gabor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ridges = cv2.bitwise_and(adaptive, gabor_binary)
    else:
        ridges = adaptive

    ridges = _clean_ridges(ridges, crop_mask, mode)
    return ridges, finger_mask, isolated


def make_preview(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = image.copy()
    if preview.ndim == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
    clean = np.zeros_like(preview)
    clean[mask > 0] = preview[mask > 0]
    clean[mask == 0] = 255
    return clean


def analyze_quality(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    score = min(100.0, 50.0 * min(1.0, sharpness / 300.0) + 50.0 * min(1.0, contrast / 60.0))
    label = "Good" if score >= 70 else "Fair" if score >= 40 else "Poor"
    return {"sharpness": sharpness, "contrast": contrast, "score": score, "label": label}


def process_fingerprint(path: str, mode: int = 1):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Could not read the image")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result, region, isolated = extract_ridges(rgb, mode=mode)
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise ValueError("Could not encode result")
    return {
        "image": result,
        "isolated_finger": isolated,
        "png_bytes": encoded.tobytes(),
        "ridge_coverage": float(np.count_nonzero(result)) / result.size * 100.0,
        "region": region,
    }
