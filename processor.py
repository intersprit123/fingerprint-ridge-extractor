import cv2
import numpy as np


def _skin_prior(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    h, s, v = cv2.split(hsv)
    _, cr, cb = cv2.split(ycrcb)
    skin_hsv = ((s > 20) & (v > 45) & (h < 35)).astype(np.uint8) * 255
    skin_ycc = ((cr > 125) & (cr < 190) & (cb > 75) & (cb < 145)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(skin_hsv, skin_ycc)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)


def _single_finger_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    prior = _skin_prior(image)
    candidates = []
    n, labels, stats, cents = cv2.connectedComponentsWithStats(prior, 8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max(500, 0.01 * h * w):
            continue
        cx, cy = cents[i]
        centrality = 1.0 - min(1.0, np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2))
        ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        elongation = max(ww, hh) / max(1, min(ww, hh))
        score = area * (0.55 + 0.45 * centrality) * (1.0 + min(1.5, elongation) * 0.15)
        candidates.append((score, (labels == i).astype(np.uint8) * 255))
    if candidates:
        mask = max(candidates, key=lambda x: x[0])[1]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        work = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gc = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
        mx, my = max(4, w // 12), max(4, h // 12)
        gc[my:h - my, mx:w - mx] = cv2.GC_PR_FGD
        rect = (max(1, w // 8), max(1, h // 8), max(2, 3 * w // 4), max(2, 3 * h // 4))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(work, gc, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
            mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            mask = np.zeros((h, w), dtype=np.uint8)
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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def _crop_to_region(image: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return image, mask
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = max(8, int(0.04 * max(x1 - x0, y1 - y0)))
    x0, x1 = max(0, x0 - pad), min(image.shape[1], x1 + pad)
    y0, y1 = max(0, y0 - pad), min(image.shape[0], y1 + pad)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def auto_zoom(image: np.ndarray, mask: np.ndarray, scale: int = 2):
    crop, crop_mask = _crop_to_region(image, mask)
    scale = 4 if int(scale) >= 4 else 2
    size = (max(1, crop.shape[1] * scale), max(1, crop.shape[0] * scale))
    zoomed = cv2.resize(crop, size, interpolation=cv2.INTER_LANCZOS4)
    zoomed_mask = cv2.resize(crop_mask, size, interpolation=cv2.INTER_NEAREST)
    blur = cv2.GaussianBlur(zoomed, (0, 0), 1.0)
    zoomed = cv2.addWeighted(zoomed, 1.12, blur, -0.12, 0)
    return zoomed, zoomed_mask, scale


def _gabor_precision(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grayf = gray.astype(np.float32) / 255.0
    responses = []
    for theta in np.linspace(0, np.pi, 16, endpoint=False):
        kernel = cv2.getGaborKernel((25, 25), 4.2, theta, 8.0, 0.55, 0, ktype=cv2.CV_32F)
        responses.append(np.abs(cv2.filter2D(grayf, cv2.CV_32F, kernel)))
    response = np.max(np.stack(responses, axis=0), axis=0)
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    response[mask == 0] = 0
    return response


def _clean_ridges(ridges: np.ndarray, mask: np.ndarray, mode: int) -> np.ndarray:
    ridges = cv2.bitwise_and(ridges, mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ridges, 8)
    clean = np.zeros_like(ridges)
    min_area = max(10 if mode == 1 else 7, (ridges.shape[0] * ridges.shape[1]) // (160000 if mode == 1 else 220000))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    k = (2, 2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k))
    if mode == 1:
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k))
    clean[mask == 0] = 0
    return clean


def extract_ridges(image: np.ndarray, mode: int = 1, zoom_scale: int = 2):
    if image.ndim != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    finger_mask = _single_finger_mask(image)
    isolated = np.full_like(image, 255)
    isolated[finger_mask > 0] = image[finger_mask > 0]
    zoomed, zoomed_mask, actual_scale = auto_zoom(isolated, finger_mask, zoom_scale)
    gray = cv2.cvtColor(zoomed, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2 if mode == 1 else 2.4, tileGridSize=(8, 8))
    enhanced = cv2.GaussianBlur(clahe.apply(gray), (3, 3), 0)

    if mode == 1:
        ridges = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4)
    else:
        # Mode 2 used to AND the adaptive and Gabor masks, which erased real ridges.
        # It now uses Gabor as a confidence/continuity guide while retaining adaptive strokes.
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4)
        gabor = _gabor_precision(enhanced, zoomed_mask)
        vals = gabor[zoomed_mask > 0]
        p58 = int(np.percentile(vals, 58)) if vals.size else 35
        p78 = int(np.percentile(vals, 78)) if vals.size else 50
        confidence = (gabor >= max(30, p58)).astype(np.uint8) * 255
        support = cv2.dilate(confidence, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        supported = cv2.bitwise_and(adaptive, support)
        strong = (gabor >= max(45, p78)).astype(np.uint8) * 255
        ridges = cv2.bitwise_or(supported, cv2.bitwise_and(strong, zoomed_mask))
        # Keep adaptive ridge components that connect to the precision result.
        bridge = cv2.dilate(ridges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        ridges = cv2.bitwise_or(ridges, cv2.bitwise_and(adaptive, bridge))

    ridges = _clean_ridges(ridges, zoomed_mask, mode)
    return ridges, finger_mask, zoomed, actual_scale


def make_preview(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    clean = np.full_like(image, 255)
    clean[mask > 0] = image[mask > 0]
    return clean


def analyze_quality(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    score = min(100.0, 50.0 * min(1.0, sharpness / 300.0) + 50.0 * min(1.0, contrast / 60.0))
    label = "Good" if score >= 70 else "Fair" if score >= 40 else "Poor"
    return {"sharpness": sharpness, "contrast": contrast, "score": score, "label": label}


def process_fingerprint(path: str, mode: int = 1, zoom_scale: int = 2):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Could not read the image")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result, region, zoomed, actual_scale = extract_ridges(rgb, mode=mode, zoom_scale=zoom_scale)
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise ValueError("Could not encode result")
    return {"image": result, "isolated_finger": zoomed, "png_bytes": encoded.tobytes(), "ridge_coverage": float(np.count_nonzero(result)) / result.size * 100.0, "region": region, "zoom_scale": actual_scale}
