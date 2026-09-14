import cv2
import numpy as np


def _skin_prior(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    h, s, v = cv2.split(hsv)
    _, cr, cb = cv2.split(ycrcb)
    skin_hsv = ((s > 15) & (v > 40) & (h < 40)).astype(np.uint8) * 255
    skin_ycc = ((cr > 120) & (cr < 195) & (cb > 70) & (cb < 150)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(skin_hsv, skin_ycc)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)


def _single_finger_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    prior = _skin_prior(image)
    candidates = []
    n, labels, stats, cents = cv2.connectedComponentsWithStats(prior, 8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max(500, 0.008 * h * w):
            continue
        cx, cy = cents[i]
        centrality = 1.0 - min(1.0, np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w / 2, h / 2))
        ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        elongation = max(ww, hh) / max(1, min(ww, hh))
        score = area * (0.60 + 0.40 * centrality) * (1.0 + min(1.5, elongation) * 0.12)
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
            cv2.grabCut(work, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
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
            score = area * (0.50 + centrality)
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
    zoomed = cv2.addWeighted(zoomed, 1.10, blur, -0.10, 0)
    return zoomed, zoomed_mask, scale


def _normalize(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    background = cv2.GaussianBlur(normalized, (0, 0), 9.0)
    normalized = cv2.divide(normalized, np.maximum(background, 1), scale=180)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _gabor_response(gray: np.ndarray, mask: np.ndarray, angles: int = 24) -> np.ndarray:
    grayf = gray.astype(np.float32) / 255.0
    responses = []
    for theta in np.linspace(0, np.pi, angles, endpoint=False):
        kernel = cv2.getGaborKernel((25, 25), 4.0, theta, 8.0, 0.55, 0, ktype=cv2.CV_32F)
        responses.append(np.abs(cv2.filter2D(grayf, cv2.CV_32F, kernel)))
    response = np.max(np.stack(responses, axis=0), axis=0)
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    response[mask == 0] = 0
    return response


def _ridge_core(mask: np.ndarray) -> np.ndarray:
    # A conservative inner ROI removes the silhouette before ridge detection.
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    positive = dist[dist > 0]
    if positive.size == 0:
        return np.zeros_like(mask)
    # Keep the interior, excluding a small percentage of the outer edge.
    margin = max(5.0, float(np.percentile(positive, 18)))
    core = (dist >= margin).astype(np.uint8) * 255
    return cv2.morphologyEx(
        core, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1
    )


def _stage2_lines(zoomed: np.ndarray, roi: np.ndarray, mode: int) -> np.ndarray:
    gray = cv2.cvtColor(zoomed, cv2.COLOR_RGB2GRAY)
    gray[roi == 0] = 255
    normalized = _normalize(gray)
    normalized[roi == 0] = 255

    # Dark-ridge enhancement suppresses broad skin shading and handwriting-like marks.
    blackhat = cv2.morphologyEx(
        normalized, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    )
    blackhat = cv2.GaussianBlur(blackhat, (3, 3), 0)
    bh_vals = blackhat[roi > 0]
    bh_thr = int(np.percentile(bh_vals, 72)) if bh_vals.size else 25
    dark_evidence = (blackhat >= max(18, bh_thr)).astype(np.uint8) * 255

    # Oriented ridge evidence. Mode 2 uses more orientations and a stricter percentile.
    gabor = _gabor_response(normalized, roi, 24 if mode == 2 else 16)
    gv = gabor[roi > 0]
    gp = 70 if mode == 2 else 62
    g_thr = int(np.percentile(gv, gp)) if gv.size else 50
    oriented = (gabor >= max(40, g_thr)).astype(np.uint8) * 255

    adaptive = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 4
    )
    adaptive = cv2.bitwise_and(adaptive, roi)

    # A pixel must have local line evidence and nearby ridge evidence.
    evidence = cv2.bitwise_or(dark_evidence, oriented)
    support = cv2.dilate(evidence, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    lines = cv2.bitwise_and(adaptive, support)

    # Preserve strong Gabor ridge portions that adaptive thresholding misses.
    strong_thr = int(np.percentile(gv, 82)) if gv.size else 70
    strong = (gabor >= max(55, strong_thr)).astype(np.uint8) * 255
    lines = cv2.bitwise_or(lines, cv2.bitwise_and(strong, roi))
    return cv2.bitwise_and(lines, roi)


def _remove_small_components(lines: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    clean = np.zeros_like(lines)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def _remove_border(lines: np.ndarray, finger_mask: np.ndarray) -> np.ndarray:
    # Remove the finger silhouette/border using distance from the segmented edge.
    dist = cv2.distanceTransform((finger_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    positive = dist[dist > 0]
    if positive.size == 0:
        return np.zeros_like(lines)
    margin = max(5.0, float(np.percentile(positive, 12)))
    inner = (dist >= margin).astype(np.uint8) * 255
    return cv2.bitwise_and(lines, inner)


def _final_representation(lines: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lines = cv2.bitwise_and(lines, mask)
    lines = cv2.morphologyEx(
        lines, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), 1
    )
    min_area = max(10, (lines.shape[0] * lines.shape[1]) // 220000)
    lines = _remove_small_components(lines, min_area)
    # Dark, clean ridge representation on white.
    final = np.full(lines.shape, 255, dtype=np.uint8)
    final[lines > 0] = 12
    return final


def process_pipeline(image: np.ndarray, mode: int = 2, zoom_scale: int = 2):
    if image.ndim != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # TASK 1: automatically select the best single finger and make all other pixels white.
    finger_mask = _single_finger_mask(image)
    stage1_full = np.full_like(image, 255)
    stage1_full[finger_mask > 0] = image[finger_mask > 0]
    zoomed, zoomed_mask, actual_scale = auto_zoom(stage1_full, finger_mask, zoom_scale)

    # TASK 2: automatically remove extra characters/marks/noise and retain ridge-like lines.
    stage2 = _stage2_lines(zoomed, zoomed_mask, mode)

    # TASK 3: remove the selected finger's outer border/silhouette.
    stage3 = _remove_border(stage2, zoomed_mask)

    # TASK 4: darken and represent the remaining lines cleanly.
    stage4 = _final_representation(stage3, zoomed_mask)

    stage1 = np.full_like(zoomed, 255)
    stage1[zoomed_mask > 0] = zoomed[zoomed_mask > 0]

    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
        "stage4": stage4,
        "mask": finger_mask,
        "zoomed": zoomed,
        "zoomed_mask": zoomed_mask,
        "zoom_scale": actual_scale,
    }


def extract_ridges(image: np.ndarray, mode: int = 2, zoom_scale: int = 2):
    pipeline = process_pipeline(image, mode=mode, zoom_scale=zoom_scale)
    return pipeline["stage4"], pipeline["mask"], pipeline["zoomed"], pipeline["zoom_scale"]


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


def process_fingerprint(path: str, mode: int = 2, zoom_scale: int = 2):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Could not read the image")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pipeline = process_pipeline(rgb, mode=mode, zoom_scale=zoom_scale)
    result = pipeline["stage4"]
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise ValueError("Could not encode result")
    return {
        "image": result,
        "isolated_finger": pipeline["zoomed"],
        "png_bytes": encoded.tobytes(),
        "ridge_coverage": float(np.count_nonzero(result < 128)) / result.size * 100.0,
        "region": pipeline["mask"],
        "zoom_scale": pipeline["zoom_scale"],
        "stages": pipeline,
    }
