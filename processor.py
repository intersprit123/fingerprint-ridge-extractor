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
    zoomed = cv2.addWeighted(zoomed, 1.10, blur, -0.10, 0)
    return zoomed, zoomed_mask, scale


def _gabor_precision(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    grayf = gray.astype(np.float32) / 255.0
    responses = []
    # Fingerprint ridges are locally oriented and approximately periodic.
    # A bank of moderately narrow filters suppresses camera/background texture.
    for theta in np.linspace(0, np.pi, 24, endpoint=False):
        kernel = cv2.getGaborKernel(
            (31, 31), 3.6, theta, 10.0, 0.52, 0, ktype=cv2.CV_32F
        )
        responses.append(np.abs(cv2.filter2D(grayf, cv2.CV_32F, kernel)))
    response = np.max(np.stack(responses, axis=0), axis=0)
    response = cv2.GaussianBlur(response, (0, 0), 0.7)
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    response[mask == 0] = 0
    return response


def _ridge_core_mask(mask: np.ndarray) -> np.ndarray:
    # Never trust the immediate finger boundary as ridge data. This is a
    # major source of the white "snow" visible in poor extractions.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    core = cv2.erode(mask, k, iterations=1)
    # Remove tiny isolated mask islands as well.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    if n <= 1:
        return core
    largest = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _clean_ridges(ridges: np.ndarray, mask: np.ndarray, mode: int) -> np.ndarray:
    ridges = cv2.bitwise_and(ridges, mask)
    # Closing repairs small breaks in genuine ridge strokes without turning
    # isolated background specks into long structures.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ridges = cv2.morphologyEx(ridges, cv2.MORPH_CLOSE, close_k, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ridges, 8)
    clean = np.zeros_like(ridges)
    area_floor = 18 if mode == 2 else 12
    scale_floor = max(1, (ridges.shape[0] * ridges.shape[1]) // (110000 if mode == 2 else 150000))
    min_area = max(area_floor, scale_floor)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        # Keep normal ridge fragments, but reject tiny compact speckles.
        if area >= min_area and max(ww, hh) >= 5:
            clean[labels == i] = 255

    if mode == 1:
        clean = cv2.morphologyEx(
            clean, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        )
    clean[mask == 0] = 0
    return clean


def extract_ridges(image: np.ndarray, mode: int = 1, zoom_scale: int = 2):
    if image.ndim != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    finger_mask = _single_finger_mask(image)
    isolated = np.full_like(image, 255)
    isolated[finger_mask > 0] = image[finger_mask > 0]
    zoomed, zoomed_mask, actual_scale = auto_zoom(isolated, finger_mask, zoom_scale)

    # Work only in the interior of the selected finger. Boundary pixels are
    # where segmentation errors most often look like fake ridge fragments.
    ridge_mask = _ridge_core_mask(zoomed_mask)
    gray = cv2.cvtColor(zoomed, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=2.0 if mode == 1 else 2.3,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    if mode == 1:
        ridges = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 4
        )
    else:
        # Precision mode: Gabor is a real ridge-evidence detector, not a
        # second hard binary mask. Adaptive threshold supplies the strokes;
        # Gabor decides whether local texture looks ridge-like.
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 35, 5
        )
        gabor = _gabor_precision(enhanced, ridge_mask)
        vals = gabor[ridge_mask > 0]
        if vals.size:
            p68 = float(np.percentile(vals, 68))
            p84 = float(np.percentile(vals, 84))
        else:
            p68, p84 = 55.0, 75.0

        # Require substantially more evidence than the old p58 threshold.
        confidence = (gabor >= max(42, p68)).astype(np.uint8) * 255
        strong = (gabor >= max(58, p84)).astype(np.uint8) * 255

        # A 5x5 support region lets a real ridge survive a small local Gabor
        # dip while still rejecting isolated texture specks.
        support = cv2.dilate(
            confidence,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        supported = cv2.bitwise_and(adaptive, support)
        ridges = cv2.bitwise_or(
            supported,
            cv2.bitwise_and(strong, adaptive)
        )

        # Connect nearby pieces only when there is already ridge evidence.
        bridge = cv2.dilate(
            ridges,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        ridges = cv2.bitwise_or(ridges, cv2.bitwise_and(adaptive, bridge))

    ridges = _clean_ridges(ridges, ridge_mask, mode)
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
    return {
        "image": result,
        "isolated_finger": zoomed,
        "png_bytes": encoded.tobytes(),
        "ridge_coverage": float(np.count_nonzero(result)) / result.size * 100.0,
        "region": region,
        "zoom_scale": actual_scale,
    }
