"""
tag_ocr_core.py
================
Core pipeline: image in -> upright tag crops + decoded values out.

    1. find_tag_centers()   - classical-CV region proposals (no trained model).
                               Tuned for recall: it's fine if this also flags a
                               few non-tag regions (shadows, glare, texture) -
                               step 4 below filters those out using OCR confidence.
    2. auto_crop_and_deskew() - crops a generous window around each candidate
                               center, then deskews it using the gradient-block
                               method (Sobel-x minus Sobel-y highlights the
                               densely-striped barcode block; minAreaRect on
                               that gives the exact angle needed to make it
                               horizontal). Works at any starting rotation.
    3. read_tag_value()     - PaddleOCR reads the human-readable digit string
                               printed under the bars. use_textline_orientation
                               handles any remaining 0/180 ambiguity natively,
                               so no separate "is this upside down" step is
                               needed (PaddleOCR replaces that hand-rolled
                               tesseract check from the earlier version).
    4. process_image()      - ties it together, writes one CSV row per tag,
                               and drops candidates with low/empty OCR
                               confidence (these are almost always the
                               occasional false-positive region from step 1,
                               not real tags).

Requires: opencv-python, numpy, paddleocr, paddlepaddle
"""
import cv2
import numpy as np
import math
import re
import csv
import os

# ----------------------------------------------------------------------
# Step 1: candidate region detection (classical CV, no ML training)
# ----------------------------------------------------------------------

def find_tag_centers(img, min_area_frac=0.0008, max_area_frac=0.45,
                      min_solidity=0.45, max_aspect=3.0):
    """
    Returns a list of {"center": (x, y), "long": <long-side px>} dicts, one
    per candidate barcode-like blob. Tuned to favor recall - false positives
    are cheap because the OCR confidence check downstream filters them out.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    gradY = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(gradX, gradY))
    blurred = cv2.blur(grad, (15, 15))
    _, th = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((33, 33), np.uint8))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    img_area = h_img * w_img

    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * img_area or area > max_area_frac * img_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        aspect = max(rw, rh) / max(1, min(rw, rh))
        if aspect > max_aspect:   # excludes shadow streaks / light seams
            continue
        blobs.append({"center": (cx, cy), "long": max(rw, rh)})

    # de-duplicate overlapping/nested detections, keep the larger one
    blobs.sort(key=lambda b: -b["long"])
    kept = []
    for b in blobs:
        cx, cy = b["center"]
        if all(math.hypot(cx - k["center"][0], cy - k["center"][1]) > max(b["long"], k["long"]) * 0.5
               for k in kept):
            kept.append(b)
    return kept


# ----------------------------------------------------------------------
# Step 2: per-candidate crop + deskew (reusable for manual crops too -
# pass window_scale=1.0 and call directly on an already-tight crop)
# ----------------------------------------------------------------------

def estimate_bar_angle(gray):
    """Gradient-block method: returns the cv2 rotation angle (degrees) that
    makes the barcode's long axis horizontal, regardless of starting angle."""
    h, w = gray.shape[:2]
    scale = min(1.0, 700 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray

    gradX = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=-1)
    gradY = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(gradX, gradY))
    blurred = cv2.blur(grad, (9, 9))
    _, th = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    closed = cv2.erode(closed, np.ones((5, 5), np.uint8), iterations=3)
    closed = cv2.dilate(closed, np.ones((5, 5), np.uint8), iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.02 * small.shape[0] * small.shape[1]:
        return 0.0

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    best_len, best_vec = 0, None
    for i in range(4):
        p1, p2 = box[i], box[(i + 1) % 4]
        v = p2 - p1
        l = np.hypot(*v)
        if l > best_len:
            best_len, best_vec = l, v

    angle_visual = math.degrees(math.atan2(-best_vec[1], best_vec[0]))
    if angle_visual > 90:
        angle_visual -= 180
    if angle_visual < -90:
        angle_visual += 180
    return -angle_visual


def rotate(img_cv, angle_deg):
    h, w = img_cv.shape[:2]
    new_w, new_h = int(w * 1.4) + 40, int(h * 1.4) + 40
    canvas = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
    y0, x0 = (new_h - h) // 2, (new_w - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img_cv
    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle_deg, 1.0)
    return cv2.warpAffine(canvas, M, (new_w, new_h), borderValue=(255, 255, 255))


def autocrop_white(img_cv, pad=20):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    mask = gray < 250
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img_cv
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img_cv.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img_cv.shape[0])
    return img_cv[y0:y1, x0:x1]


def auto_crop_and_deskew(img, center, long_dim, window_scale=2.0, upscale_target=1400):
    """Crop a generous axis-aligned window around `center`, then deskew it."""
    cx, cy = center
    half = int(long_dim * window_scale / 2)
    h_img, w_img = img.shape[:2]
    x0, y0 = max(int(cx - half), 0), max(int(cy - half), 0)
    x1, y1 = min(int(cx + half), w_img), min(int(cy + half), h_img)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0.0

    long_side = max(crop.shape[:2])
    up = max(1.0, min(3.0, upscale_target / long_side))
    if up > 1.0:
        crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_LANCZOS4)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    angle = estimate_bar_angle(gray)
    rotated = rotate(crop, angle)
    rotated = autocrop_white(rotated, pad=20)
    return rotated, angle


# ----------------------------------------------------------------------
# Step 3: OCR with PaddleOCR
# ----------------------------------------------------------------------

_OCR_ENGINE = None


def get_ocr_engine():
    """Lazily construct a single shared PaddleOCR instance.
    Doc-level orientation/unwarping are disabled - we already deskew
    ourselves; use_textline_orientation handles any local 0/180 flip.

    Use ONNX runtime backend for faster CPU inference, avoiding Paddle
    framework issues with MKL-DNN / new executor on Windows.
    """
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from paddleocr import PaddleOCR
        _OCR_ENGINE = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang="en",
            engine="onnxruntime",
        )
    return _OCR_ENGINE


BINARY8_RE = re.compile(r"^[01]{8}$")
DIGITS_RE = re.compile(r"\d+")


def read_tag_value(ocr_engine, crop_bgr):
    """
    Runs PaddleOCR on the crop and returns (decoded_value, confidence, raw_text).
    Prefers a clean 8-bit 0/1 string (this dataset's tag format); otherwise
    falls back to the highest-confidence recognized text fragment.
    """
    results = ocr_engine.predict(crop_bgr)
    if not results:
        return None, 0.0, ""

    texts, scores = [], []
    for res in results:
        texts.extend(res.get("rec_texts", []))
        scores.extend(res.get("rec_scores", []))

    if not texts:
        return None, 0.0, ""

    raw_text = " | ".join(texts)

    # Prefer an exact 8-bit binary match (this tag format)
    for t, s in zip(texts, scores):
        cleaned = t.strip().replace(" ", "")
        if BINARY8_RE.match(cleaned):
            return cleaned, float(s), raw_text

    # Fallback: longest digit run, weighted by confidence
    best_val, best_score = None, 0.0
    for t, s in zip(texts, scores):
        for m in DIGITS_RE.finditer(t):
            if len(m.group()) >= len(best_val or "") and s >= best_score:
                best_val, best_score = m.group(), s

    return best_val, float(best_score), raw_text


# ----------------------------------------------------------------------
# Step 4: full pipeline for one image -> CSV rows
# ----------------------------------------------------------------------

def process_image(image_path, ocr_engine=None, save_crops_dir=None,
                   min_confidence=0.3):
    """
    Returns a list of dict rows ready to write to CSV:
        source_image, tag_index, center_x, center_y, rotation_deg,
        decoded_value, confidence, raw_ocr_text, crop_path
    Candidates with no OCR text or confidence below `min_confidence` are
    dropped - in practice these are the false-positive regions from
    find_tag_centers (shadows, glare), not real tags.
    """
    if ocr_engine is None:
        ocr_engine = get_ocr_engine()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    candidates = find_tag_centers(img)
    base = os.path.splitext(os.path.basename(image_path))[0]

    rows = []
    kept_idx = 0
    for cand in candidates:
        crop, angle = auto_crop_and_deskew(img, cand["center"], cand["long"])
        if crop is None:
            continue

        decoded, conf, raw = read_tag_value(ocr_engine, crop)

        if decoded is None or conf < min_confidence:
            continue  # likely a false-positive region, not a real tag

        crop_path = ""
        if save_crops_dir:
            os.makedirs(save_crops_dir, exist_ok=True)
            crop_path = os.path.join(save_crops_dir, f"{base}_tag{kept_idx}.png")
            cv2.imwrite(crop_path, crop)

        rows.append({
            "source_image": os.path.basename(image_path),
            "tag_index": kept_idx,
            "center_x": round(cand["center"][0], 1),
            "center_y": round(cand["center"][1], 1),
            "rotation_deg": round(angle, 1),
            "decoded_value": decoded,
            "confidence": round(conf, 3),
            "raw_ocr_text": raw,
            "crop_path": crop_path,
        })
        kept_idx += 1

    return rows


def process_images_to_csv(image_paths, out_csv, save_crops_dir=None, min_confidence=0.3):
    ocr_engine = get_ocr_engine()
    fieldnames = ["source_image", "tag_index", "center_x", "center_y",
                  "rotation_deg", "decoded_value", "confidence",
                  "raw_ocr_text", "crop_path"]

    all_rows = []
    for path in image_paths:
        rows = process_image(path, ocr_engine=ocr_engine,
                              save_crops_dir=save_crops_dir,
                              min_confidence=min_confidence)
        all_rows.extend(rows)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return all_rows
