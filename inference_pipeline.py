"""
RoadHealth AI — Robust Inference Pipeline
Context-aware multi-scale inference with class-specific thresholding,
road-region validation, and noise/hallucination prevention.
"""

import cv2
import numpy as np
from config import (
    IMG_SIZE, TILE_SIZE, TILE_OVERLAP,
    CONF_POTHOLE, CONF_CRACK,
    CONF_TILE_POTHOLE, CONF_TILE_CRACK,
    USE_CLAHE_AUX, IOU_MERGE_THRESHOLD,
)
from utils import normalize_class_name, box_iou


# ── Image Quality Assessment ───────────────────────────────────

def check_image_quality(img_bgr: np.ndarray) -> dict:
    """
    Assess brightness, contrast, and sharpness of the input image.
    Provides diagnostic warnings without rejecting legitimate captures.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_b = float(np.mean(gray))
    std_c = float(np.std(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings = []
    if mean_b < 25.0:
        warnings.append("Low lighting: scene is very dark; defect visibility may be reduced.")
    elif mean_b > 230.0:
        warnings.append("Overexposed: scene is extremely bright.")
    if lap_var < 20.0:
        warnings.append("Low sharpness: image appears blurry.")

    return {
        "mean_brightness": round(mean_b, 2),
        "contrast_std": round(std_c, 2),
        "sharpness_lap_var": round(lap_var, 2),
        "is_low_quality": len(warnings) > 0,
        "warnings": warnings,
    }


# ── Road-Region & Defect Plausibility Validation ──────────────

def is_valid_defect_box(img_bgr: np.ndarray, box: list, class_name: str, conf: float) -> tuple:
    """
    Validate that a bounding box belongs to a plausible road defect:
    1. Rejects sub-pixel micro-noise (< 6px).
    2. Rejects non-road pitch-black silhouettes (mean < 25, low texture).
    3. Rejects bright sun glare / overexposed sky patches (mean > 220, low variance).
    4. Rejects high-luminance detections residing in the upper sky zone.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    bw = x2 - x1
    bh = y2 - y1
    if bw < 6 or bh < 6:
        return False, "subpixel_noise"

    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False, "empty_roi"

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray_roi))
    std_val = float(np.std(gray_roi))

    # 1. Dark silhouette / non-road solid black shadow (mean < 25, low texture)
    if mean_val < 25.0 and std_val < 18.0:
        return False, "dark_silhouette_no_texture"

    # 2. Pure sun glare / bright sky
    if mean_val > 220.0 and std_val < 25.0:
        return False, "bright_sun_glare"

    # 3. Sky region in top 25% of image with high brightness
    if (y2 / h) < 0.25 and mean_val > 175.0:
        return False, "upper_sky_region"

    return True, "valid"


# ── Helpers ────────────────────────────────────────────────────

def clahe_image(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def nms_class_aware(detections: list, iou_thr: float = 0.45) -> list:
    """Merge duplicate detections generated across inference passes."""
    if not detections:
        return []
    output = []
    for cls_name in ["pothole", "crack"]:
        cls_dets = [d for d in detections if d["class_name"] == cls_name]
        cls_dets.sort(key=lambda d: d["confidence"], reverse=True)
        kept = []
        for d in cls_dets:
            duplicate = False
            for k in kept:
                if box_iou(d["box"], k["box"]) >= iou_thr:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(d)
        output.extend(kept)
    return sorted(output, key=lambda d: (d["class_name"], -d["confidence"]))


# ── Single-pass runner ─────────────────────────────────────────

def _run_pass(model, img_bgr, source_name, x_offset=0, y_offset=0,
              img_w=None, img_h=None, imgsz=IMG_SIZE,
              conf_pothole=CONF_POTHOLE, conf_crack=CONF_CRACK,
              full_image_ref=None):
    """
    Run YOLO on one image/tile with coordinate remapping & road validation.
    """
    names = model.names
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    W = img_w or img_bgr.shape[1]
    H = img_h or img_bgr.shape[0]
    validation_ref = full_image_ref if full_image_ref is not None else img_bgr

    results = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=min(conf_pothole, conf_crack),
        iou=0.45,
        verbose=False,
    )

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id not in names:
                continue
            raw_name = normalize_class_name(names[cls_id])

            if "pothole" in raw_name:
                cls_name, threshold = "pothole", conf_pothole
            elif "crack" in raw_name:
                cls_name, threshold = "crack", conf_crack
            else:
                continue

            if conf < threshold:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(float)
            x1, y1, x2, y2 = xyxy
            x1 += x_offset; x2 += x_offset
            y1 += y_offset; y2 += y_offset
            x1 = max(0.0, min(float(W - 1), x1))
            x2 = max(0.0, min(float(W - 1), x2))
            y1 = max(0.0, min(float(H - 1), y1))
            y2 = max(0.0, min(float(H - 1), y2))
            if x2 <= x1 or y2 <= y1:
                continue

            full_box = [float(x1), float(y1), float(x2), float(y2)]

            # Validate against road/sky/silhouette filters
            is_valid, _ = is_valid_defect_box(validation_ref, full_box, cls_name, conf)
            if not is_valid:
                continue

            detections.append({
                "class_name": cls_name,
                "class_id": cls_id,
                "confidence": conf,
                "box": full_box,
                "source": source_name,
            })
    return detections


# ── Full pipeline ──────────────────────────────────────────────

def run_inference(model, image_bgr, progress_callback=None):
    """
    Run the improved context-aware multi-scale inference pipeline.
    Returns (all_raw, nms_detections).
    """
    H, W = image_bgr.shape[:2]
    all_detections = []

    # 1. Global Full-Image Pass (imgsz=1280 for large images, 640 for standard)
    if progress_callback:
        progress_callback("Running full-image inference...")
    primary_imgsz = 1280 if max(H, W) >= 1600 else IMG_SIZE
    all_detections.extend(
        _run_pass(
            model, image_bgr, "full_original",
            img_w=W, img_h=H, imgsz=primary_imgsz,
            conf_pothole=CONF_POTHOLE, conf_crack=CONF_CRACK,
            full_image_ref=image_bgr
        )
    )

    # 2. Context-Aware Macro-Tiling for very high-resolution images (>= 1600px)
    if max(H, W) >= 1600:
        if progress_callback:
            progress_callback("Running context-aware multi-scale pass...")
        macro_tiles = [
            (image_bgr[0:int(H * 0.6), 0:int(W * 0.6)], 0, 0),
            (image_bgr[0:int(H * 0.6), int(W * 0.4):W], int(W * 0.4), 0),
            (image_bgr[int(H * 0.4):H, 0:int(W * 0.6)], 0, int(H * 0.4)),
            (image_bgr[int(H * 0.4):H, int(W * 0.4):W], int(W * 0.4), int(H * 0.4)),
        ]
        for i, (tile, ox, oy) in enumerate(macro_tiles):
            gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            mean_t = float(np.mean(gray_tile))
            std_t = float(np.std(gray_tile))
            # Skip uniform dark silhouettes or pure sky tiles
            if mean_t < 25.0 or (mean_t > 215.0 and std_t < 20.0):
                continue

            all_detections.extend(
                _run_pass(
                    model, tile, f"macro_tile_{i}",
                    x_offset=ox, y_offset=oy,
                    img_w=W, img_h=H, imgsz=640,
                    conf_pothole=CONF_TILE_POTHOLE,
                    conf_crack=CONF_TILE_CRACK,
                    full_image_ref=image_bgr
                )
            )

    # 3. Optional CLAHE auxiliary pass (if enabled and image has low contrast)
    if USE_CLAHE_AUX:
        gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if float(np.std(gray_full)) < 35.0:
            if progress_callback:
                progress_callback("Running CLAHE auxiliary inference...")
            enhanced = clahe_image(image_bgr)
            all_detections.extend(
                _run_pass(
                    model, enhanced, "full_clahe",
                    img_w=W, img_h=H, imgsz=primary_imgsz,
                    conf_pothole=CONF_TILE_POTHOLE,
                    conf_crack=CONF_TILE_CRACK,
                    full_image_ref=image_bgr
                )
            )

    # 4. Class-aware NMS
    if progress_callback:
        progress_callback("Applying class-aware NMS...")
    nms_detections = nms_class_aware(all_detections, IOU_MERGE_THRESHOLD)

    return all_detections, nms_detections
