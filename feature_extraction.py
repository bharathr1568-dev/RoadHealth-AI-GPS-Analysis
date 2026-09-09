"""
RoadHealth AI — Feature Extraction
Damage severity / morphology features  (notebook cell 7 — CELL 10).
"""

import math
import cv2
import numpy as np
from utils import clamp01


def _crop_box(img, box):
    x1, y1, x2, y2 = map(int, box)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def crack_structure_features(roi):
    """Extract image-based crack morphology proxies."""
    empty = {
        "crack_pixel_ratio": 0.0,
        "crack_length_score": 0.0,
        "crack_width_score": 0.0,
        "connectivity_score": 0.0,
        "branching_score": 0.0,
        "edge_density": 0.0,
    }
    if roi is None or roi.size == 0:
        return empty

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2. Multi-scale black-hat
    blackhat_masks = []
    for ks in [9, 15, 21]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        bh = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
        thr = np.percentile(bh, 88)
        _, mask = cv2.threshold(bh, thr, 255, cv2.THRESH_BINARY)
        blackhat_masks.append(mask)
    crack_mask = np.maximum.reduce(blackhat_masks)

    # 3. Morphological cleanup
    crack_mask = cv2.morphologyEx(
        crack_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
    )
    crack_mask = cv2.morphologyEx(
        crack_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1,
    )

    # 4. Crack pixel ratio
    crack_pixels = np.sum(crack_mask > 0)
    total_pixels = crack_mask.shape[0] * crack_mask.shape[1]
    crack_pixel_ratio = crack_pixels / max(1, total_pixels)

    # 5. Connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crack_mask, 8)
    valid = []
    for lb in range(1, n_labels):
        x, y, w, h, area = stats[lb]
        if area < 8:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        valid.append({"w": w, "h": h, "area": area, "aspect": aspect})

    # 6. Length proxy
    if valid:
        largest = max(valid, key=lambda c: c["area"])
        length_px = max(largest["w"], largest["h"])
        diag = math.sqrt(roi.shape[1] ** 2 + roi.shape[0] ** 2)
        crack_length_score = clamp01(length_px / max(1.0, 0.60 * diag))
    else:
        crack_length_score = 0.0

    # 7. Width proxy
    if valid:
        median_width = float(np.median([min(c["w"], c["h"]) for c in valid]))
        ref = max(1.0, 0.08 * min(roi.shape[0], roi.shape[1]))
        crack_width_score = clamp01(median_width / ref)
    else:
        crack_width_score = 0.0

    # 8. Connectivity
    if valid:
        largest_area = max(c["area"] for c in valid)
        total_area = sum(c["area"] for c in valid)
        connectivity_score = clamp01(largest_area / max(1.0, total_area))
    else:
        connectivity_score = 0.0

    # 9. Branching proxy
    if valid:
        branching_score = clamp01(len([c for c in valid if c["aspect"] >= 2.0]) / 5.0)
    else:
        branching_score = 0.0

    # 10. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    return {
        "crack_pixel_ratio": float(crack_pixel_ratio),
        "crack_length_score": float(crack_length_score),
        "crack_width_score": float(crack_width_score),
        "connectivity_score": float(connectivity_score),
        "branching_score": float(branching_score),
        "edge_density": float(edge_density),
    }


def extract_features(detections, image_bgr):
    """
    Compute damage features for every detection.
    Returns a new list of enriched detection dicts.
    """
    H, W = image_bgr.shape[:2]
    enriched = []

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        bbox_area_ratio = (bw * bh) / (W * H)
        bbox_area_score = clamp01(bbox_area_ratio / 0.10)
        aspect = max(bw / bh, bh / bw)
        elongation_score = clamp01((aspect - 1.0) / 8.0)
        roi = _crop_box(image_bgr, d["box"])

        if d["class_name"] == "pothole":
            edge_density = 0.0
            if roi is not None and roi.size > 0:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 60, 150)
                edge_density = float(np.mean(edges > 0))

            severity = clamp01(
                0.65 * bbox_area_score
                + 0.20 * edge_density / 0.25
                + 0.15 * d["confidence"]
            )
            crack_feats = {
                "crack_pixel_ratio": 0.0,
                "crack_length_score": 0.0,
                "crack_width_score": 0.0,
                "connectivity_score": 0.0,
                "branching_score": 0.0,
            }
        else:
            cf = crack_structure_features(roi)
            severity = clamp01(
                0.25 * cf["crack_length_score"]
                + 0.15 * cf["crack_width_score"]
                + 0.20 * cf["crack_pixel_ratio"] / 0.15
                + 0.15 * cf["connectivity_score"]
                + 0.10 * cf["branching_score"]
                + 0.10 * cf["edge_density"] / 0.25
                + 0.05 * d["confidence"]
            )
            crack_feats = cf
            edge_density = cf["edge_density"]

        d2 = dict(d)
        d2.update({
            "bbox_area_ratio": bbox_area_ratio,
            "elongation_score": elongation_score,
            "edge_density": edge_density,
            "crack_pixel_ratio": crack_feats.get("crack_pixel_ratio", 0.0),
            "crack_length_score": crack_feats.get("crack_length_score", 0.0),
            "crack_width_score": crack_feats.get("crack_width_score", 0.0),
            "connectivity_score": crack_feats.get("connectivity_score", 0.0),
            "branching_score": crack_feats.get("branching_score", 0.0),
            "severity_score": severity,
        })
        enriched.append(d2)

    return enriched
