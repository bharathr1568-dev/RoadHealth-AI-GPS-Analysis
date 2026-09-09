"""
RoadHealth AI — Detection Processing
Physical pothole merging + crack bounding-box refinement  (notebook cells 5-6).
"""

import math
import cv2
import numpy as np
from utils import (
    box_area, intersection_area, box_iou,
    small_box_overlap, clamp01,
)


# ═══════════════════════════════════════════════════════════════
# 1. PHYSICAL POTHOLE MERGING  (Union-Find based)
# ═══════════════════════════════════════════════════════════════

def merge_potholes(detections, overlap_threshold=0.40, iou_threshold=0.25):
    """
    Convert multiple overlapping YOLO pothole detections into
    physical pothole candidates using Union-Find clustering.
    """
    potholes = [dict(d) for d in detections if d["class_name"] == "pothole"]
    others = [dict(d) for d in detections if d["class_name"] != "pothole"]

    if len(potholes) <= 1:
        return others + potholes

    parent = list(range(len(potholes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(potholes)):
        for j in range(i + 1, len(potholes)):
            box_i, box_j = potholes[i]["box"], potholes[j]["box"]
            ai, aj = box_area(box_i), box_area(box_j)
            if ai <= 0 or aj <= 0:
                continue
            inter = intersection_area(box_i, box_j)
            if inter <= 0:
                continue
            iou_val = box_iou(box_i, box_j)
            overlap_small = inter / min(ai, aj)
            if overlap_small >= overlap_threshold or iou_val >= iou_threshold:
                union(i, j)

    clusters = {}
    for i in range(len(potholes)):
        root = find(i)
        clusters.setdefault(root, []).append(potholes[i])

    merged = []
    for cluster in clusters.values():
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        x1 = min(d["box"][0] for d in cluster)
        y1 = min(d["box"][1] for d in cluster)
        x2 = max(d["box"][2] for d in cluster)
        y2 = max(d["box"][3] for d in cluster)
        best = max(cluster, key=lambda d: d["confidence"])
        m = dict(best)
        m["box"] = [x1, y1, x2, y2]
        m["confidence"] = max(d["confidence"] for d in cluster)
        m["merged_count"] = len(cluster)
        m["merged_from"] = [d["confidence"] for d in cluster]
        m["source"] = "merged_physical_pothole"
        merged.append(m)

    return others + merged


# ═══════════════════════════════════════════════════════════════
# 2. PHYSICAL CRACK MERGING (Union-Find + Collinear Continuity)
# ═══════════════════════════════════════════════════════════════

def are_cracks_connected(box_a, box_b):
    """
    Check if two crack detections belong to the same continuous physical crack.
    Handles:
      1. Direct overlap / nesting (IoU or IoMin).
      2. Longitudinal continuity (vertical alignment + gap/overlap).
      3. Transverse continuity (horizontal alignment + gap/overlap).
      4. General spatial proximity between crack segments.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    wa, ha = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    wb, hb = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    cxa, cya = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    cxb, cyb = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

    # 1. Direct overlap
    inter = intersection_area(box_a, box_b)
    if inter > 0:
        ai, aj = box_area(box_a), box_area(box_b)
        overlap_small = inter / min(ai, aj) if min(ai, aj) > 0 else 0
        iou_val = inter / (ai + aj - inter) if (ai + aj - inter) > 0 else 0
        if overlap_small >= 0.15 or iou_val >= 0.08:
            return True

    # Spatial gaps between boxes
    gap_x = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0.0, max(ay1, by1) - min(ay2, by2))
    max_w = max(wa, wb)
    max_h = max(ha, hb)

    # 2. Longitudinal (Vertical) Crack Alignment & Continuity
    h_alignment = abs(cxa - cxb) <= max(45.0, 1.5 * max_w) and gap_x <= max(35.0, 0.8 * max_w)
    v_continuity = gap_y <= max(60.0, 0.25 * max_h)
    if h_alignment and v_continuity:
        return True

    # 3. Transverse (Horizontal) Crack Alignment & Continuity
    v_alignment = abs(cya - cyb) <= max(45.0, 1.5 * max_h) and gap_y <= max(35.0, 0.8 * max_h)
    h_continuity = gap_x <= max(60.0, 0.25 * max_w)
    if v_alignment and h_continuity:
        return True

    # 4. Direct Euclidean distance between box borders
    box_dist = math.sqrt(gap_x ** 2 + gap_y ** 2)
    if box_dist <= 25.0:
        return True

    return False


def merge_cracks(detections):
    """
    Convert multiple fragmented/overlapping YOLO crack detections into
    continuous physical crack objects using Union-Find clustering.
    """
    cracks = [dict(d) for d in detections if d["class_name"] == "crack"]
    others = [dict(d) for d in detections if d["class_name"] != "crack"]

    if len(cracks) <= 1:
        return others + cracks

    parent = list(range(len(cracks)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(cracks)):
        for j in range(i + 1, len(cracks)):
            if are_cracks_connected(cracks[i]["box"], cracks[j]["box"]):
                union(i, j)

    clusters = {}
    for i in range(len(cracks)):
        root = find(i)
        clusters.setdefault(root, []).append(cracks[i])

    merged = []
    for cluster in clusters.values():
        if len(cluster) == 1:
            merged.append(cluster[0])
            continue
        x1 = min(d["box"][0] for d in cluster)
        y1 = min(d["box"][1] for d in cluster)
        x2 = max(d["box"][2] for d in cluster)
        y2 = max(d["box"][3] for d in cluster)
        best = max(cluster, key=lambda d: d["confidence"])
        m = dict(best)
        m["box"] = [x1, y1, x2, y2]
        m["confidence"] = max(d["confidence"] for d in cluster)
        m["merged_count"] = len(cluster)
        m["merged_from"] = [d["confidence"] for d in cluster]
        m["source"] = "merged_continuous_crack"
        merged.append(m)

    return others + merged


# ═══════════════════════════════════════════════════════════════
# 3. CRACK BOX REFINEMENT  (morphological Black-Hat)
# ═══════════════════════════════════════════════════════════════

def refine_crack_box(image_bgr, original_box, min_component_area=12):
    """
    Refine an oversized YOLO crack box using multi-scale
    morphological Black-Hat filtering inside the ROI.
    """
    x1, y1, x2, y2 = map(int, original_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image_bgr.shape[1], x2)
    y2 = min(image_bgr.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return original_box

    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return original_box

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    masks = []
    for k in [9, 15, 21]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
        threshold = np.percentile(blackhat, 90)
        _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
        masks.append(mask)

    crack_mask = np.maximum.reduce(masks)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        crack_mask, connectivity=8
    )

    roi_h, roi_w = crack_mask.shape
    candidates = []
    for label_id in range(1, n_labels):
        cx, cy, w, h, area = stats[label_id]
        if area < min_component_area:
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect < 1.5 and area < 100:
            continue
        center_x, center_y = cx + w / 2, cy + h / 2
        distance = math.sqrt((center_x - roi_w / 2) ** 2 + (center_y - roi_h / 2) ** 2)
        max_distance = math.sqrt(roi_w ** 2 + roi_h ** 2)
        center_score = 1.0 - distance / max(1.0, max_distance)
        elongation_score = clamp01((aspect - 1.0) / 8.0)
        area_score = clamp01(area / max(1.0, 0.03 * roi_w * roi_h))
        score = 0.50 * elongation_score + 0.30 * area_score + 0.20 * center_score
        candidates.append((score, cx, cy, w, h, area))

    if not candidates:
        return original_box

    candidates.sort(key=lambda c: c[0], reverse=True)
    best = candidates[:8]
    rx1 = max(0, min(c[1] for c in best))
    ry1 = max(0, min(c[2] for c in best))
    rx2 = min(roi_w, max(c[1] + c[3] for c in best))
    ry2 = min(roi_h, max(c[2] + c[4] for c in best))

    if (rx2 - rx1) * (ry2 - ry1) < 0.01 * roi_w * roi_h:
        return original_box

    return [x1 + rx1, y1 + ry1, x1 + rx2, y1 + ry2]


# ═══════════════════════════════════════════════════════════════
# 4. FULL POST-PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════

def process_detections(nms_detections, image_bgr):
    """
    Apply pothole merging → crack merging → crack refinement.
    Returns list of final detections.
    """
    # 1. Pothole physical-object merging
    merged = merge_potholes(nms_detections)

    # 2. Crack continuous physical merging
    merged = merge_cracks(merged)

    # 3. Crack box refinement
    final = []
    for d in merged:
        d2 = dict(d)
        if d2["class_name"] == "crack":
            original_box = d2["box"]
            d2["original_box"] = original_box
            d2["box"] = refine_crack_box(image_bgr, original_box)
            d2["box_refined"] = True
        else:
            d2["box_refined"] = False
        final.append(d2)

    return final
