"""
RoadHealth AI — Shared utilities
Box-geometry helpers used across multiple modules.
"""

import math
import numpy as np


def clamp01(x: float) -> float:
    """Clamp value to [0, 1]."""
    return float(np.clip(float(x), 0.0, 1.0))


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def box_iou(a, b):
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def small_box_overlap(box_a, box_b):
    """Intersection / area-of-smaller-box (for nested detection merging)."""
    inter = intersection_area(box_a, box_b)
    smaller = min(box_area(box_a), box_area(box_b))
    return inter / smaller if smaller > 0 else 0.0


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_box_distance(p, box):
    px, py = p
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.sqrt(dx * dx + dy * dy)


def normalize_class_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def get_zone_id(box, img_w, img_h, zone_rows=4, zone_cols=4):
    """Return 1-based zone id for the centre of *box*."""
    cx, cy = box_center(box)
    col = int(cx / (img_w / zone_cols))
    row = int(cy / (img_h / zone_rows))
    col = min(col, zone_cols - 1)
    row = min(row, zone_rows - 1)
    return row * zone_cols + col + 1
