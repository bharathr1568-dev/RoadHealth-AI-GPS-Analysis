"""
RoadHealth AI — Visualization
Detection overlays + 4×4 zone heatmap  (notebook cells 6+9 — CELL 8+13).
"""

import cv2
import numpy as np
from config import ZONE_ROWS, ZONE_COLS
from zone_analysis import get_zone_risk


# ── BGR zone colours (light, from notebook) ────────────────────
ZONE_COLORS_BGR = {
    "LOW":      (190, 235, 190),
    "MEDIUM":   (170, 220, 255),
    "HIGH":     (170, 190, 255),
    "CRITICAL": (170, 170, 255),
}


def draw_detections(img_bgr, detections):
    """Draw bounding boxes + labels on a copy of the image."""
    return draw_custom_detections(img_bgr, detections)


def draw_custom_detections(
    img_bgr,
    detections,
    show_potholes: bool = True,
    show_cracks: bool = True,
    show_labels: bool = True,
    show_severity: bool = True,
):
    """
    Draw customizable bounding boxes and class overlays on the image.
    Supports filtering by defect class (potholes/cracks) and toggling annotations.
    """
    out = img_bgr.copy()
    if detections is None:
        return out

    for d in detections:
        cls = d.get("class_name", "").lower()
        if cls == "pothole" and not show_potholes:
            continue
        if cls == "crack" and not show_cracks:
            continue

        box = d.get("box", [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, box)
        conf = float(d.get("confidence", 0.0))
        sev = d.get("severity_score")

        # Pothole = Cyan (#22D3EE in BGR: 238, 211, 34), Crack = Amber/Orange (#F59E0B in BGR: 11, 158, 245)
        color = (238, 211, 34) if cls == "pothole" else (11, 158, 245)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

        if show_labels:
            label = f"{cls.title()} {conf:.2f}"
            if show_severity and sev is not None:
                label += f" | S:{sev:.2f}"

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            # Label background pill
            cv2.rectangle(
                out,
                (x1, max(0, y1 - text_size[1] - 10)),
                (x1 + text_size[0] + 10, max(text_size[1] + 10, y1)),
                (15, 23, 42),
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1 + 5, max(text_size[1] + 4, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                color,
                2,
                cv2.LINE_AA,
            )

    return out


def draw_zone_heatmap(img_bgr, zone_damage, detections, critical_zone_id):
    """Draw the complete 4×4 zone damage map with translucent overlays."""
    H, W = img_bgr.shape[:2]
    zone_w = W / ZONE_COLS
    zone_h = H / ZONE_ROWS

    base = img_bgr.copy()
    overlay = img_bgr.copy()

    # Fill zones
    for z in range(1, ZONE_ROWS * ZONE_COLS + 1):
        row = (z - 1) // ZONE_COLS
        col = (z - 1) % ZONE_COLS
        x1 = int(col * zone_w)
        y1 = int(row * zone_h)
        x2 = int((col + 1) * zone_w)
        y2 = int((row + 1) * zone_h)
        info = zone_damage.get(z, {})
        score = float(info.get("damage_score", 0.0))
        risk = info.get("risk_level", get_zone_risk(score))
        fill = ZONE_COLORS_BGR.get(risk, ZONE_COLORS_BGR["LOW"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)

    out = cv2.addWeighted(overlay, 0.22, base, 0.78, 0)

    # Borders + labels
    for z in range(1, ZONE_ROWS * ZONE_COLS + 1):
        row = (z - 1) // ZONE_COLS
        col = (z - 1) % ZONE_COLS
        x1 = int(col * zone_w)
        y1 = int(row * zone_h)
        x2 = int((col + 1) * zone_w)
        y2 = int((row + 1) * zone_h)
        info = zone_damage.get(z, {})
        score = float(info.get("damage_score", 0.0))
        risk = info.get("risk_level", get_zone_risk(score))

        if z == critical_zone_id:
            border_color, thickness = (0, 0, 255), 4
        else:
            border_color, thickness = (220, 220, 220), 2

        cv2.rectangle(out, (x1, y1), (x2, y2), border_color, thickness)

        label = f"Z{z} | Damage {score:.2f} | {risk}"
        
        # Calculate font scale dynamically so the text fits within the zone width
        fs = 0.55
        thickness = 2
        while fs > 0.15:
            t_thick = 2 if fs >= 0.48 else 1
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, t_thick)[0]
            if t_size[0] + 14 <= zone_w:
                thickness = t_thick
                text_size = t_size
                break
            fs -= 0.02
        else:
            thickness = 1
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thickness)[0]

        # Calculate heights and offsets proportionally to font scale
        scale_ratio = fs / 0.52
        box_h = max(16, int(30 * scale_ratio))
        text_offset = max(12, int(23 * scale_ratio))

        bg_x1 = x1 + 3
        bg_y1 = y1 + 4
        bg_x2 = x1 + text_size[0] + 12
        bg_y2 = bg_y1 + box_h

        cv2.rectangle(
            out, (bg_x1, bg_y1), (bg_x2, bg_y2),
            (245, 245, 245), -1,
        )
        cv2.putText(
            out, label, (x1 + 7, bg_y1 + text_offset),
            cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 40, 40), thickness, cv2.LINE_AA,
        )

    # Damage boxes
    for d in detections:
        bx1, by1, bx2, by2 = map(int, d["box"])
        cls = d["class_name"]
        conf = float(d["confidence"])
        color = (0, 220, 255) if cls == "pothole" else (255, 220, 0)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 3)
        cv2.putText(
            out, f"{cls.title()} {conf:.2f}",
            (bx1, max(25, by1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA,
        )

    return out
