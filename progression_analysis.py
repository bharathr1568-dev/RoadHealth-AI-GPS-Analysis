"""
RoadHealth AI — Crack→Pothole Progression Analysis
Explainable progression risk  (notebook cell 10 — CELL 15).
"""

import math
import numpy as np
from utils import clamp01, box_center, point_box_distance, get_zone_id


def compute_progression(features, img_w, img_h):
    """
    Compute crack-to-pothole progression risk for every crack.
    Returns (progression_records, highest_risk, highest_level).
    """
    potholes = [d for d in features if d["class_name"] == "pothole"]
    cracks = [d for d in features if d["class_name"] == "crack"]

    progression_records = []
    diagonal = math.sqrt(img_w ** 2 + img_h ** 2)

    for i, crack in enumerate(cracks, 1):
        c = box_center(crack["box"])

        if potholes:
            distances = [point_box_distance(c, p["box"]) for p in potholes]
            nearest_distance = min(distances)
            proximity = 1.0 - clamp01(nearest_distance / (0.30 * diagonal))
        else:
            nearest_distance = None
            proximity = 0.0

        same_zone = any(
            get_zone_id(crack["box"], img_w, img_h)
            == get_zone_id(p["box"], img_w, img_h)
            for p in potholes
        )
        same_zone_score = 1.0 if same_zone else 0.0

        crack_severity = crack.get("severity_score", 0.0)
        morphology = clamp01(crack.get("edge_density", 0.0) / 0.25)

        risk = clamp01(
            0.45 * crack_severity
            + 0.25 * morphology
            + 0.20 * proximity
            + 0.10 * same_zone_score
        )

        if risk >= 0.70:
            level = "HIGH"
        elif risk >= 0.45:
            level = "MEDIUM"
        else:
            level = "LOW"

        progression_records.append({
            "crack_id": i,
            "progression_risk_score": risk,
            "risk_level": level,
            "nearest_pothole_distance_px": nearest_distance,
            "same_zone_as_pothole": same_zone,
            "explanation": {
                "crack_severity": crack_severity,
                "crack_morphology_proxy": morphology,
                "pothole_proximity": proximity,
                "same_zone": same_zone_score,
            },
        })

    if progression_records:
        highest_risk = max(r["progression_risk_score"] for r in progression_records)
        highest_level = max(
            progression_records, key=lambda r: r["progression_risk_score"]
        )["risk_level"]
    else:
        highest_risk = None
        highest_level = "NOT ESTIMATED"

    return progression_records, highest_risk, highest_level
