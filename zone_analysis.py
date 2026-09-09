"""
RoadHealth AI — Zone Analysis
4×4 spatial zone damage calculation  (notebook cells 8-9 / CELL 12-13).
"""

from config import ZONE_ROWS, ZONE_COLS
from utils import clamp01, box_area, intersection_area


def get_zone_risk(score):
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.50:
        return "HIGH"
    elif score >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"


def get_zone_box(zone_id, img_w, img_h):
    zone_w = img_w / ZONE_COLS
    zone_h = img_h / ZONE_ROWS
    row = (zone_id - 1) // ZONE_COLS
    col = (zone_id - 1) % ZONE_COLS
    return [
        int(col * zone_w),
        int(row * zone_h),
        int((col + 1) * zone_w),
        int((row + 1) * zone_h),
    ]


def compute_zone_damage(features, img_w, img_h):
    """
    Build the 4×4 zone damage map.
    *features* is the list of enriched detection dicts (with severity_score).
    Returns (zone_damage dict, critical_zone_id, critical_zone_score, critical_zone_risk).
    """
    total_zones = ZONE_ROWS * ZONE_COLS
    zone_damage = {}

    # Build severity lookup from features list indices
    severity_lookup = {}
    for idx, f in enumerate(features):
        severity_lookup[idx] = clamp01(f.get("severity_score", 0.0))

    for zone_id in range(1, total_zones + 1):
        zone_box = get_zone_box(zone_id, img_w, img_h)
        z_area = box_area(zone_box)

        pothole_scores, crack_scores = [], []
        pothole_coverage, crack_coverage = [], []
        defect_count = 0

        for det_idx, d in enumerate(features):
            det_box = d["box"]
            cls = str(d["class_name"]).lower()
            confidence = clamp01(d.get("confidence", 0.0))
            overlap = intersection_area(det_box, zone_box)
            if overlap <= 0:
                continue
            defect_count += 1
            coverage = clamp01(overlap / z_area) if z_area > 0 else 0.0
            severity = severity_lookup.get(det_idx, 0.0)
            if severity <= 0:
                severity = clamp01(0.65 * coverage + 0.35 * confidence)

            if "pothole" in cls:
                pothole_scores.append(clamp01(0.70 * severity + 0.20 * coverage + 0.10 * confidence))
                pothole_coverage.append(coverage)
            elif "crack" in cls:
                crack_scores.append(clamp01(0.55 * severity + 0.30 * coverage + 0.15 * confidence))
                crack_coverage.append(coverage)

        max_pothole = max(pothole_scores) if pothole_scores else 0.0
        max_crack = max(crack_scores) if crack_scores else 0.0
        total_pot_cov = clamp01(sum(pothole_coverage))
        total_crack_cov = clamp01(sum(crack_coverage))
        defect_density = clamp01(defect_count / 5.0)
        damage_concentration = clamp01(max(total_pot_cov, total_crack_cov) * 4.0)

        base_damage = clamp01(
            0.50 * max_pothole
            + 0.25 * max_crack
            + 0.15 * damage_concentration
            + 0.10 * defect_density
        )

        # Pothole safety floor
        if max_pothole >= 0.75:
            damage_score = max(base_damage, 0.50)
        elif max_pothole >= 0.55:
            damage_score = max(base_damage, 0.25)
        else:
            damage_score = base_damage

        # Multiple-damage boost
        if max_pothole > 0 and max_crack > 0:
            damage_score += 0.08
        if defect_count >= 3:
            damage_score += 0.05
        damage_score = clamp01(damage_score)

        risk_level = get_zone_risk(damage_score)

        # Count potholes / cracks in this zone
        n_pot = sum(
            1 for d in features
            if "pothole" in str(d["class_name"]).lower()
            and intersection_area(d["box"], zone_box) > 0
        )
        n_crack = sum(
            1 for d in features
            if "crack" in str(d["class_name"]).lower()
            and intersection_area(d["box"], zone_box) > 0
        )

        zone_damage[zone_id] = {
            "potholes": n_pot,
            "cracks": n_crack,
            "max_pothole_severity": float(max_pothole),
            "max_crack_severity": float(max_crack),
            "damage_concentration": float(damage_concentration),
            "defect_density": float(defect_density),
            "damage_score": float(damage_score),
            "risk_level": risk_level,
        }

    critical_zone_id = max(zone_damage, key=lambda z: zone_damage[z]["damage_score"])
    critical_zone_score = zone_damage[critical_zone_id]["damage_score"]
    critical_zone_risk = zone_damage[critical_zone_id]["risk_level"]

    return zone_damage, critical_zone_id, critical_zone_score, critical_zone_risk
