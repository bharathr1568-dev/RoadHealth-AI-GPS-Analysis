"""
RoadHealth AI — Road Health Assessment
PHS, condition classification, maintenance decision  (notebook cell 11 — CELL 17).
"""

import numpy as np
from config import MAINTENANCE_HIGH, MAINTENANCE_MEDIUM
from utils import clamp01


def compute_road_health(features, zone_damage, progression_records):
    """
    Compute Pavement Health Score + maintenance decision.
    Returns report_summary dict.
    """
    potholes = [d for d in features if d["class_name"] == "pothole"]
    cracks = [d for d in features if d["class_name"] == "crack"]

    num_potholes = len(potholes)
    num_cracks = len(cracks)

    pothole_damage = float(np.mean([d["severity_score"] for d in potholes])) if potholes else 0.0
    crack_damage = float(np.mean([d["severity_score"] for d in cracks])) if cracks else 0.0

    if zone_damage:
        max_zone_damage = max([z["damage_score"] for z in zone_damage.values()], default=0.0)
        critical_zone_id = max(zone_damage, key=lambda z: zone_damage[z]["damage_score"])
        critical_zone_score = zone_damage[critical_zone_id]["damage_score"]
    else:
        max_zone_damage = max([d.get("severity_score", 0.0) for d in features], default=0.0)
        critical_zone_id = 1
        critical_zone_score = max_zone_damage

    overall_damage = clamp01(
        0.55 * pothole_damage
        + 0.20 * crack_damage
        + 0.25 * max_zone_damage
    )

    PHS = 100.0 * (1.0 - overall_damage)

    if PHS >= 85:
        road_condition = "EXCELLENT"
    elif PHS >= 70:
        road_condition = "GOOD"
    elif PHS >= 50:
        road_condition = "FAIR"
    elif PHS >= 30:
        road_condition = "POOR"
    else:
        road_condition = "CRITICAL"

    if critical_zone_score >= MAINTENANCE_HIGH or pothole_damage >= 0.75:
        maintenance_priority = "HIGH"
        recommendation = "Prioritize field inspection and localized repair of the most severe damaged zone."
    elif critical_zone_score >= MAINTENANCE_MEDIUM or pothole_damage >= 0.45 or crack_damage >= 0.60:
        maintenance_priority = "MEDIUM"
        recommendation = "Schedule targeted maintenance and monitor deterioration-prone damage."
    elif num_potholes or num_cracks:
        maintenance_priority = "LOW"
        recommendation = "Continue routine monitoring and plan preventive maintenance as appropriate."
    else:
        maintenance_priority = "ROUTINE"
        recommendation = "No significant detected distress; continue routine monitoring."

    if progression_records:
        highest_progression = max(r["progression_risk_score"] for r in progression_records)
        highest_progression_level = max(
            progression_records, key=lambda r: r["progression_risk_score"]
        )["risk_level"]
    else:
        highest_progression = None
        highest_progression_level = "NOT ESTIMATED"

    return {
        "detected_potholes": num_potholes,
        "detected_cracks": num_cracks,
        "total_defects": num_potholes + num_cracks,
        "pothole_damage": round(pothole_damage, 4),
        "crack_damage": round(crack_damage, 4),
        "overall_damage": round(overall_damage, 4),
        "pavement_health_score": round(PHS, 2),
        "road_health_score": round(PHS, 2),
        "road_condition": road_condition,
        "critical_zone": f"Z{critical_zone_id}",
        "critical_zone_id": critical_zone_id,
        "critical_zone_damage_score": round(critical_zone_score, 3),
        "maintenance_priority": maintenance_priority,
        "maintenance_recommendation": recommendation,
        "highest_crack_progression_risk": (
            None if highest_progression is None else round(highest_progression, 3)
        ),
        "crack_progression_level": highest_progression_level,
    }
