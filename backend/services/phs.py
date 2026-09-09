"""
Road Health Score Service
Isolated, configurable RHS/PHS computation module.

This wraps the existing road_health.compute_road_health() function and
provides additional helper methods for the web API response format.

NOTE: The PHS formula uses the project's existing research-derived
methodology from road_health.py. The condition thresholds below are
adapted for the citizen-facing web interface.

IMPORTANT: All values are sanitized to native Python types before
returning to FastAPI to avoid numpy serialization errors.
"""

import numpy as np


def _sanitize(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ── PHS condition mapping for the citizen web interface ──────
PHS_CONDITIONS = [
    {"min": 90, "label": "EXCELLENT",  "emoji": "🟢", "color": "#10b981", "description": "Road is in excellent condition with minimal or no defects."},
    {"min": 75, "label": "GOOD",       "emoji": "🔵", "color": "#3b82f6", "description": "Road is in good condition with minor surface issues."},
    {"min": 60, "label": "MODERATE",   "emoji": "🟡", "color": "#f59e0b", "description": "Road has noticeable defects requiring attention."},
    {"min": 40, "label": "POOR",       "emoji": "🟠", "color": "#f97316", "description": "Road has significant damage affecting safety and comfort."},
    {"min": 0,  "label": "CRITICAL",   "emoji": "🔴", "color": "#ef4444", "description": "Road is severely damaged and requires immediate repair."},
]


def get_phs_condition(phs: float) -> dict:
    """Map a PHS score (0-100) to a condition label with display metadata."""
    for cond in PHS_CONDITIONS:
        if phs >= cond["min"]:
            return cond
    return PHS_CONDITIONS[-1]


def format_phs_response(health_summary: dict) -> dict:
    """
    Transform the raw health_summary from road_health.compute_road_health()
    into a clean, web-API-friendly response.
    All numpy types are converted to native Python types.
    """
    # Sanitize the entire summary first
    summary = _sanitize(health_summary)

    phs = float(summary.get("pavement_health_score", 100.0))
    condition = get_phs_condition(phs)

    return {
        "phs": round(phs, 1),
        "rhs": round(phs, 1),
        "condition": condition["label"],
        "condition_emoji": condition["emoji"],
        "condition_color": condition["color"],
        "condition_description": condition["description"],
        "overall_damage": round(float(summary.get("overall_damage", 0.0)), 4),
        "pothole_damage": round(float(summary.get("pothole_damage", 0.0)), 4),
        "crack_damage": round(float(summary.get("crack_damage", 0.0)), 4),
        "maintenance_priority": str(summary.get("maintenance_priority", "ROUTINE")),
        "maintenance_recommendation": str(summary.get(
            "maintenance_recommendation",
            "No significant detected distress; continue routine monitoring."
        )),
        "pothole_count": int(summary.get("detected_potholes", 0)),
        "crack_count": int(summary.get("detected_cracks", 0)),
        "detected_potholes": int(summary.get("detected_potholes", 0)),
        "detected_cracks": int(summary.get("detected_cracks", 0)),
        "total_defects": int(summary.get("total_defects", 0)),
    }
