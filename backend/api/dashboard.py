"""
Dashboard API
Endpoints for Officer Dashboard KPI statistics, defect analytics,
and map marker visualization.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends

from backend.services.complaint_service import get_dashboard_stats, get_map_complaints
from backend.api.auth import get_current_officer

logger = logging.getLogger("roadhealth.dashboard_api")
router = APIRouter(prefix="/api", tags=["Dashboard & Map"])


@router.get("/dashboard/stats")
async def dashboard_stats():
    """Retrieve summarized statistics, distributions, and critical alert roads."""
    try:
        stats = get_dashboard_stats()
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error computing dashboard stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute dashboard stats.")


@router.get("/map/complaints")
async def map_complaints():
    """Retrieve complaint geospatial data for Leaflet map markers."""
    try:
        markers = get_map_complaints()
        return {"success": True, "count": len(markers), "complaints": markers}
    except Exception as e:
        logger.error(f"Error fetching map complaints: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch map data.")
