"""
Complaints API
Endpoints for citizen complaint submission, officer complaint management,
status updates, filtering, and timeline tracking.
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, Depends

from backend.services.complaint_service import (
    create_complaint,
    get_complaint_by_id,
    get_complaints,
    update_complaint_status,
    delete_complaint,
)
from backend.api.auth import get_current_officer, verify_token

logger = logging.getLogger("roadhealth.complaints_api")
router = APIRouter(prefix="/api/complaints", tags=["Complaints"])


# ── Request / Response Schemas ────────────────────────────────────

class ComplaintSubmitRequest(BaseModel):
    original_image: str
    annotated_image: str
    latitude: float
    longitude: float
    gps_accuracy: Optional[float] = None
    altitude: Optional[float] = None
    heading: Optional[float] = None
    speed: Optional[float] = None
    timestamp: Optional[str] = None
    phs_data: Dict[str, Any]
    detections: List[Dict[str, Any]] = []
    description: Optional[str] = ""
    citizen_phone: Optional[str] = ""


class StatusUpdateRequest(BaseModel):
    status: str
    officer_notes: Optional[str] = None
    assigned_to: Optional[str] = None


# ── 1. Submit Complaint (Citizen) ─────────────────────────────────

@router.post("", status_code=201)
async def submit_complaint(req: ComplaintSubmitRequest):
    """
    Citizen submits a road defect complaint after AI analysis has completed.
    Generates unique ID (e.g. RCA-2026-000001) and saves complete detection context.
    """
    try:
        complaint = create_complaint(
            original_image=req.original_image,
            annotated_image=req.annotated_image,
            latitude=req.latitude,
            longitude=req.longitude,
            gps_accuracy=req.gps_accuracy,
            timestamp=req.timestamp,
            phs_data=req.phs_data,
            detections=req.detections,
            description=req.description,
            citizen_phone=req.citizen_phone,
            altitude=req.altitude,
            heading=req.heading,
            speed=req.speed,
        )
        return {
            "success": True,
            "message": "Complaint submitted successfully.",
            "complaint": complaint,
        }
    except Exception as e:
        logger.error(f"Error submitting complaint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit complaint: {str(e)}")


# ── 2. Query / List Complaints (Citizen & Officer) ─────────────────

@router.get("")
async def list_complaints(
    status: Optional[str] = Query(None, description="Filter by status (NEW, UNDER REVIEW, VERIFIED, IN PROGRESS, RESOLVED, REJECTED, ALL)"),
    condition: Optional[str] = Query(None, description="Filter by condition (CRITICAL, POOR, MODERATE, GOOD, ALL)"),
    priority: Optional[str] = Query(None, description="Filter by priority (CRITICAL, HIGH, MEDIUM, LOW, ALL)"),
    search: Optional[str] = Query(None, description="Search term for complaint ID, description, or notes"),
    sort_by: str = Query("newest", description="Sort order: newest, oldest, phs_asc, phs_desc, defects_desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Retrieve list of complaints with search, filtering, and pagination."""
    try:
        data = get_complaints(
            status=status,
            condition=condition,
            priority=priority,
            search=search,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
        )
        return {"success": True, **data}
    except Exception as e:
        logger.error(f"Error listing complaints: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list complaints: {str(e)}")


# ── 3. Get Single Complaint Details ───────────────────────────────

@router.get("/{complaint_id}")
async def get_complaint(complaint_id: str):
    """Retrieve detailed complaint by ID including original & AI images, detections, and timeline."""
    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")
    return {"success": True, "complaint": complaint}


# ── 4. Update Complaint Status (Officer) ──────────────────────────

@router.patch("/{complaint_id}/status")
async def update_status(
    complaint_id: str,
    req: StatusUpdateRequest,
    officer: dict = Depends(get_current_officer),
):
    """
    Officer updates complaint status (UNDER REVIEW, VERIFIED, IN PROGRESS, RESOLVED, etc.)
    and appends notes to the audit timeline.
    """
    try:
        officer_name = officer.get("name", "Officer")
        updated = update_complaint_status(
            complaint_id=complaint_id,
            new_status=req.status,
            officer_notes=req.officer_notes,
            assigned_to=req.assigned_to,
            officer_name=officer_name,
        )
        if not updated:
            raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")

        return {
            "success": True,
            "message": f"Complaint status updated to {req.status}.",
            "complaint": updated,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating complaint status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update status: {str(e)}")


# ── 5. Delete / Remove Complaint (Officer) ────────────────────────

@router.delete("/{complaint_id}")
async def delete_complaint_endpoint(
    complaint_id: str,
    officer: dict = Depends(get_current_officer),
):
    """
    Authorized officer removes/deletes a road complaint and its linked notifications.
    """
    try:
        deleted = delete_complaint(complaint_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Complaint '{complaint_id}' not found.")
        return {
            "success": True,
            "message": f"Complaint '{complaint_id}' removed successfully.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting complaint {complaint_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete complaint: {str(e)}")

