"""
Notifications API
Endpoints for querying and marking notifications for the Municipality Officer.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends

from backend.services.notification_service import (
    get_notifications,
    mark_notification_read,
    mark_all_notifications_read,
)
from backend.api.auth import get_current_officer

logger = logging.getLogger("roadhealth.notifications_api")
router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


@router.get("")
async def list_notifications(
    unread_only: bool = Query(False, description="Filter only unread notifications"),
    limit: int = Query(50, ge=1, le=100),
):
    """Fetch notifications for the officer portal ordered newest first."""
    try:
        data = get_notifications(limit=limit, unread_only=unread_only)
        return {"success": True, **data}
    except Exception as e:
        logger.error(f"Error fetching notifications: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch notifications.")


@router.patch("/{notification_id}/read")
async def read_notification(
    notification_id: int,
    officer: dict = Depends(get_current_officer),
):
    """Mark a specific notification as read."""
    success = mark_notification_read(notification_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found.")
    return {"success": True, "message": "Notification marked as read."}


@router.post("/read-all")
async def read_all_notifications(
    officer: dict = Depends(get_current_officer),
):
    """Mark all unread notifications as read."""
    count = mark_all_notifications_read()
    return {"success": True, "marked_read_count": count}
