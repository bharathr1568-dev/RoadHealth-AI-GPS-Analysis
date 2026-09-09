"""
Notification Service — SQLAlchemy ORM Compatible
Handles creating and querying in-app notifications for the municipality officer.
Supports both SQLite and PostgreSQL.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from backend.database.database import get_db_session, NotificationModel

logger = logging.getLogger("roadhealth.notifications")


def create_notification(
    complaint_id: str,
    message: str,
    detail: Optional[str] = None,
    notif_type: str = "NEW_COMPLAINT",
) -> int:
    """Insert a new notification for the officer dashboard."""
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with get_db_session() as session:
        notif = NotificationModel(
            complaint_id=complaint_id,
            type=notif_type,
            message=message,
            detail=detail,
            is_read=0,
            created_at=created_at,
        )
        session.add(notif)
        session.flush()
        notif_id = notif.id
        logger.info(f"🔔 Notification created [#{notif_id}]: {message}")
        return notif_id


def get_notifications(
    limit: int = 50,
    unread_only: bool = False,
) -> Dict[str, Any]:
    """Retrieve notifications ordered with newest first, plus unread count."""
    with get_db_session() as session:
        # Count total unread
        unread_count = session.query(NotificationModel).filter(NotificationModel.is_read == 0).count()

        # Query notifications
        query = session.query(NotificationModel)
        if unread_only:
            query = query.filter(NotificationModel.is_read == 0)

        notif_objs = query.order_by(NotificationModel.id.desc()).limit(limit).all()
        notifications = [n.to_dict() for n in notif_objs]

        return {
            "unread_count": unread_count,
            "total_returned": len(notifications),
            "notifications": notifications,
        }


def mark_notification_read(notification_id: int) -> bool:
    """Mark a specific notification as read."""
    with get_db_session() as session:
        updated = session.query(NotificationModel).filter(NotificationModel.id == notification_id).update({"is_read": 1})
        return updated > 0


def mark_all_notifications_read() -> int:
    """Mark all unread notifications as read."""
    with get_db_session() as session:
        updated = session.query(NotificationModel).filter(NotificationModel.is_read == 0).update({"is_read": 1})
        return updated
