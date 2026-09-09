"""
Complaint Service — SQLAlchemy ORM Compatible
Business logic for complaint creation, ID generation, timeline auditing,
filtering, status management, and dashboard aggregation.
Supports both SQLite and PostgreSQL.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import func, or_

from backend.database.database import get_db_session, ComplaintModel, NotificationModel
from backend.services.notification_service import create_notification

logger = logging.getLogger("roadhealth.complaints")


def compute_ai_priority(phs: float, pothole_count: int, crack_count: int, total_defects: int) -> str:
    """
    Calculate AI-assisted priority indication.
    Note: As specified in requirements, this is clearly labeled
    'AI-assisted priority indication' to assist officers in triage.
    """
    if phs < 40.0 or pothole_count >= 4:
        return "CRITICAL"
    elif phs < 60.0 or pothole_count >= 2 or crack_count >= 4 or total_defects >= 5:
        return "HIGH"
    elif phs < 75.0 or total_defects >= 2:
        return "MEDIUM"
    else:
        return "LOW"


def generate_next_complaint_id() -> str:
    """Generate sequential complaint ID in format RCA-2026-000001."""
    year = datetime.utcnow().strftime("%Y")
    prefix = f"RCA-{year}-"

    with get_db_session() as session:
        last_complaint = (
            session.query(ComplaintModel)
            .filter(ComplaintModel.complaint_id.like(f"{prefix}%"))
            .order_by(ComplaintModel.id.desc())
            .first()
        )

        if last_complaint and last_complaint.complaint_id:
            try:
                last_id = last_complaint.complaint_id
                num_part = int(last_id.replace(prefix, ""))
                next_num = num_part + 1
            except Exception:
                next_num = 1
        else:
            next_num = 1

        return f"{prefix}{next_num:06d}"


def create_complaint(
    original_image: str,
    annotated_image: str,
    latitude: float,
    longitude: float,
    gps_accuracy: Optional[float],
    timestamp: Optional[str],
    phs_data: Dict[str, Any],
    detections: List[Dict[str, Any]],
    description: Optional[str] = None,
    citizen_phone: Optional[str] = None,
    altitude: Optional[float] = None,
    heading: Optional[float] = None,
    speed: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Persist a newly submitted citizen complaint into database,
    build timeline, and dispatch officer notification.
    """
    complaint_id = generate_next_complaint_id()
    now_utc = datetime.utcnow()
    created_at_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    display_time = now_utc.strftime("%d %b %Y, %I:%M %p")

    # Resolve defect counts from phs_data or direct detections list
    pothole_count = phs_data.get("pothole_count")
    if pothole_count is None:
        pothole_count = phs_data.get("detected_potholes")
    if pothole_count is None or (pothole_count == 0 and detections):
        pothole_count = sum(1 for d in detections if "pothole" in str(d.get("class_name", "")).lower())
    pothole_count = int(pothole_count or 0)

    crack_count = phs_data.get("crack_count")
    if crack_count is None:
        crack_count = phs_data.get("detected_cracks")
    if crack_count is None or (crack_count == 0 and detections):
        crack_count = sum(1 for d in detections if "crack" in str(d.get("class_name", "")).lower())
    crack_count = int(crack_count or 0)

    total_defects = int(phs_data.get("total_defects") or (pothole_count + crack_count) or len(detections))
    phs = float(phs_data.get("phs", 100.0))
    condition = str(phs_data.get("condition", "GOOD")).upper()

    priority = compute_ai_priority(phs, pothole_count, crack_count, total_defects)

    # Initial timeline events
    timeline = [
        {
            "timestamp": display_time,
            "event": "Complaint Submitted",
            "actor": "Citizen",
            "notes": "Citizen submitted road condition report with GPS capture",
        },
        {
            "timestamp": display_time,
            "event": "AI Defect Analysis Completed",
            "actor": "YOLO11m-P2P3-CBAM",
            "notes": f"Detected {total_defects} defects ({pothole_count} potholes, {crack_count} cracks). RHS: {phs:.1f} ({condition})",
        }
    ]

    severity_scores = [d.get("severity_score", 0.0) for d in detections if "severity_score" in d]
    severity_avg = round(sum(severity_scores) / max(len(severity_scores), 1), 4)

    with get_db_session() as session:
        complaint = ComplaintModel(
            complaint_id=complaint_id,
            created_at=created_at_str,
            status="NEW",
            original_image=original_image,
            annotated_image=annotated_image,
            latitude=latitude,
            longitude=longitude,
            gps_accuracy=gps_accuracy,
            altitude=altitude,
            heading=heading,
            speed=speed,
            timestamp=timestamp or now_utc.isoformat(),
            pothole_count=pothole_count,
            crack_count=crack_count,
            total_defects=total_defects,
            phs=phs,
            condition=condition,
            priority=priority,
            severity_avg=severity_avg,
            description=description or "",
            citizen_phone=citizen_phone or "",
            detections_json=json.dumps(detections),
            timeline_json=json.dumps(timeline),
        )
        session.add(complaint)

    # Send Notification to Officer
    issue_desc = []
    if pothole_count > 0:
        issue_desc.append(f"{pothole_count} Pothole{'s' if pothole_count > 1 else ''}")
    if crack_count > 0:
        issue_desc.append(f"{crack_count} Crack{'s' if crack_count > 1 else ''}")
    issue_str = ", ".join(issue_desc) if issue_desc else "Road defect"

    notif_msg = f"New {priority.lower()} complaint {complaint_id}: {issue_str} (RHS: {phs:.1f})"
    create_notification(
        complaint_id=complaint_id,
        message=notif_msg,
        detail=f"Lat: {latitude:.4f}, Lon: {longitude:.4f} | Condition: {condition}",
        notif_type="NEW_COMPLAINT",
    )

    logger.info(f"✅ Complaint {complaint_id} saved to database.")
    return get_complaint_by_id(complaint_id)


def get_complaint_by_id(complaint_id: str) -> Optional[Dict[str, Any]]:
    """Fetch single complaint by ID with parsed JSON objects."""
    with get_db_session() as session:
        complaint = (
            session.query(ComplaintModel)
            .filter(ComplaintModel.complaint_id == complaint_id)
            .first()
        )
        if not complaint:
            return None
        return complaint.to_dict()


def get_complaints(
    status: Optional[str] = None,
    condition: Optional[str] = None,
    priority: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "newest",  # newest, oldest, phs_asc, phs_desc, defects_desc
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """Query complaints list with dynamic filtering, search, and sorting."""
    with get_db_session() as session:
        query = session.query(ComplaintModel)

        if status and status.upper() != "ALL":
            query = query.filter(ComplaintModel.status == status.upper())

        if condition and condition.upper() != "ALL":
            query = query.filter(ComplaintModel.condition == condition.upper())

        if priority and priority.upper() != "ALL":
            query = query.filter(ComplaintModel.priority == priority.upper())

        if search and search.strip():
            term = f"%{search.strip()}%"
            query = query.filter(
                or_(
                    ComplaintModel.complaint_id.ilike(term),
                    ComplaintModel.description.ilike(term),
                    ComplaintModel.citizen_phone.ilike(term),
                    ComplaintModel.assigned_to.ilike(term),
                    ComplaintModel.detections_json.ilike(term),
                )
            )

        # Sorting
        if sort_by == "oldest":
            query = query.order_by(ComplaintModel.id.asc())
        elif sort_by == "phs_asc":
            query = query.order_by(ComplaintModel.phs.asc(), ComplaintModel.id.desc())
        elif sort_by == "phs_desc":
            query = query.order_by(ComplaintModel.phs.desc(), ComplaintModel.id.desc())
        elif sort_by == "defects_desc":
            query = query.order_by(ComplaintModel.total_defects.desc(), ComplaintModel.id.desc())
        else:  # default newest
            query = query.order_by(ComplaintModel.id.desc())

        total_count = query.count()
        rows = query.limit(limit).offset(offset).all()
        results = [r.to_dict() for r in rows]

        return {
            "total": total_count,
            "count": len(results),
            "complaints": results,
        }


def update_complaint_status(
    complaint_id: str,
    new_status: str,
    officer_notes: Optional[str] = None,
    assigned_to: Optional[str] = None,
    officer_name: str = "Officer",
) -> Optional[Dict[str, Any]]:
    """
    Update complaint status, assign officer/crew, record notes,
    and append an entry to the audit timeline.
    """
    valid_statuses = {"NEW", "UNDER REVIEW", "VERIFIED", "ASSIGNED", "IN PROGRESS", "RESOLVED", "REJECTED"}
    new_status = new_status.upper().strip()
    if new_status not in valid_statuses:
        raise ValueError(f"Invalid status '{new_status}'. Allowed: {', '.join(sorted(valid_statuses))}")

    with get_db_session() as session:
        complaint = (
            session.query(ComplaintModel)
            .filter(ComplaintModel.complaint_id == complaint_id)
            .first()
        )
        if not complaint:
            return None

        try:
            timeline = json.loads(complaint.timeline_json or "[]")
        except Exception:
            timeline = []

        now_str = datetime.utcnow().strftime("%d %b %Y, %I:%M %p")
        note_text = officer_notes or f"Status changed to {new_status}"
        if assigned_to:
            note_text += f" | Assigned to: {assigned_to}"

        timeline.append({
            "timestamp": now_str,
            "event": f"Status Updated to {new_status}",
            "actor": officer_name,
            "notes": note_text,
        })

        complaint.status = new_status
        if officer_notes is not None:
            complaint.officer_notes = officer_notes
        if assigned_to is not None:
            complaint.assigned_to = assigned_to
        complaint.timeline_json = json.dumps(timeline)

    # Trigger status change notification
    create_notification(
        complaint_id=complaint_id,
        message=f"{complaint_id} status updated to {new_status}",
        detail=note_text,
        notif_type="STATUS_UPDATE",
    )

    return get_complaint_by_id(complaint_id)


def get_dashboard_stats() -> Dict[str, Any]:
    """Aggregate high-level KPIs and distributions for the Officer Dashboard."""
    with get_db_session() as session:
        total_complaints = session.query(ComplaintModel).count()
        new_complaints = session.query(ComplaintModel).filter(ComplaintModel.status == "NEW").count()
        under_review = session.query(ComplaintModel).filter(ComplaintModel.status == "UNDER REVIEW").count()
        verified = session.query(ComplaintModel).filter(ComplaintModel.status == "VERIFIED").count()
        in_progress = session.query(ComplaintModel).filter(ComplaintModel.status == "IN PROGRESS").count()
        resolved = session.query(ComplaintModel).filter(ComplaintModel.status == "RESOLVED").count()

        critical_count = session.query(ComplaintModel).filter(ComplaintModel.condition == "CRITICAL").count()
        poor_count = session.query(ComplaintModel).filter(ComplaintModel.condition == "POOR").count()
        moderate_count = session.query(ComplaintModel).filter(ComplaintModel.condition == "MODERATE").count()
        good_count = session.query(ComplaintModel).filter(ComplaintModel.condition.in_(["GOOD", "EXCELLENT"])).count()

        sums = session.query(
            func.sum(ComplaintModel.pothole_count),
            func.sum(ComplaintModel.crack_count),
            func.sum(ComplaintModel.total_defects),
        ).first()

        sum_potholes = int(sums[0] or 0) if sums and sums[0] is not None else 0
        sum_cracks = int(sums[1] or 0) if sums and sums[1] is not None else 0
        sum_defects = int(sums[2] or 0) if sums and sums[2] is not None else 0

        avg_phs_row = session.query(func.avg(ComplaintModel.phs)).scalar()
        avg_phs = round(float(avg_phs_row), 1) if avg_phs_row is not None else 100.0

        critical_query = (
            session.query(ComplaintModel)
            .filter(
                ComplaintModel.status.in_(["NEW", "UNDER REVIEW"]),
                or_(ComplaintModel.priority == "CRITICAL", ComplaintModel.condition == "CRITICAL")
            )
            .order_by(ComplaintModel.phs.asc(), ComplaintModel.id.desc())
            .limit(5)
            .all()
        )

        critical_alerts = [
            {
                "complaint_id": c.complaint_id,
                "condition": c.condition,
                "priority": c.priority,
                "phs": c.phs,
                "pothole_count": c.pothole_count,
                "crack_count": c.crack_count,
                "total_defects": c.total_defects,
                "latitude": c.latitude,
                "longitude": c.longitude,
                "created_at": c.created_at,
                "status": c.status,
            }
            for c in critical_query
        ]

        return {
            "total_complaints": total_complaints,
            "new_complaints": new_complaints,
            "under_review": under_review,
            "verified": verified,
            "in_progress": in_progress,
            "resolved": resolved,
            "critical_roads": critical_count,
            "condition_distribution": {
                "critical": critical_count,
                "poor": poor_count,
                "moderate": moderate_count,
                "good": good_count,
            },
            "defects_summary": {
                "potholes": sum_potholes,
                "cracks": sum_cracks,
                "total": sum_defects,
            },
            "average_phs": avg_phs,
            "critical_alerts": critical_alerts,
        }


def delete_complaint(complaint_id: str) -> bool:
    """
    Delete a complaint by ID from database and delete its associated notifications.
    Returns True if successfully deleted, False if not found.
    """
    with get_db_session() as session:
        complaint = (
            session.query(ComplaintModel)
            .filter(ComplaintModel.complaint_id == complaint_id)
            .first()
        )
        if not complaint:
            return False

        # Delete associated notifications
        session.query(NotificationModel).filter(
            NotificationModel.complaint_id == complaint_id
        ).delete()

        # Delete the complaint record
        session.delete(complaint)
        logger.info(f"🗑️ Complaint {complaint_id} and associated notifications deleted.")
        return True


def get_map_complaints() -> List[Dict[str, Any]]:
    """Return lightweight geo-data for Leaflet markers with accurate defect counts."""
    with get_db_session() as session:
        complaints = session.query(ComplaintModel).order_by(ComplaintModel.id.desc()).all()
        results = []
        for c in complaints:
            d = c.to_dict()
            results.append({
                "complaint_id": d["complaint_id"],
                "status": d["status"],
                "condition": d["condition"],
                "priority": d["priority"],
                "phs": d["phs"],
                "pothole_count": d["pothole_count"],
                "crack_count": d["crack_count"],
                "total_defects": d["total_defects"],
                "latitude": d["latitude"],
                "longitude": d["longitude"],
                "gps_accuracy": d["gps_accuracy"],
                "created_at": d["created_at"],
                "annotated_image": d["annotated_image"],
                "original_image": d["original_image"],
            })
        return results
