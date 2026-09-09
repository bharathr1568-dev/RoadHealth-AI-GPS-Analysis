"""
Seed Data Module
Initializes default officer user and sample complaints using SQLAlchemy.
Supports both SQLite and PostgreSQL.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from backend.config import (
    DEFAULT_OFFICER_EMAIL,
    DEFAULT_OFFICER_PASSWORD,
    DEFAULT_OFFICER_NAME,
    PROJECT_ROOT,
    UPLOADS_DIR,
    RESULTS_DIR,
)
from backend.database.database import (
    get_db_session,
    OfficerModel,
    ComplaintModel,
    NotificationModel,
)
from backend.services.storage import get_storage_provider

logger = logging.getLogger("roadhealth.seed")


def hash_password(password: str) -> str:
    """SHA-256 password hashing for officer authentication."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def seed_default_officer():
    """Ensure at least one officer account exists in the database if credentials are provided."""
    import os
    email = os.environ.get("OFFICER_EMAIL", DEFAULT_OFFICER_EMAIL).strip().lower()
    password = os.environ.get("OFFICER_PASSWORD", DEFAULT_OFFICER_PASSWORD).strip()
    name = os.environ.get("OFFICER_NAME", DEFAULT_OFFICER_NAME).strip()

    if not email or not password:
        logger.info("ℹ️ OFFICER_EMAIL or OFFICER_PASSWORD not set in environment. Skipping officer account auto-seeding.")
        return

    with get_db_session() as session:
        existing = session.query(OfficerModel).filter_by(email=email).first()
        if not existing:
            officer = OfficerModel(
                email=email,
                password_hash=hash_password(password),
                name=name,
                role="Municipal Road Inspector",
                department="Highway & Road Infrastructure Dept.",
                created_at=datetime.utcnow().isoformat() + "Z",
            )
            session.add(officer)
            logger.info(f"✅ Default officer seeded: {email}")
        else:
            existing.password_hash = hash_password(password)
            existing.name = name
            logger.info(f"✅ Default officer updated: {email}")


def seed_demo_complaints():
    """Seed initial sample complaints if they do not exist."""
    with get_db_session() as session:
        logger.info("Verifying demo complaints for municipality dashboard...")

        # Prepare demo images
        sample_sources = list((PROJECT_ROOT / "data").glob("**/*.jpg")) + \
                         list((PROJECT_ROOT / "data").glob("**/*.png")) + \
                         list((PROJECT_ROOT / "backend" / "uploads").glob("*.jpg"))

        storage = get_storage_provider()
        sample_img_1 = "/uploads/sample_pothole_demo.jpg"
        sample_img_2 = "/uploads/sample_crack_demo.jpg"
        sample_img_3 = "/uploads/sample_mixed_demo.jpg"
        annot_img_1 = "/results/annotated_sample_pothole_demo.jpg"
        annot_img_2 = "/results/annotated_sample_crack_demo.jpg"
        annot_img_3 = "/results/annotated_sample_mixed_demo.jpg"

        if sample_sources:
            try:
                with open(sample_sources[0], "rb") as f:
                    b1 = f.read()
                sample_img_1 = storage.save_bytes(b1, "sample_pothole_demo.jpg", "uploads")
                annot_img_1 = storage.save_bytes(b1, "annotated_sample_pothole_demo.jpg", "results")
            except Exception:
                pass

        now = datetime.utcnow()

        demo_data = [
            {
                "complaint_id": "RCA-2026-000001",
                "created_at": (now - timedelta(hours=3, minutes=15)).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "status": "NEW",
                "original_image": sample_img_1,
                "annotated_image": annot_img_1,
                "latitude": 13.0827,
                "longitude": 80.2707,
                "gps_accuracy": 4.5,
                "altitude": 14.2,
                "heading": 90.0,
                "speed": 0.0,
                "timestamp": (now - timedelta(hours=3, minutes=15)).isoformat(),
                "pothole_count": 5,
                "crack_count": 2,
                "total_defects": 7,
                "phs": 34.5,
                "condition": "CRITICAL",
                "priority": "CRITICAL",
                "severity_avg": 0.78,
                "description": "Deep multiple potholes near Anna Salai junction causing severe traffic slowdown.",
                "citizen_phone": "+91 98401 23456",
                "detections": [
                    {"class_name": "Pothole", "confidence": 0.94, "severity_score": 0.85, "box": [120, 150, 240, 290]},
                    {"class_name": "Pothole", "confidence": 0.91, "severity_score": 0.79, "box": [310, 200, 420, 310]},
                    {"class_name": "Pothole", "confidence": 0.87, "severity_score": 0.72, "box": [450, 220, 560, 330]},
                    {"class_name": "Crack", "confidence": 0.82, "severity_score": 0.65, "box": [80, 340, 520, 410]}
                ],
                "timeline": [
                    {"timestamp": (now - timedelta(hours=3, minutes=15)).strftime("%d %b %H:%M"), "event": "Complaint Submitted", "actor": "Citizen", "notes": "Submitted via RoadCare Web App"},
                    {"timestamp": (now - timedelta(hours=3, minutes=14)).strftime("%d %b %H:%M"), "event": "AI Analysis Completed", "actor": "YOLO11m-P2P3-CBAM", "notes": "PHS: 34.5 (CRITICAL), 7 defects detected"}
                ],
                "notif_msg": "New critical road complaint RCA-2026-000001: 5 Potholes, 2 Cracks (PHS: 34.5)",
            },
            {
                "complaint_id": "RCA-2026-000002",
                "created_at": (now - timedelta(hours=1, minutes=40)).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "status": "UNDER REVIEW",
                "original_image": sample_img_2,
                "annotated_image": annot_img_2,
                "latitude": 13.0604,
                "longitude": 80.2496,
                "gps_accuracy": 3.8,
                "altitude": 11.5,
                "heading": 180.0,
                "speed": 0.0,
                "timestamp": (now - timedelta(hours=1, minutes=40)).isoformat(),
                "pothole_count": 0,
                "crack_count": 4,
                "total_defects": 4,
                "phs": 58.0,
                "condition": "POOR",
                "priority": "HIGH",
                "severity_avg": 0.52,
                "description": "Long transverse and alligator cracking along the left lane.",
                "citizen_phone": "+91 97100 87654",
                "detections": [
                    {"class_name": "Crack", "confidence": 0.89, "severity_score": 0.61, "box": [50, 100, 580, 220]},
                    {"class_name": "Crack", "confidence": 0.84, "severity_score": 0.55, "box": [110, 240, 500, 320]}
                ],
                "timeline": [
                    {"timestamp": (now - timedelta(hours=1, minutes=40)).strftime("%d %b %H:%M"), "event": "Complaint Submitted", "actor": "Citizen", "notes": "Submitted via RoadCare Web App"},
                    {"timestamp": (now - timedelta(hours=1, minutes=39)).strftime("%d %b %H:%M"), "event": "AI Analysis Completed", "actor": "YOLO11m-P2P3-CBAM", "notes": "PHS: 58.0 (POOR), 4 cracks detected"},
                    {"timestamp": (now - timedelta(minutes=45)).strftime("%d %b %H:%M"), "event": "Status Updated to UNDER REVIEW", "actor": "Officer", "notes": "Inspection scheduled for Ward 117"}
                ],
                "notif_msg": "New road complaint RCA-2026-000002: 4 Cracks (PHS: 58.0)",
            },
            {
                "complaint_id": "RCA-2026-000003",
                "created_at": (now - timedelta(hours=6, minutes=20)).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "status": "RESOLVED",
                "original_image": sample_img_3,
                "annotated_image": annot_img_3,
                "latitude": 13.0382,
                "longitude": 80.2184,
                "gps_accuracy": 5.0,
                "altitude": 9.8,
                "heading": 270.0,
                "speed": 0.0,
                "timestamp": (now - timedelta(hours=6, minutes=20)).isoformat(),
                "pothole_count": 1,
                "crack_count": 1,
                "total_defects": 2,
                "phs": 76.5,
                "condition": "GOOD",
                "priority": "MEDIUM",
                "severity_avg": 0.35,
                "description": "Minor surface crack and edge pothole on secondary street.",
                "citizen_phone": "+91 94440 11223",
                "detections": [
                    {"class_name": "Crack", "confidence": 0.78, "severity_score": 0.35, "box": [200, 180, 450, 260]}
                ],
                "timeline": [
                    {"timestamp": (now - timedelta(hours=6, minutes=20)).strftime("%d %b %H:%M"), "event": "Complaint Submitted", "actor": "Citizen", "notes": "Reported"},
                    {"timestamp": (now - timedelta(hours=6, minutes=19)).strftime("%d %b %H:%M"), "event": "AI Analysis Completed", "actor": "YOLO11m-P2P3-CBAM", "notes": "PHS: 76.5 (GOOD)"},
                    {"timestamp": (now - timedelta(hours=2)).strftime("%d %b %H:%M"), "event": "Status Updated to RESOLVED", "actor": "Officer", "notes": "Bitumen patch completed by Quick Repair Team"}
                ],
                "notif_msg": "Road complaint RCA-2026-000003: 1 Pothole, 1 Crack (PHS: 76.5)",
            }
        ]

        for item in demo_data:
            existing_c = session.query(ComplaintModel).filter_by(complaint_id=item["complaint_id"]).first()
            if not existing_c:
                c = ComplaintModel(
                    complaint_id=item["complaint_id"],
                    created_at=item["created_at"],
                    status=item["status"],
                    original_image=item["original_image"],
                    annotated_image=item["annotated_image"],
                    latitude=item["latitude"],
                    longitude=item["longitude"],
                    gps_accuracy=item["gps_accuracy"],
                    altitude=item["altitude"],
                    heading=item["heading"],
                    speed=item["speed"],
                    timestamp=item["timestamp"],
                    pothole_count=item["pothole_count"],
                    crack_count=item["crack_count"],
                    total_defects=item["total_defects"],
                    phs=item["phs"],
                    condition=item["condition"],
                    priority=item["priority"],
                    severity_avg=item["severity_avg"],
                    description=item["description"],
                    citizen_phone=item["citizen_phone"],
                    detections_json=json.dumps(item["detections"]),
                    timeline_json=json.dumps(item["timeline"]),
                )
                session.add(c)

                n = NotificationModel(
                    complaint_id=item["complaint_id"],
                    type="NEW_COMPLAINT",
                    message=item["notif_msg"],
                    detail=f"Lat: {item['latitude']}, Lon: {item['longitude']}",
                    is_read=0 if item["status"] == "NEW" else 1,
                    created_at=item["created_at"],
                )
                session.add(n)

        logger.info("✅ Demo complaints verified/seeded.")
