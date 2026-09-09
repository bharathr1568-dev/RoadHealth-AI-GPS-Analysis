"""
Database Module — Unified SQLite & PostgreSQL Support
Provides database engine management, connection pooling, and schema initialization
for both local development (SQLite) and Render/Cloud deployments (PostgreSQL).
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Text,
    Boolean,
    Index,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.config import DATABASE_URL, IS_POSTGRES, DB_TYPE

logger = logging.getLogger("roadhealth.database")

# ── SQLAlchemy Engine & Session Factory ──────────────────────────────

if IS_POSTGRES:
    # PostgreSQL configuration with connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )
else:
    # SQLite configuration for local development
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── SQLAlchemy Models ────────────────────────────────────────────────

class ComplaintModel(Base):
    """Stores citizen road defect reports and AI detections."""
    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    complaint_id = Column(String(64), unique=True, nullable=False, index=True)
    created_at = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="NEW", index=True)
    original_image = Column(Text, nullable=False)
    annotated_image = Column(Text, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    gps_accuracy = Column(Float, nullable=True)
    altitude = Column(Float, nullable=True)
    heading = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    timestamp = Column(String(64), nullable=True)
    pothole_count = Column(Integer, default=0)
    crack_count = Column(Integer, default=0)
    total_defects = Column(Integer, default=0)
    phs = Column(Float, default=100.0)
    condition = Column(String(32), default="GOOD", index=True)
    priority = Column(String(32), default="NORMAL", index=True)
    severity_avg = Column(Float, default=0.0)
    description = Column(Text, nullable=True)
    citizen_phone = Column(String(64), nullable=True)
    assigned_to = Column(String(128), nullable=True)
    officer_notes = Column(Text, nullable=True)
    detections_json = Column(Text, nullable=False, default="[]")
    timeline_json = Column(Text, nullable=False, default="[]")

    def to_dict(self):
        import json
        d = {
            "id": self.id,
            "complaint_id": self.complaint_id,
            "created_at": self.created_at,
            "status": self.status,
            "original_image": self.original_image,
            "annotated_image": self.annotated_image,
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "gps_accuracy": float(self.gps_accuracy) if self.gps_accuracy is not None else None,
            "altitude": float(self.altitude) if self.altitude is not None else None,
            "heading": float(self.heading) if self.heading is not None else None,
            "speed": float(self.speed) if self.speed is not None else None,
            "timestamp": self.timestamp,
            "pothole_count": int(self.pothole_count or 0),
            "crack_count": int(self.crack_count or 0),
            "total_defects": int(self.total_defects or 0),
            "phs": float(self.phs or 100.0),
            "condition": str(self.condition or "GOOD"),
            "priority": str(self.priority or "NORMAL"),
            "severity_avg": float(self.severity_avg or 0.0),
            "description": self.description or "",
            "citizen_phone": self.citizen_phone or "",
            "assigned_to": self.assigned_to or "",
            "officer_notes": self.officer_notes or "",
        }
        try:
            d["detections"] = json.loads(self.detections_json or "[]")
        except Exception:
            d["detections"] = []
        try:
            d["timeline"] = json.loads(self.timeline_json or "[]")
        except Exception:
            d["timeline"] = []

        # Derive accurate defect counts from detections if stored counts are 0
        if d["pothole_count"] == 0 and d["crack_count"] == 0 and d["detections"]:
            p_cnt = sum(1 for det in d["detections"] if "pothole" in str(det.get("class_name", "")).lower())
            c_cnt = sum(1 for det in d["detections"] if "crack" in str(det.get("class_name", "")).lower())
            d["pothole_count"] = p_cnt
            d["crack_count"] = c_cnt
            if d["total_defects"] == 0:
                d["total_defects"] = p_cnt + c_cnt

        return d


class NotificationModel(Base):
    """Stores in-app alerts for municipality officers."""
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    complaint_id = Column(String(64), nullable=False, index=True)
    type = Column(String(32), nullable=False, default="NEW_COMPLAINT")
    message = Column(Text, nullable=False)
    detail = Column(Text, nullable=True)
    is_read = Column(Integer, nullable=False, default=0, index=True)
    created_at = Column(String(64), nullable=False, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "complaint_id": self.complaint_id,
            "type": self.type,
            "message": self.message,
            "detail": self.detail,
            "is_read": bool(self.is_read),
            "created_at": self.created_at,
        }


class OfficerModel(Base):
    """Stores authorized municipal officers for portal access."""
    __tablename__ = "officers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    name = Column(String(128), nullable=False)
    role = Column(String(64), default="Officer")
    department = Column(String(128), default="Highway & Road Infrastructure Dept.")
    created_at = Column(String(64), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "department": self.department,
            "created_at": self.created_at,
        }


# ── Database Session Context Manager ─────────────────────────────────

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for SQLAlchemy ORM sessions with automatic commit/rollback."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        session.close()


def init_db():
    """Create all required tables if they don't exist in SQLite or PostgreSQL."""
    logger.info(f"Initializing {DB_TYPE} database schema...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"✅ {DB_TYPE} database tables verified/created successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database schema: {e}")
        raise
