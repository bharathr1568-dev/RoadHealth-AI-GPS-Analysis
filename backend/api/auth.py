"""
Auth API — SQLAlchemy ORM Compatible
Officer authentication endpoints and session token validation.
Supports both SQLite and PostgreSQL.
"""

import hmac
import hashlib
import time
import json
import base64
import logging
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Header, Depends

from backend.config import SECRET_KEY, DEFAULT_OFFICER_EMAIL, DEFAULT_OFFICER_PASSWORD, DEFAULT_OFFICER_NAME
from backend.database.database import get_db_session, OfficerModel
from backend.database.seed_data import hash_password

logger = logging.getLogger("roadhealth.auth")

router = APIRouter(prefix="/api/officer", tags=["Officer Auth"])


class LoginRequest(BaseModel):
    email: str
    password: str


class OfficerResponse(BaseModel):
    email: str
    name: str
    role: str
    department: str
    token: str


def create_token(email: str, name: str) -> str:
    """Generate a simple signed token for session authentication."""
    payload = {
        "email": email,
        "name": name,
        "exp": int(time.time()) + (86400 * 7),  # 7 days
    }
    payload_str = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    signature = hmac.new(SECRET_KEY.encode(), payload_str.encode(), hashlib.sha256).hexdigest()
    return f"{payload_str}.{signature}"


def verify_token(token: Optional[str]) -> Optional[dict]:
    """Verify signed token and return payload if valid."""
    if not token:
        return None
    try:
        # Strip Bearer if present
        if token.startswith("Bearer "):
            token = token[7:]
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_str, signature = parts
        expected_sig = hmac.new(SECRET_KEY.encode(), payload_str.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        payload = json.loads(base64.urlsafe_b64decode(payload_str.encode()).decode())
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def get_current_officer(authorization: Optional[str] = Header(None)) -> dict:
    """FastAPI dependency to enforce officer authentication."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    payload = verify_token(authorization)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired session token. Please log in again.")
    return payload


@router.post("/login")
async def officer_login(req: LoginRequest):
    """Authenticate officer with email and password."""
    email = req.email.strip().lower()
    password = req.password.strip()

    with get_db_session() as session:
        officer = (
            session.query(OfficerModel)
            .filter(OfficerModel.email == email)
            .first()
        )

        import os
        env_email = os.environ.get("OFFICER_EMAIL", DEFAULT_OFFICER_EMAIL).strip().lower()
        env_password = os.environ.get("OFFICER_PASSWORD", DEFAULT_OFFICER_PASSWORD).strip()
        env_name = os.environ.get("OFFICER_NAME", DEFAULT_OFFICER_NAME).strip()

        if not officer:
            # Check if environment credentials match (only when non-empty password is configured)
            if env_password and email == env_email and password == env_password:
                # Seed default officer into database
                new_officer = OfficerModel(
                    email=env_email,
                    password_hash=hash_password(env_password),
                    name=env_name,
                    role="Municipal Road Inspector",
                    department="Highway & Road Infrastructure Dept.",
                    created_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                )
                session.add(new_officer)
                token = create_token(env_email, env_name)
                return {
                    "success": True,
                    "officer": {
                        "email": env_email,
                        "name": env_name,
                        "role": "Municipal Road Inspector",
                        "department": "Highway & Road Infrastructure Dept.",
                    },
                    "token": token,
                }
            raise HTTPException(status_code=401, detail="Invalid officer User ID or password.")

        expected_hash = hash_password(password)
        if officer.password_hash != expected_hash:
            if env_password and email == env_email and password == env_password:
                officer.password_hash = expected_hash
                officer.name = env_name
            else:
                raise HTTPException(status_code=401, detail="Invalid officer User ID or password.")

        token = create_token(officer.email, officer.name)
        return {
            "success": True,
            "officer": {
                "email": officer.email,
                "name": officer.name,
                "role": officer.role,
                "department": officer.department,
            },
            "token": token,
        }


@router.get("/me")
async def get_me(officer: dict = Depends(get_current_officer)):
    """Return currently logged-in officer profile."""
    return {"success": True, "officer": officer}
