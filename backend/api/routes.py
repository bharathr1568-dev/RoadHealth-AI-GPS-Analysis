"""
API Routes Aggregator
FastAPI endpoints for road condition analysis, complaints, notifications, auth, and dashboard.
"""

import logging
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from backend.config import MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, MODEL_INFO, get_rss_mb
from backend.services.inference import run_full_pipeline, get_class_names, load_model
from backend.services.phs import format_phs_response
from backend.services.location import (
    validate_coordinates,
    format_location_response,
)

from backend.api.auth import router as auth_router
from backend.api.complaints import router as complaints_router
from backend.api.notifications import router as notifications_router
from backend.api.dashboard import router as dashboard_router

logger = logging.getLogger("roadhealth.api")
router = APIRouter()

# Include sub-routers
router.include_router(auth_router)
router.include_router(complaints_router)
router.include_router(notifications_router)
router.include_router(dashboard_router)


# ── Health check ──────────────────────────────────────────────────

@router.get("/api/health")
async def health_check():
    """Return server health and model readiness status."""
    try:
        model = load_model()
        model_ready = model is not None
        class_names = get_class_names()
    except Exception as e:
        model_ready = False
        class_names = {}
        logger.error(f"Model health check failed: {e}")

    return {
        "status": "healthy",
        "model_ready": model_ready,
        "model_info": MODEL_INFO,
        "class_names": class_names,
    }


# ── Main analysis endpoint ───────────────────────────────────────

@router.post("/api/analyze")
async def analyze_road(
    image: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    accuracy: float = Form(None),
    timestamp: str = Form(None),
):
    """
    Analyze a road image for pavement defects using YOLO11m-P2P3-CBAM.

    Accepts multipart/form-data with:
      - image: road photograph (JPG, PNG, WEBP)
      - latitude: GPS latitude
      - longitude: GPS longitude
      - accuracy: GPS accuracy in meters (optional)
      - timestamp: capture timestamp (optional)
    """
    t_start = time.time()

    # ── 1. Validate GPS coordinates ──────────────────────────────
    coord_check = validate_coordinates(latitude, longitude, accuracy)
    if not coord_check["valid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid GPS coordinates: {'; '.join(coord_check['errors'])}"
        )

    # ── 2. Validate uploaded file ────────────────────────────────
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")

    # Check extension
    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check content type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {image.content_type}. Expected an image."
        )

    # ── 3. Read and decode image ─────────────────────────────────
    try:
        file_bytes = await image.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # Check file size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Decode image
    try:
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Image decode error: {e}")
        raise HTTPException(status_code=400, detail="Failed to decode image. File may be corrupted.")

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Failed to decode image. Ensure the file is a valid image."
        )

    h, w = image_bgr.shape[:2]
    if h < 32 or w < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({w}×{h}). Minimum 32×32 pixels."
        )

    logger.info(f"Image accepted: {image.filename} ({w}×{h}, {size_mb:.2f} MB)")

    # ── 4. Run AI inference pipeline ─────────────────────────────
    try:
        pipeline_result = run_full_pipeline(image_bgr)
    except Exception as e:
        logger.error(f"Inference pipeline error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {str(e)}"
        )

    # ── 5. Build response ────────────────────────────────────────
    t_elapsed = time.time() - t_start

    # PHS formatting
    phs_data = format_phs_response(pipeline_result["summary"])

    # Location formatting
    location_data = format_location_response(
        latitude, longitude, accuracy, timestamp
    )

    original_url = pipeline_result.get("original_image_url") or f"/uploads/{pipeline_result.get('original_image_filename', '')}"
    annotated_url = pipeline_result.get("result_image_url") or f"/results/{pipeline_result.get('result_image_filename', '')}"

    response = {
        "success": True,
        "detections": pipeline_result["detections"],
        "summary": phs_data,
        "location": location_data,
        "original_image": original_url,
        "result_image": annotated_url,
        "annotated_image": annotated_url,
        "analysis_meta": {
            "raw_detections": pipeline_result["raw_count"],
            "after_nms": pipeline_result["nms_count"],
            "final_detections": pipeline_result["final_count"],
            "image_size": f"{pipeline_result['image_width']}×{pipeline_result['image_height']}",
            "processing_time_sec": round(t_elapsed, 2),
            "model": MODEL_INFO["name"],
        },
    }

    logger.info(
        f"[DIAG 10/10] Analysis complete: {phs_data['total_defects']} defects, "
        f"PHS={phs_data['phs']}, {t_elapsed:.2f}s | [MEM] Before API response: {get_rss_mb():.1f} MB. Returning API response..."
    )

    return response
