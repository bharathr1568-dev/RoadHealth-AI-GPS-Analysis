"""
RoadHealth AI — FastAPI Application
Main entry point for the two-sided road complaint and AI analysis platform.

Serves:
  - Frontend static files (HTML/CSS/JS) at /
  - Citizen uploads at /uploads/
  - Annotated result images at /results/
  - API endpoints at /api/*

Run with:
  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
# Restrict multi-threaded memory allocation for CPU runtime
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
from contextlib import asynccontextmanager

import torch
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import FRONTEND_DIR, RESULTS_DIR, UPLOADS_DIR, MODEL_INFO, get_rss_mb
from backend.database.database import init_db
from backend.database.seed_data import seed_default_officer, seed_demo_complaints
from backend.api.routes import router

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("roadhealth.main")


# ── Lifespan: DB initialization & Model Pre-loading ──────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite database, seed default data, and pre-load AI model."""
    logger.info("=" * 65)
    logger.info("  RoadCare AI — Starting Municipal & Citizen Platform")
    logger.info(f"  AI Model: {MODEL_INFO['name']}")
    logger.info(f"  [MEM] Application startup | RSS: {get_rss_mb():.1f} MB")
    logger.info("=" * 65)

    # 1. Initialize SQLite database & seed
    try:
        init_db()
        seed_default_officer()
        seed_demo_complaints()
        logger.info("✅ SQLite database and seed records ready.")
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")

    # 2. Pre-load YOLO11m-P2P3-CBAM model
    try:
        from backend.services.inference import load_model
        load_model()
        logger.info(f"✅ Model pre-loaded successfully | RSS: {get_rss_mb():.1f} MB. Server is ready.")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        logger.error("   The server will start but /api/analyze will fail.")

    yield
    logger.info("Server shutting down.")


# ── FastAPI application ──────────────────────────────────────────

app = FastAPI(
    title="RoadCare AI — Citizen & Municipality Platform",
    description=(
        "Professional two-sided road complaint and AI analysis platform "
        "using YOLO11m-P2P3-CBAM with RHS scoring and GIS tracking."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS (allow local development from any origin) ───────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ───────────────────────────────────────────────────
app.include_router(router)

# ── Static file mounts ───────────────────────────────────────────
# Citizen uploads (original photos)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# AI Results (annotated images)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Frontend (HTML/CSS/JS) — mounted last as catch-all
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
