"""
Backend Configuration
Server settings and paths for the FastAPI web application.
Reuses the existing project's model paths and inference parameters.
"""

import os
from pathlib import Path

# ── Project root (parent of backend/) ──────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))).parent

# ── Model paths ────────────────────────────────────────────────
# Baseline YOLO11m architecture shell (for Ultralytics wrapper)
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

# Fine-tuned YOLO11m-P2P3-CBAM champion weights
CBAM_MODEL_PATH = PROJECT_ROOT / "models" / "cbam_best.pt"

# Fallback to external paths if local copies are missing
if not BASELINE_MODEL_PATH.exists():
    BASELINE_MODEL_PATH = Path(
        r"D:\STUDIES\FINAL YEAR PROJECT\Models"
        r"\RoadHealthDataset_Final3000_YOLO11m"
        r"\weights\best.pt"
    )

if not CBAM_MODEL_PATH.exists():
    CBAM_MODEL_PATH = Path(
        r"D:\STUDIES\FINAL YEAR PROJECT\Models"
        r"\YOLO11m_P2P3_CBAM_FineTuned10"
        r"\weights\best.pt"
    )

# ── Output and storage directories ─────────────────────────────
BACKEND_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = BACKEND_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = BACKEND_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Database & Storage Configuration ──────────────────────────
DATABASE_DIR = BACKEND_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATABASE_DIR / "roadhealth.db"

# Dynamic Database URL (SQLite locally, PostgreSQL on Render)
_raw_db_url = os.environ.get("DATABASE_URL", "").strip()
if _raw_db_url:
    # Render provides postgres:// which SQLAlchemy expects as postgresql://
    if _raw_db_url.startswith("postgres://"):
        _raw_db_url = _raw_db_url.replace("postgres://", "postgresql://", 1)
    DATABASE_URL = _raw_db_url
    IS_POSTGRES = True
    DB_TYPE = "PostgreSQL"
else:
    DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"
    IS_POSTGRES = False
    DB_TYPE = "SQLite"

# Storage Configuration
STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local")  # "local", "cloudinary", "auto"

# ── Frontend path ─────────────────────────────────────────────
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Load .env if present locally (never committed)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        with open(_env_file, "r", encoding="utf-8") as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
    except Exception:
        pass

# ── Auth settings ─────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "roadcare_secret_key_2026_municipality")
DEFAULT_OFFICER_EMAIL = os.environ.get("OFFICER_EMAIL", "bharath4330")
DEFAULT_OFFICER_PASSWORD = os.environ.get("OFFICER_PASSWORD", "")
DEFAULT_OFFICER_NAME = os.environ.get("OFFICER_NAME", "Bharath")

# ── Inference parameters (from existing config.py) ────────────
IMG_SIZE = 640
TILE_SIZE = 640
TILE_OVERLAP = 0.20
CONF_POTHOLE = 0.35
CONF_CRACK = 0.25
CONF_TILE_POTHOLE = 0.42
CONF_TILE_CRACK = 0.35
USE_CLAHE_AUX = False
IOU_MERGE_THRESHOLD = 0.45

# ── Upload constraints ────────────────────────────────────────
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Server ────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000

# ── Model info ────────────────────────────────────────────────
MODEL_INFO = {
    "name": "YOLO11m-P2P3-CBAM (Proposed Champion)",
    "architecture": "YOLO11m + P2→P3 Detail Branch + CBAM Attention",
    "parameters": 20_392_696,
    "gflops": 68.3,
    "classes": {0: "Pothole", 1: "Crack"},
    "num_classes": 2,
    "dataset": "RoadHealthDataset_Final3000",
    "precision": 0.884,
    "recall": 0.778,
    "mAP50": 0.842,
    "mAP50_95": 0.583,
}


def get_rss_mb() -> float:
    """Return current process RSS memory in MB."""
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    except Exception:
        try:
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
        except Exception:
            return 0.0
