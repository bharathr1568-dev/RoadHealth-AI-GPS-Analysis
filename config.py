"""
RoadHealth AI — Configuration
All tuneable parameters derived from the research notebook.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Baseline YOLO11m — used only as the Ultralytics wrapper shell
BASELINE_MODEL_PATH = Path(
    r"D:\STUDIES\FINAL YEAR PROJECT\Models"
    r"\RoadHealthDataset_Final3000_YOLO11m"
    r"\weights\best.pt"
)

# Proposed YOLO11m-P2P3-CBAM — Fine-Tuned Best Checkpoint
CBAM_MODEL_PATH = Path(
    r"D:\STUDIES\FINAL YEAR PROJECT\Models"
    r"\YOLO11m_P2P3_CBAM_FineTuned10"
    r"\weights\best.pt"
)

# Backwards compatibility alias
P2P3_MODEL_PATH = CBAM_MODEL_PATH
MODEL_PATH = CBAM_MODEL_PATH
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Inference ──────────────────────────────────────────────────
IMG_SIZE = 640
TILE_SIZE = 640
TILE_OVERLAP = 0.20

# Class-specific confidence thresholds
CONF_POTHOLE = 0.35
CONF_CRACK = 0.25
CONF_TILE_POTHOLE = 0.42
CONF_TILE_CRACK = 0.35

# CLAHE auxiliary inference pass
USE_CLAHE_AUX = False

# ── Detection processing ──────────────────────────────────────
IOU_MERGE_THRESHOLD = 0.45
POTHOLE_OVERLAP_THRESHOLD = 0.40
POTHOLE_IOU_THRESHOLD = 0.25

# ── Zone analysis ─────────────────────────────────────────────
ZONE_ROWS = 4
ZONE_COLS = 4

# ── Damage weights (notebook prototypes) ──────────────────────
POTHOLE_DAMAGE_WEIGHT = 1.00
CRACK_DAMAGE_WEIGHT = 0.55

# ── Maintenance thresholds ────────────────────────────────────
MAINTENANCE_HIGH = 0.70
MAINTENANCE_MEDIUM = 0.45

# ── Model info (YOLO11m-P2P3-CBAM — Proposed Champion Model) ──
MODEL_INFO = {
    "name": "YOLO11m-P2P3-CBAM (Proposed)",
    "yaml": "yolo11m.yaml (+ P2→P3 CBAM Attention)",
    "parameters": 20_392_696,
    "gflops": 68.3,
    "layers": 233,
    "task": "Object Detection",
    "classes": {0: "Pothole", 1: "Crack"},
    "num_classes": 2,
    "dataset": "RoadHealthDataset_Final3000",
    # Official validation metrics (900 test images)
    "precision": 0.884,
    "recall": 0.778,
    "mAP50": 0.8420,
    "mAP50_95": 0.583,
}

# ── Groq LLM ──────────────────────────────────────────────────
# Load .env if present locally (never committed)
_env_file = BASE_DIR / ".env"
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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


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


def trim_memory() -> bool:
    """
    Safely return unused heap memory back to the OS.
    Uses glibc's malloc_trim(0) on Linux (e.g. Render environment).
    Returns True if malloc_trim was successfully called, False otherwise.
    """
    try:
        import sys
        if sys.platform.startswith("linux"):
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                res = libc.malloc_trim(0)
                return bool(res)
    except Exception:
        pass
    return False


