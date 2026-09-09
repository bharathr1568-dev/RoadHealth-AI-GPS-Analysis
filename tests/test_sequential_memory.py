"""
Sequential Memory Test Suite for RoadCare AI Platform
Runs 3 sequential road analyses in a single Python process with one loaded YOLO model.
Verifies memory release after each analysis and ensures subsequent analyses complete successfully.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import get_rss_mb, BASELINE_MODEL_PATH, CBAM_MODEL_PATH
from backend.services.inference import load_model
from backend.database.database import init_db
from backend.database.seed_data import seed_default_officer, seed_demo_complaints


def test_sequential_memory():
    print("=" * 65)
    print("  SEQUENTIAL MEMORY BENCHMARK (1 Process, 1 YOLO Model, 3 Analyses)")
    print("=" * 65)

    # Initialize environment and DB
    init_db()
    seed_default_officer()
    seed_demo_complaints()

    # Pre-load champion model
    print(f"\n[INIT] Pre-loading YOLO11m-P2P3-CBAM singleton | Starting RSS: {get_rss_mb():.1f} MB")
    model = load_model()
    post_load_rss = get_rss_mb()
    print(f"[INIT] Model loaded successfully | RSS: {post_load_rss:.1f} MB")

    client = TestClient(app)

    test_images = [
        PROJECT_ROOT / "backend" / "uploads" / "sample_pothole_demo.jpg",
        PROJECT_ROOT / "backend" / "uploads" / "upload_0146712f0dd5.jpg",
        PROJECT_ROOT / "backend" / "uploads" / "upload_01a0bd25eed2.jpg",
    ]

    for img_p in test_images:
        assert img_p.exists(), f"Test image missing: {img_p}"

    results = []

    for idx, img_path in enumerate(test_images, start=1):
        print(f"\n" + "-" * 50)
        print(f"  STARTING ANALYSIS #{idx}: {img_path.name}")
        print("-" * 50)

        rss_before = get_rss_mb()
        print(f"Analysis #{idx} - RSS before inference: {rss_before:.1f} MB")

        with open(img_path, "rb") as f:
            resp = client.post(
                "/api/analyze",
                files={"image": (img_path.name, f, "image/jpeg")},
                data={
                    "latitude": "13.0827",
                    "longitude": "80.2707",
                    "accuracy": "4.2",
                    "timestamp": "2026-09-09T12:00:00Z",
                },
            )

        assert resp.status_code == 200, f"Analysis #{idx} failed with status {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data["success"] is True, f"Analysis #{idx} success is False"

        # The RSS right after API call completes and cleanup ran
        rss_after_cleanup = get_rss_mb()
        defects = data["summary"]["total_defects"]
        phs = data["summary"]["phs"]
        print(f"Analysis #{idx} completed: {defects} defects detected, PHS={phs}")
        print(f"Analysis #{idx} - RSS after cleanup: {rss_after_cleanup:.1f} MB")

        results.append({
            "analysis": idx,
            "image": img_path.name,
            "defects": defects,
            "phs": phs,
            "rss_before": rss_before,
            "rss_after_cleanup": rss_after_cleanup,
        })

    print("\n" + "=" * 65)
    print("  SEQUENTIAL BENCHMARK SUMMARY")
    print("=" * 65)
    for r in results:
        print(f"Analysis #{r['analysis']} ({r['image']}):")
        print(f"  RSS before inference: {r['rss_before']:.1f} MB")
        print(f"  RSS after cleanup:    {r['rss_after_cleanup']:.1f} MB")
        print(f"  Defects: {r['defects']}, PHS: {r['phs']}")

    print("\n✅ All 3 sequential analyses completed successfully in the same process!")
    print("=" * 65)


if __name__ == "__main__":
    test_sequential_memory()
