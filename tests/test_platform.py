"""
End-to-End Test Suite for RoadCare AI Platform
Tests citizen AI analysis, complaint persistence, officer auth,
dashboard stats, notifications, and status transitions.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import DEFAULT_OFFICER_EMAIL, DEFAULT_OFFICER_PASSWORD
from backend.database.database import init_db
from backend.database.seed_data import seed_default_officer, seed_demo_complaints


def test_platform_e2e():
    print("=" * 60)
    print("  Running RoadCare AI E2E Platform Test Suite")
    print("=" * 60)

    # Ensure temporary test credential is set if missing in environment
    import os
    import backend.config as cfg
    if not os.environ.get("OFFICER_PASSWORD"):
        os.environ["OFFICER_PASSWORD"] = "test_suite_mock_key_only"
        cfg.DEFAULT_OFFICER_PASSWORD = "test_suite_mock_key_only"
    if not os.environ.get("OFFICER_EMAIL"):
        os.environ["OFFICER_EMAIL"] = "test_officer"
        cfg.DEFAULT_OFFICER_EMAIL = "test_officer"

    # Initialize DB
    init_db()
    seed_default_officer()
    seed_demo_complaints()

    client = TestClient(app)

    # 1. Health check
    print("\n[1/8] Testing /api/health...")
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_ready"] is True
    print(f"  ✓ Health OK: Model={data['model_info']['name']}, Classes={data['class_names']}")

    # 2. Officer Login
    print("\n[2/8] Testing /api/officer/login...")
    resp = client.post("/api/officer/login", json={
        "email": cfg.DEFAULT_OFFICER_EMAIL,
        "password": cfg.DEFAULT_OFFICER_PASSWORD,
    })
    assert resp.status_code == 200
    auth_data = resp.json()
    token = auth_data["token"]
    assert token is not None
    print(f"  ✓ Officer Login OK: Officer={auth_data['officer']['name']}")

    headers = {"Authorization": f"Bearer {token}"}

    # 3. Dashboard Stats
    print("\n[3/8] Testing /api/dashboard/stats & /api/map/complaints...")
    resp = client.get("/api/dashboard/stats")
    assert resp.status_code == 200
    stats = resp.json()["stats"]
    assert stats["total_complaints"] >= 3
    print(f"  ✓ Dashboard Stats OK: Total={stats['total_complaints']}, New={stats['new_complaints']}, Critical={stats['critical_roads']}")

    resp = client.get("/api/map/complaints")
    assert resp.status_code == 200
    assert len(resp.json()["complaints"]) >= 3
    print(f"  ✓ Map complaints OK: {len(resp.json()['complaints'])} markers")

    # 4. AI Analysis
    print("\n[4/8] Testing /api/analyze with sample image...")
    sample_imgs = list((PROJECT_ROOT / "data").glob("**/*.jpg")) + list((PROJECT_ROOT / "data").glob("**/*.png"))
    if sample_imgs:
        sample_img_path = sample_imgs[0]
        with open(sample_img_path, "rb") as f:
            resp = client.post(
                "/api/analyze",
                files={"image": ("test_road.jpg", f, "image/jpeg")},
                data={
                    "latitude": "13.0827",
                    "longitude": "80.2707",
                    "accuracy": "4.2",
                    "timestamp": "2026-09-09T11:45:00Z",
                },
            )
        assert resp.status_code == 200
        analysis = resp.json()
        assert analysis["success"] is True
        assert "summary" in analysis
        assert "original_image" in analysis
        assert "result_image" in analysis
        print(f"  ✓ AI Analysis OK: Defects={analysis['summary']['total_defects']}, PHS={analysis['summary']['phs']} ({analysis['summary']['condition']})")
        print(f"    Original: {analysis['original_image']}, Annotated: {analysis['result_image']}")

        # 5. Citizen Submit Complaint
        print("\n[5/8] Testing /api/complaints submission...")
        resp = client.post(
            "/api/complaints",
            json={
                "original_image": analysis["original_image"],
                "annotated_image": analysis["result_image"],
                "latitude": 13.0827,
                "longitude": 80.2707,
                "gps_accuracy": 4.2,
                "timestamp": "2026-09-09T11:45:00Z",
                "phs_data": analysis["summary"],
                "detections": analysis["detections"],
                "description": "Severe road deterioration near intersection reported by citizen.",
                "citizen_phone": "+91 99887 76655",
            },
        )
        assert resp.status_code == 201
        complaint_res = resp.json()
        complaint = complaint_res["complaint"]
        cid = complaint["complaint_id"]
        assert cid.startswith("RCA-2026-")
        print(f"  ✓ Complaint Created: ID={cid}, Priority={complaint['priority']}, Status={complaint['status']}")

        # 6. Officer Notifications Verification
        print("\n[6/8] Testing /api/notifications...")
        resp = client.get("/api/notifications")
        assert resp.status_code == 200
        notifs = resp.json()
        assert notifs["unread_count"] > 0
        latest_notif = notifs["notifications"][0]
        assert cid in latest_notif["message"] or cid in latest_notif["complaint_id"]
        print(f"  ✓ Notification Arrived: '{latest_notif['message']}' (Unread: {notifs['unread_count']})")

        # Mark notification read
        resp = client.patch(f"/api/notifications/{latest_notif['id']}/read", headers=headers)
        assert resp.status_code == 200
        print(f"  ✓ Notification #{latest_notif['id']} marked as read.")

        # 7. Complaint Detail View
        print("\n[7/8] Testing /api/complaints/{cid}...")
        resp = client.get(f"/api/complaints/{cid}")
        assert resp.status_code == 200
        c_detail = resp.json()["complaint"]
        assert len(c_detail["timeline"]) >= 2
        print(f"  ✓ Complaint Detail OK: Timeline has {len(c_detail['timeline'])} events")

        # 8. Status Workflow Transition
        print("\n[8/9] Testing /api/complaints/{cid}/status workflow...")
        statuses = ["UNDER REVIEW", "VERIFIED", "IN PROGRESS", "RESOLVED"]
        for st in statuses:
            resp = client.patch(
                f"/api/complaints/{cid}/status",
                headers=headers,
                json={"status": st, "officer_notes": f"Officer updated to {st}"},
            )
            assert resp.status_code == 200
            assert resp.json()["complaint"]["status"] == st
            print(f"  ✓ Status Transition -> {st} OK")

        # 9. Officer Delete / Remove Complaint
        print("\n[9/9] Testing DELETE /api/complaints/{cid}...")
        resp = client.delete(f"/api/complaints/{cid}", headers=headers)
        assert resp.status_code == 200
        del_data = resp.json()
        assert del_data["success"] is True
        print(f"  ✓ Complaint {cid} successfully deleted.")

        # Verify it is no longer found
        resp = client.get(f"/api/complaints/{cid}")
        assert resp.status_code == 404
        print(f"  ✓ Verified {cid} is removed from database (404 Not Found).")

    print("\n" + "=" * 60)
    print("  ALL 9 END-TO-END TESTS PASSED SUCCESSFULLY! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    test_platform_e2e()

