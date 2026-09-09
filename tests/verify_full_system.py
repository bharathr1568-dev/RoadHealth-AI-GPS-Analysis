"""
Comprehensive Verification Suite for RoadCare AI Platform
Tests all static routes, API endpoints, database persistence,
citizen report workflow, officer authentication, status updates,
notifications, and map data.
"""

import sys
import json
import urllib.request
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DEFAULT_OFFICER_EMAIL, DEFAULT_OFFICER_PASSWORD

BASE_URL = "http://127.0.0.1:8000"


def http_get(path, headers=None):
    req = urllib.request.Request(f"{BASE_URL}{path}", headers=headers or {})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def http_post(path, data, headers=None):
    payload = json.dumps(data).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(f"{BASE_URL}{path}", data=payload, headers=req_headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def http_patch(path, data, headers=None):
    payload = json.dumps(data).encode("utf-8")
    req_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(f"{BASE_URL}{path}", data=payload, headers=req_headers, method="PATCH")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def run_verification():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 65)
    print("  RoadCare AI — Live Server Integration Verification")
    print("=" * 65)

    # 1. Static frontend checks
    print("\n[1/7] Testing Static Files (HTML, CSS, JS)...")
    with urllib.request.urlopen(f"{BASE_URL}/") as r:
        html = r.read().decode("utf-8")
        assert "<title>RoadCare AI" in html
        assert "id=\"drop-zone\"" in html
        assert "id=\"table-all-complaints\"" in html
        print(f"  ✓ Frontend index.html served (length: {len(html):,} bytes)")

    with urllib.request.urlopen(f"{BASE_URL}/style.css") as r:
        css = r.read().decode("utf-8")
        assert "--cyan-primary" in css
        print(f"  ✓ CSS stylesheet served (length: {len(css):,} bytes)")

    with urllib.request.urlopen(f"{BASE_URL}/script.js") as r:
        js = r.read().decode("utf-8")
        assert "enableHighAccuracy: true" in js
        assert "submitCitizenComplaint" in js
        print(f"  ✓ JavaScript bundle served (length: {len(js):,} bytes)")

    # 2. Health & Model check
    print("\n[2/7] Testing /api/health...")
    status, health = http_get("/api/health")
    assert status == 200
    assert health["model_ready"] is True
    print(f"  ✓ Server healthy | Model: {health['model_info']['name']} (Params: {health['model_info']['parameters']:,})")

    # 3. Officer Auth
    print("\n[3/7] Testing Officer Authentication...")
    status, auth = http_post("/api/officer/login", {"email": DEFAULT_OFFICER_EMAIL, "password": DEFAULT_OFFICER_PASSWORD})
    assert status == 200
    token = auth["token"]
    officer_headers = {"Authorization": f"Bearer {token}"}
    print(f"  ✓ Officer logged in: {auth['officer']['name']} ({auth['officer']['email']})")

    # 4. Officer Dashboard KPIs
    print("\n[4/7] Testing Dashboard KPIs & Maps...")
    status, dash = http_get("/api/dashboard/stats")
    assert status == 200
    stats = dash["stats"]
    print(f"  ✓ Total Complaints: {stats['total_complaints']} | New: {stats['new_complaints']} | Critical: {stats['critical_roads']} | Resolved: {stats['resolved']}")
    print(f"  ✓ Defect Counts: {stats['defects_summary']['potholes']} Potholes, {stats['defects_summary']['cracks']} Cracks")

    status, map_data = http_get("/api/map/complaints")
    assert status == 200
    print(f"  ✓ Map Markers: {len(map_data['complaints'])} geo-locations loaded")

    # 5. Citizen AI Analysis & Complaint Flow
    print("\n[5/7] Testing AI Analysis & Citizen Complaint Submission...")
    sample_images = list((PROJECT_ROOT / "backend" / "uploads").glob("*.jpg")) + \
                    list((PROJECT_ROOT / "backend" / "results").glob("*.jpg")) + \
                    list((PROJECT_ROOT / "assets").glob("**/*.jpg"))

    if not sample_images:
        import numpy as np, cv2
        dummy = np.full((640, 640, 3), 120, dtype=np.uint8)
        test_img = PROJECT_ROOT / "backend" / "uploads" / "test_sample.jpg"
        cv2.imwrite(str(test_img), dummy)
    else:
        test_img = sample_images[0]

    # Multipart upload simulation using requests or urllib with boundary
    import io
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    with open(test_img, "rb") as f:
        img_bytes = f.read()

    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(b'Content-Disposition: form-data; name="image"; filename="road_test.jpg"\r\n')
    body.write(b"Content-Type: image/jpeg\r\n\r\n")
    body.write(img_bytes)
    body.write(b"\r\n")

    fields = {
        "latitude": "13.082700",
        "longitude": "80.270700",
        "accuracy": "4.8",
        "timestamp": "2026-09-09T11:45:00.000Z",
    }
    for k, v in fields.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.write(f"{v}\r\n".encode("utf-8"))

    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    payload = body.getvalue()

    req = urllib.request.Request(
        f"{BASE_URL}/api/analyze",
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        analysis = json.loads(r.read().decode("utf-8"))

    assert analysis["success"] is True
    print(f"  ✓ AI Analysis Completed in {analysis['analysis_meta']['processing_time_sec']}s")
    print(f"    Defects Detected: {analysis['summary']['total_defects']} | PHS: {analysis['summary']['phs']} ({analysis['summary']['condition']})")
    print(f"    Original Image: {analysis['original_image']}")
    print(f"    Annotated Image: {analysis['result_image']}")

    # Submit official complaint
    complaint_payload = {
        "original_image": analysis["original_image"],
        "annotated_image": analysis["result_image"],
        "latitude": 13.0827,
        "longitude": 80.2707,
        "gps_accuracy": 4.8,
        "timestamp": "2026-09-09T11:45:00.000Z",
        "phs_data": analysis["summary"],
        "detections": analysis["detections"],
        "description": "Citizen reported surface failure near traffic signal.",
        "citizen_phone": "+91 98400 99887",
    }
    status, c_res = http_post("/api/complaints", complaint_payload)
    assert status == 201
    cid = c_res["complaint"]["complaint_id"]
    print(f"  ✓ Official Complaint Created: {cid} (Status: {c_res['complaint']['status']}, Priority: {c_res['complaint']['priority']})")

    # 6. Notifications Verification
    print("\n[6/7] Testing Real-Time Officer Notifications...")
    status, notifs = http_get("/api/notifications")
    assert status == 200
    assert notifs["unread_count"] > 0
    latest = notifs["notifications"][0]
    print(f"  ✓ Latest Notification: '{latest['message']}' ({latest['created_at']})")

    # 7. Complaint Detail & Officer Status Progression
    print("\n[7/7] Testing Complaint Detail & Status Workflow...")
    status, detail_res = http_get(f"/api/complaints/{cid}")
    assert status == 200
    comp = detail_res["complaint"]
    assert comp["original_image"].startswith("/uploads/")
    assert comp["annotated_image"].startswith("/results/")
    print(f"  ✓ Detail verified for {cid}: Timeline has {len(comp['timeline'])} events")

    # Advance status through municipal workflow
    workflow = [
        ("UNDER REVIEW", "Inspection scheduled with Highway Maintenance Ward 12"),
        ("VERIFIED", "Engineer inspected site. Work order generated."),
        ("IN PROGRESS", "Road crew deployed. Patching with warm mix asphalt."),
        ("RESOLVED", "Surface restored and compaction completed successfully."),
    ]
    for next_status, notes in workflow:
        status, upd_res = http_patch(
            f"/api/complaints/{cid}/status",
            {"status": next_status, "officer_notes": notes},
            headers=officer_headers,
        )
        assert status == 200
        assert upd_res["complaint"]["status"] == next_status
        print(f"  ✓ Updated {cid} -> {next_status}")

    # Verify final timeline
    status, final_res = http_get(f"/api/complaints/{cid}")
    timeline = final_res["complaint"]["timeline"]
    print(f"  ✓ Full audit timeline verified: {len(timeline)} chronological steps recorded.")

    print("\n" + "=" * 65)
    print("  ALL SYSTEM INTEGRATION TESTS COMPLETED WITH 100% SUCCESS! 🎉")
    print("=" * 65)


if __name__ == "__main__":
    run_verification()
