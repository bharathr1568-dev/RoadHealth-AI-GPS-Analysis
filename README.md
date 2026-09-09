# RoadCare AI — Two-Sided Road Complaint & Municipal Road Intelligence Platform

An end-to-end civic technology platform for AI-powered road defect analysis, citizen reporting, and municipal complaint management.

Powered by the research champion model: **YOLO11m-P2P3-CBAM (Channel & Spatial Attention)** with **Road Health Scoring (RHS)**, high-accuracy GPS defect localization, and interactive GIS mapping.

---

## 🌟 Key Features

### 🏙️ Citizen Portal
- **Hero & Reporting Wizard**: 4-step mobile-first reporting interface.
- **Image Capture**: Direct camera capture or photo upload (JPG, PNG, WEBP).
- **High-Precision GPS**: Browser Geolocation API with `enableHighAccuracy: true` and accuracy threshold warnings.
- **AI Road Analysis**: Real-time YOLO11m-P2P3-CBAM inference with multi-pass tiling and CLAHE.
- **Visual Defect Comparison**: Side-by-side view of **Original Citizen Image** vs **AI Detected Image (with bounding boxes, confidence %, and severities)**.
- **Road Health Score (RHS)**: 0–100 numerical index (Critical, Poor, Moderate, Good, Excellent).
- **Complaint Submission**: Generates sequential IDs (e.g. `RCA-2026-000001`) stored in SQLite database.
- **Track Complaint / My Reports**: Lookup any complaint ID or review complaints reported on the device with live audit timelines.

### 🏛️ Municipality Officer Portal
- **Officer Sign In**: Secure municipal officer authentication configured via environment variables.
- **Real-Time Notifications**: Live notification badge polling (every 8s) alerting officers when new complaints arrive.
- **Executive Dashboard**: KPI summary cards (Total Complaints, New, Under Review, Critical Roads, Resolved) and AI-Assisted Priority Alert banners.
- **Complaints Management**: Filter by status (`NEW`, `UNDER REVIEW`, `VERIFIED`, `IN PROGRESS`, `RESOLVED`, `REJECTED`) and condition (`CRITICAL`, `POOR`, `MODERATE`, `GOOD`), with search and sorting.
- **Detailed Inspection Modal**: High-res original vs annotated AI images, defect itemization, interactive Leaflet map, audit timeline, and status progression controls.
- **Municipal GIS Map**: Leaflet OpenStreetMap with color-coded condition markers (🔴 Red, 🟠 Orange, 🟡 Yellow, 🟢 Green) and direct inspection popups.
- **Defect Analytics**: Potholes vs cracks distribution and road condition health breakdowns.

---

## 🧠 AI Model Architecture

- **Base Network**: YOLO11m (20.4M parameters)
- **Detail Branch**: P2 (high-resolution) downsampled and fused into P3 via CBAM (Convolutional Block Attention Module)
- **Attention Modules**:
  - **Channel Attention**: Emphasizes feature channels corresponding to defect textures
  - **Spatial Attention**: Pinpoints spatial bounding locations of potholes and cracks
- **Inference Pipeline**: Multi-pass full-image + tiled inference + CLAHE auxiliary enhancement + morphological black-hat crack refinement + class-aware NMS

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Platform Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3. Open in Browser
- **URL**: `http://localhost:8000`
- **Officer Login**: Configured via `OFFICER_EMAIL` and `OFFICER_PASSWORD` environment variables (or `.env` file).

---

## 🧪 Running Tests

```bash
# Run complete E2E test suite
python tests/test_platform.py

# Run live server integration verification
python tests/verify_full_system.py
```

---

## 📁 Repository Structure

```
d:/RoadHealth_AI_GPS_Analysis/
├── backend/
│   ├── main.py                     # FastAPI application entrypoint & static mounts
│   ├── config.py                   # Server settings, paths, model metadata
│   ├── database/
│   │   ├── database.py             # SQLite connection & schema initialization
│   │   └── seed_data.py            # Sample complaints & default officer account
│   ├── api/
│   │   ├── routes.py               # Route aggregator
│   │   ├── auth.py                 # Officer login & session validation
│   │   ├── complaints.py           # Complaint creation, querying, & status updates
│   │   ├── notifications.py        # Real-time notification endpoints
│   │   └── dashboard.py            # Dashboard KPIs & map markers endpoint
│   ├── services/
│   │   ├── inference.py            # YOLO11m-P2P3-CBAM champion inference engine
│   │   ├── phs.py                  # Road Health Score computation
│   │   ├── location.py             # GPS validation & map formatting
│   │   ├── complaint_service.py    # Complaint persistence & timeline auditing
│   │   └── notification_service.py # In-app notification dispatcher
│   ├── uploads/                    # Stored citizen original photos
│   └── results/                    # Stored AI annotated defect images
│
├── frontend/
│   ├── index.html                  # Responsive two-sided UI (Citizen + Municipality)
│   ├── style.css                   # Dark slate glassmorphism civic-tech design system
│   └── script.js                   # Client SPA logic, Leaflet maps, & live polling
│
├── models/
│   ├── best.pt                     # Baseline YOLO11m shell
│   └── cbam_best.pt                # Champion YOLO11m-P2P3-CBAM weights
│
└── tests/
    ├── test_platform.py            # Platform E2E test suite
    └── verify_full_system.py       # Live server integration verification
```
