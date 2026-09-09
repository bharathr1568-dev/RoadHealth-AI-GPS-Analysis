/**
 * RoadCare AI — Client Application Script
 * Continuous High-Accuracy GPS Acquisition with Satellite Locking,
 * AI Defect Visualization, Complaint Submissions, and Municipality Officer Portal.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ═══════════════════════════════════════════════════════════════════
    // APPLICATION STATE
    // ═══════════════════════════════════════════════════════════════════
    const state = {
        activePortal: "citizen", // "citizen" | "officer"
        activeCitizenTab: "report", // "report" | "track" | "about"
        activeOfficerTab: "dash", // "dash" | "complaints" | "map" | "analytics" | "notifs"

        // Citizen Workflow State
        selectedFile: null,
        analysisResult: null,
        submittedComplaintId: null,

        // Continuous GPS Tracking State
        gpsWatchId: null,
        gpsReadingsCount: 0,
        bestGpsReading: null,
        confirmedGpsData: null,
        gpsData: null, // Points to confirmedGpsData once locked

        // Maps
        gpsLiveMap: null,
        gpsLiveMarker: null,
        gpsLiveCircle: null,
        citizenMap: null,
        citizenMarker: null,
        citizenCircle: null,
        officerMap: null,
        officerMarkers: [],
        modalMap: null,
        modalMarker: null,
        modalCircle: null,

        // Officer Session (Persists across page refreshes)
        officerToken: localStorage.getItem("roadcare_officer_token") || sessionStorage.getItem("roadcare_officer_token") || null,
        officerUser: JSON.parse(localStorage.getItem("roadcare_officer_user") || sessionStorage.getItem("roadcare_officer_user") || "null"),

        // Polling Timer
        pollingTimer: null,
    };

    // ═══════════════════════════════════════════════════════════════════
    // DOM ELEMENT REFERENCES
    // ═══════════════════════════════════════════════════════════════════
    const el = {
        // Portal Switchers
        btnPortalCitizen: document.getElementById("btn-portal-citizen"),
        btnPortalOfficer: document.getElementById("btn-portal-officer"),
        viewCitizen: document.getElementById("view-citizen"),
        viewOfficer: document.getElementById("view-officer"),
        navNotifBadge: document.getElementById("nav-notif-badge"),
        officerSessionPill: document.getElementById("officer-session-pill"),
        navOfficerName: document.getElementById("nav-officer-name"),
        btnOfficerLogout: document.getElementById("btn-officer-logout"),
        brandHomeBtn: document.getElementById("brand-home-btn"),

        // Citizen Subnav
        cnavReport: document.getElementById("cnav-report"),
        cnavTrack: document.getElementById("cnav-track"),
        cnavAbout: document.getElementById("cnav-about"),
        citizenReportView: document.getElementById("citizen-report-view"),
        citizenTrackView: document.getElementById("citizen-track-view"),
        citizenAboutView: document.getElementById("citizen-about-view"),
        myReportsCountBadge: document.getElementById("my-reports-count"),

        // Citizen Step 1 (Image)
        inputCamera: document.getElementById("input-camera"),
        inputUpload: document.getElementById("input-upload"),
        dropZone: document.getElementById("drop-zone"),
        imagePreviewBox: document.getElementById("image-preview-box"),
        imagePreview: document.getElementById("image-preview"),
        btnClearImg: document.getElementById("btn-clear-img"),
        previewMetaBadge: document.getElementById("preview-meta-badge"),

        // Citizen Step 2 (GPS Location)
        gpsInitialBox: document.getElementById("gps-initial-box"),
        btnGetGps: document.getElementById("btn-get-gps"),
        gpsBtnText: document.getElementById("gps-btn-text"),
        gpsStatusMsg: document.getElementById("gps-status-msg"),
        gpsConfirmedPanel: document.getElementById("gps-confirmed-panel"),
        btnReacquireGps: document.getElementById("btn-reacquire-gps"),
        gpsLat: document.getElementById("gps-lat"),
        gpsLon: document.getElementById("gps-lon"),
        gpsAcc: document.getElementById("gps-acc"),
        gpsTime: document.getElementById("gps-time"),


        // Citizen Step 3 (Analyze)
        btnAnalyze: document.getElementById("btn-analyze"),
        analyzeHint: document.getElementById("analyze-hint"),
        analysisLoading: document.getElementById("analysis-loading"),
        loadingStageText: document.getElementById("loading-stage-text"),

        // Citizen Step 4 (Results)
        resultSection: document.getElementById("result-section"),
        resOriginalImg: document.getElementById("res-original-img"),
        resAnnotatedImg: document.getElementById("res-annotated-img"),
        resPhsVal: document.getElementById("res-phs-val"),
        resConditionBadge: document.getElementById("res-condition-badge"),
        resPriorityLabel: document.getElementById("res-priority-label"),
        resTotalDefects: document.getElementById("res-total-defects"),
        resPotholeCount: document.getElementById("res-pothole-count"),
        resCrackCount: document.getElementById("res-crack-count"),
        resDetectionsList: document.getElementById("res-detections-list"),
        btnOpenGmaps: document.getElementById("btn-open-gmaps"),
        resLat: document.getElementById("res-lat"),
        resLon: document.getElementById("res-lon"),
        resAcc: document.getElementById("res-acc"),

        // Citizen Step 5 (Submit Complaint)
        cardSubmitComplaint: document.getElementById("card-submit-complaint"),
        citizenInputPhone: document.getElementById("citizen-input-phone"),
        citizenInputDesc: document.getElementById("citizen-input-desc"),
        btnSubmitComplaint: document.getElementById("btn-submit-complaint"),
        submissionSuccessBox: document.getElementById("submission-success-box"),
        successComplaintId: document.getElementById("success-complaint-id"),
        btnViewMyComplaint: document.getElementById("btn-view-my-complaint"),
        btnNewReport: document.getElementById("btn-new-report"),

        // Citizen Track / Reports
        trackSearchInput: document.getElementById("track-search-input"),
        btnTrackSearch: document.getElementById("btn-track-search"),
        trackedComplaintResult: document.getElementById("tracked-complaint-result"),
        myReportsContainer: document.getElementById("my-reports-container"),

        // Officer Login
        officerLoginCard: document.getElementById("officer-login-card"),
        officerAuthenticatedView: document.getElementById("officer-authenticated-view"),
        formOfficerLogin: document.getElementById("form-officer-login"),
        loginEmail: document.getElementById("login-email"),
        loginPassword: document.getElementById("login-password"),
        loginErrorMsg: document.getElementById("login-error-msg"),
        btnQuickFillDemo: document.getElementById("btn-quick-fill-demo"),

        // Officer Sidebar & Tabs
        officerSidebarName: document.getElementById("officer-sidebar-name"),
        sidebarNotifBadge: document.getElementById("sidebar-notif-badge"),
        btnSidebarSignout: document.getElementById("btn-sidebar-signout"),
        snavBtns: document.querySelectorAll(".snav-btn"),
        officerTabContents: document.querySelectorAll(".officer-tab-content"),

        // Officer Dashboard
        btnRefreshDash: document.getElementById("btn-refresh-dash"),
        kpiTotal: document.getElementById("kpi-total"),
        kpiNew: document.getElementById("kpi-new"),
        kpiReview: document.getElementById("kpi-review"),
        kpiCritical: document.getElementById("kpi-critical"),
        kpiResolved: document.getElementById("kpi-resolved"),
        dashPriorityBanner: document.getElementById("dash-priority-banner"),
        dashPriorityText: document.getElementById("dash-priority-text"),
        tableRecentComplaints: document.getElementById("table-recent-complaints").querySelector("tbody"),
        btnGotoAllComplaints: document.getElementById("btn-goto-all-complaints"),

        // Officer Complaints List
        filterSearchInput: document.getElementById("filter-search-input"),
        filterStatusSelect: document.getElementById("filter-status-select"),
        filterConditionSelect: document.getElementById("filter-condition-select"),
        filterSortSelect: document.getElementById("filter-sort-select"),
        tableAllComplaints: document.getElementById("table-all-complaints").querySelector("tbody"),

        // Officer Notifications
        btnMarkAllRead: document.getElementById("btn-mark-all-read"),
        officerNotificationsContainer: document.getElementById("officer-notifications-container"),

        // Officer Analytics
        analyticsDefectBars: document.getElementById("analytics-defect-bars"),
        analyticsConditionBars: document.getElementById("analytics-condition-bars"),

        // Complaint Detail Modal
        modalComplaintDetail: document.getElementById("modal-complaint-detail"),
        btnCloseDetailModal: document.getElementById("btn-close-detail-modal"),
        modalCid: document.getElementById("modal-cid"),
        modalStatusBadge: document.getElementById("modal-status-badge"),
        modalPriorityBadge: document.getElementById("modal-priority-badge"),
        modalCreatedTime: document.getElementById("modal-created-time"),
        modalOriginalImg: document.getElementById("modal-original-img"),
        modalAnnotatedImg: document.getElementById("modal-annotated-img"),
        modalPhsScore: document.getElementById("modal-phs-score"),
        modalConditionBadge: document.getElementById("modal-condition-badge"),
        modalPotholesCount: document.getElementById("modal-potholes-count"),
        modalCracksCount: document.getElementById("modal-cracks-count"),
        modalTotalCount: document.getElementById("modal-total-count"),
        modalDetectionsList: document.getElementById("modal-detections-list"),
        modalGmapsLink: document.getElementById("modal-gmaps-link"),
        modalLocText: document.getElementById("modal-loc-text"),
        modalCitizenDesc: document.getElementById("modal-citizen-desc"),
        modalCitizenPhone: document.getElementById("modal-citizen-phone"),
        modalTimelineContainer: document.getElementById("modal-timeline-container"),
        modalOfficerControls: document.getElementById("modal-officer-controls"),
        modalOfficerNotes: document.getElementById("modal-officer-notes"),
        btnDeleteComplaint: document.getElementById("btn-delete-complaint"),
    };

    // ═══════════════════════════════════════════════════════════════════
    // 1. INITIALIZATION & PORTAL SWITCHING
    // ═══════════════════════════════════════════════════════════════════

    function init() {
        setupEventListeners();
        updateMyReportsBadge();

        // Check if officer was logged in
        if (state.officerToken && state.officerUser) {
            showOfficerSession();
        }

        // Check URL hash or query params
        const params = new URLSearchParams(window.location.search);
        if (params.get("portal") === "officer" || window.location.hash.startsWith("#officer")) {
            switchPortal("officer");
        } else {
            switchPortal("citizen");
        }

        // Start background polling for notifications
        startNotificationPolling();
    }

    function switchPortal(portal) {
        state.activePortal = portal;
        if (portal === "citizen") {
            el.btnPortalCitizen.classList.add("active");
            el.btnPortalOfficer.classList.remove("active");
            el.viewCitizen.style.display = "block";
            el.viewOfficer.style.display = "none";
        } else {
            el.btnPortalCitizen.classList.remove("active");
            el.btnPortalOfficer.classList.add("active");
            el.viewCitizen.style.display = "none";
            el.viewOfficer.style.display = "block";

            if (state.officerToken) {
                el.officerLoginCard.style.display = "none";
                el.officerAuthenticatedView.style.display = "grid";
                loadOfficerDashboard();
            } else {
                el.officerLoginCard.style.display = "flex";
                el.officerAuthenticatedView.style.display = "none";
            }
        }
    }

    function switchCitizenTab(tab) {
        state.activeCitizenTab = tab;
        el.cnavReport.classList.toggle("active", tab === "report");
        el.cnavTrack.classList.toggle("active", tab === "track");
        el.cnavAbout.classList.toggle("active", tab === "about");

        el.citizenReportView.style.display = tab === "report" ? "block" : "none";
        el.citizenTrackView.style.display = tab === "track" ? "block" : "none";
        el.citizenAboutView.style.display = tab === "about" ? "block" : "none";

        if (tab === "track") {
            renderMyReportsList();
        }
    }

    function switchOfficerTab(tabId) {
        state.activeOfficerTab = tabId.replace("tab-", "");
        el.snavBtns.forEach(btn => {
            btn.classList.toggle("active", btn.dataset.target === tabId);
        });
        el.officerTabContents.forEach(content => {
            content.style.display = content.id === tabId ? "block" : "none";
        });

        if (tabId === "tab-dash") loadOfficerDashboard();
        if (tabId === "tab-complaints") loadAllComplaints();
        if (tabId === "tab-map") loadOfficerMap();
        if (tabId === "tab-analytics") loadAnalytics();
        if (tabId === "tab-notifs") loadNotificationsList();
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2. CITIZEN IMAGE HANDLING
    // ═══════════════════════════════════════════════════════════════════

    function handleFileSelection(file) {
        if (!file) return;

        const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
        if (!allowed.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp|bmp)$/i)) {
            alert("Please select a valid road photograph (JPG, PNG, or WEBP).");
            return;
        }

        if (file.size > 20 * 1024 * 1024) {
            alert("File size exceeds 20MB limit.");
            return;
        }

        state.selectedFile = file;

        // Display preview
        const reader = new FileReader();
        reader.onload = (e) => {
            el.imagePreview.src = e.target.result;
            el.imagePreviewBox.style.display = "block";
            el.dropZone.style.display = "none";
            const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
            el.previewMetaBadge.textContent = `${file.name} (${sizeMb} MB)`;
            validateAnalyzeButtonState();
        };
        reader.readAsDataURL(file);
    }

    function clearSelectedImage() {
        state.selectedFile = null;
        el.inputCamera.value = "";
        el.inputUpload.value = "";
        el.imagePreview.src = "";
        el.imagePreviewBox.style.display = "none";
        el.dropZone.style.display = "block";
        validateAnalyzeButtonState();
    }

    // ═══════════════════════════════════════════════════════════════════
    // ═══════════════════════════════════════════════════════════════════
    // 3. GPS GEOLOCATION & ROAD LOCATION PINPOINTING
    // ═══════════════════════════════════════════════════════════════════

    function acquireLocation() {
        if (!navigator.geolocation) {
            alert("Geolocation is not supported by your browser.");
            return;
        }

        if (el.gpsBtnText) el.gpsBtnText.textContent = "⏳ Acquiring device location...";
        if (el.gpsStatusMsg) el.gpsStatusMsg.innerHTML = "Requesting device GPS coordinates. Please allow location access...";
        if (el.btnGetGps) el.btnGetGps.disabled = true;

        const highAccuracyOptions = {
            enableHighAccuracy: true,
            timeout: 12000,
            maximumAge: 0,
        };

        const standardAccuracyOptions = {
            enableHighAccuracy: false,
            timeout: 10000,
            maximumAge: 60000,
        };

        // Try high-accuracy first; fallback to standard accuracy if needed
        navigator.geolocation.getCurrentPosition(
            (pos) => onLocationSuccess(pos),
            (err) => {
                console.warn("High-accuracy location failed or timed out, trying standard accuracy...", err);
                navigator.geolocation.getCurrentPosition(
                    (pos) => onLocationSuccess(pos),
                    (finalErr) => onLocationError(finalErr),
                    standardAccuracyOptions
                );
            },
            highAccuracyOptions
        );
    }

    function onLocationSuccess(pos) {
        const coords = pos.coords;
        const lat = coords.latitude;
        const lon = coords.longitude;
        const acc = coords.accuracy ? Math.round(coords.accuracy * 10) / 10 : 10.0;
        const timeStr = new Date(pos.timestamp).toLocaleTimeString();
        const isoTime = new Date(pos.timestamp).toISOString();

        state.gpsData = {
            latitude: lat,
            longitude: lon,
            accuracy: acc,
            altitude: coords.altitude || null,
            heading: coords.heading || null,
            speed: coords.speed || null,
            timestamp: isoTime,
            displayTime: timeStr,
        };

        // Switch panels to confirmed view
        if (el.gpsInitialBox) el.gpsInitialBox.style.display = "none";
        if (el.gpsConfirmedPanel) el.gpsConfirmedPanel.style.display = "block";

        if (el.gpsLat) el.gpsLat.textContent = lat.toFixed(6);
        if (el.gpsLon) el.gpsLon.textContent = lon.toFixed(6);
        if (el.gpsAcc) el.gpsAcc.textContent = `±${acc} meters`;
        if (el.gpsTime) el.gpsTime.textContent = timeStr;

        if (el.gpsBtnText) el.gpsBtnText.textContent = "📍 Get My Current Location";
        if (el.btnGetGps) el.btnGetGps.disabled = false;

        // Render interactive Leaflet preview map with draggable marker
        renderGpsPreviewMap(lat, lon, acc);

        validateAnalyzeButtonState();
    }

    function onLocationError(err) {
        if (el.btnGetGps) el.btnGetGps.disabled = false;
        if (el.gpsBtnText) el.gpsBtnText.textContent = "📍 Retry Location";

        let msg = "Could not obtain location.";
        if (err.code === 1) msg = "Location permission denied. Please allow location access in your browser settings.";
        else if (err.code === 2) msg = "Location unavailable. Please check if your device location / GPS is turned on.";
        else if (err.code === 3) msg = "Location request timed out. Please click Retry Location.";

        if (el.gpsStatusMsg) el.gpsStatusMsg.innerHTML = `<span style="color:#ef4444;">⚠️ ${msg}</span>`;
        alert(`Location Notice: ${msg}`);
    }

    function renderGpsPreviewMap(lat, lon, accuracy) {
        const mapContainer = document.getElementById("gps-live-leaflet-map");
        if (!mapContainer) return;

        if (!state.gpsLiveMap) {
            state.gpsLiveMap = L.map("gps-live-leaflet-map", {
                zoomControl: true,
                attributionControl: false,
            }).setView([lat, lon], 16);

            L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                maxZoom: 19,
            }).addTo(state.gpsLiveMap);

            // Create draggable marker
            state.gpsLiveMarker = L.marker([lat, lon], {
                draggable: true,
                title: "Drag to fine-tune exact defect spot",
            }).addTo(state.gpsLiveMap);

            // Bind drag events so user can adjust location if needed
            state.gpsLiveMarker.on("dragend", function (e) {
                const newPos = e.target.getLatLng();
                updateCoordinatesManually(newPos.lat, newPos.lng);
            });

            // Also click on map to reposition pin
            state.gpsLiveMap.on("click", function (e) {
                state.gpsLiveMarker.setLatLng(e.latlng);
                updateCoordinatesManually(e.latlng.lat, e.latlng.lng);
            });
        } else {
            state.gpsLiveMap.setView([lat, lon], 16);
            if (state.gpsLiveMarker) {
                state.gpsLiveMarker.setLatLng([lat, lon]);
            }
        }

        setTimeout(() => {
            if (state.gpsLiveMap) state.gpsLiveMap.invalidateSize();
        }, 300);
    }

    function updateCoordinatesManually(newLat, newLon) {
        if (!state.gpsData) {
            state.gpsData = {
                timestamp: new Date().toISOString(),
                displayTime: new Date().toLocaleTimeString(),
                accuracy: 5.0,
            };
        }
        state.gpsData.latitude = newLat;
        state.gpsData.longitude = newLon;
        if (el.gpsLat) el.gpsLat.textContent = newLat.toFixed(6);
        if (el.gpsLon) el.gpsLon.textContent = newLon.toFixed(6);
        validateAnalyzeButtonState();
    }

    function reacquireLocation() {
        if (el.gpsConfirmedPanel) el.gpsConfirmedPanel.style.display = "none";
        if (el.gpsInitialBox) el.gpsInitialBox.style.display = "block";
        acquireLocation();
    }

    function validateAnalyzeButtonState() {
        const hasImage = !!state.selectedFile;
        const hasGps = !!state.gpsData;

        if (hasImage && hasGps) {
            el.btnAnalyze.disabled = false;
            el.analyzeHint.textContent = `✓ Road photo selected & Location captured (±${state.gpsData.accuracy}m). Click 'Analyze Road' to run YOLO11m-P2P3-CBAM.`;
            el.analyzeHint.style.color = "var(--cyan-primary)";
        } else if (!hasImage && hasGps) {
            el.btnAnalyze.disabled = true;
            el.analyzeHint.textContent = "Please take or upload a road photograph.";
            el.analyzeHint.style.color = "var(--text-muted)";
        } else if (hasImage && !hasGps) {
            el.btnAnalyze.disabled = true;
            el.analyzeHint.textContent = "Please capture your road location above.";
            el.analyzeHint.style.color = "var(--text-muted)";
        } else {
            el.btnAnalyze.disabled = true;
            el.analyzeHint.textContent = "Please select a road image and capture your location to proceed.";
            el.analyzeHint.style.color = "var(--text-muted)";
        }
    }


    // ═══════════════════════════════════════════════════════════════════
    // 4. AI ROAD ANALYSIS EXECUTION
    // ═══════════════════════════════════════════════════════════════════

    async function runAiRoadAnalysis() {
        if (!state.selectedFile || !state.gpsData) return;

        el.btnAnalyze.disabled = true;
        el.analysisLoading.style.display = "flex";
        el.loadingStageText.textContent = "Initializing YOLO11m-P2P3-CBAM pipeline...";

        // Staged progress messages for feedback
        const stages = [
            "Running multi-pass tiling and CLAHE contrast enhancement...",
            "Applying P2→P3 Detail feature fusion and CBAM Attention...",
            "Performing morphological black-hat crack refinement & physical merging...",
            "Calculating Road Health Score (RHS) & GIS coordinates...",
        ];
        let stageIdx = 0;
        const stageInterval = setInterval(() => {
            if (stageIdx < stages.length) {
                el.loadingStageText.textContent = stages[stageIdx++];
            }
        }, 800);

        const formData = new FormData();
        formData.append("image", state.selectedFile);
        formData.append("latitude", state.gpsData.latitude);
        formData.append("longitude", state.gpsData.longitude);
        formData.append("accuracy", state.gpsData.accuracy);
        formData.append("timestamp", state.gpsData.timestamp);

        try {
            const resp = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });

            clearInterval(stageInterval);

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || `Analysis failed with status ${resp.status}`);
            }

            const data = await resp.json();
            state.analysisResult = data;

            el.analysisLoading.style.display = "none";
            renderAnalysisResults(data);
        } catch (err) {
            clearInterval(stageInterval);
            el.analysisLoading.style.display = "none";
            el.btnAnalyze.disabled = false;
            alert(`Analysis Error: ${err.message}`);
        }
    }

    function renderAnalysisResults(data) {
        const phs = data.summary;
        const loc = data.location;
        const detections = data.detections || [];

        // 1. Dual Images (Original & Annotated)
        el.resOriginalImg.src = data.original_image;
        el.resAnnotatedImg.src = data.result_image;

        // 2. PHS Gauge
        const phsScore = phs.phs != null ? Math.round(phs.phs * 10) / 10 : 100;
        el.resPhsVal.textContent = phsScore;
        const cond = (phs.condition || "GOOD").toUpperCase();
        el.resConditionBadge.textContent = cond;
        el.resConditionBadge.className = `condition-badge ${cond.toLowerCase()}`;

        // Compute AI Priority indication
        let priority = "NORMAL";
        if (phsScore < 40 || (phs.pothole_count || 0) >= 4) priority = "CRITICAL 🔴";
        else if (phsScore < 60 || (phs.pothole_count || 0) >= 2 || (phs.crack_count || 0) >= 4) priority = "HIGH 🟠";
        else if (phsScore < 75 || (phs.total_defects || 0) >= 2) priority = "MEDIUM 🟡";
        else priority = "LOW 🟢";
        el.resPriorityLabel.textContent = priority;

        // 3. Defect Counters
        let pCount = phs.pothole_count != null ? phs.pothole_count : (phs.detected_potholes || 0);
        let cCount = phs.crack_count != null ? phs.crack_count : (phs.detected_cracks || 0);
        if (pCount === 0 && cCount === 0 && detections.length > 0) {
            pCount = detections.filter(d => (d.class_name || "").toLowerCase().includes("pothole")).length;
            cCount = detections.filter(d => (d.class_name || "").toLowerCase().includes("crack")).length;
        }
        el.resTotalDefects.textContent = phs.total_defects || (pCount + cCount);
        el.resPotholeCount.textContent = pCount;
        el.resCrackCount.textContent = cCount;

        // 4. Itemized Detections List
        el.resDetectionsList.innerHTML = "";
        if (detections.length === 0) {
            el.resDetectionsList.innerHTML = `<div class="det-item-row" style="color:var(--text-dim);">No significant road defects detected on surface.</div>`;
        } else {
            detections.forEach((d, idx) => {
                const row = document.createElement("div");
                row.className = "det-item-row";
                const confPct = Math.round(d.confidence * 100);
                const sevPct = d.severity_score != null ? ` • Severity: ${Math.round(d.severity_score * 100)}%` : "";
                row.innerHTML = `
                    <span class="det-class-tag ${(d.class_name || '').toLowerCase()}">#${idx + 1} ${d.class_name}</span>
                    <span class="det-conf">${confPct}% conf${sevPct}</span>
                `;
                el.resDetectionsList.appendChild(row);
            });
        }

        // 5. Location & Map
        el.resLat.textContent = loc.latitude.toFixed(6);
        el.resLon.textContent = loc.longitude.toFixed(6);
        el.resAcc.textContent = loc.accuracy ? `±${loc.accuracy} m` : "—";
        el.btnOpenGmaps.href = loc.google_maps_url || `https://www.google.com/maps?q=${loc.latitude},${loc.longitude}`;

        // Initialize / Update Citizen Leaflet Map with Accuracy Circle
        initOrUpdateCitizenMap(loc.latitude, loc.longitude, cond, phsScore, loc.accuracy);

        // Show Results Section & Scroll
        el.resultSection.style.display = "flex";
        el.cardSubmitComplaint.style.display = "block";
        el.submissionSuccessBox.style.display = "none";

        el.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function initOrUpdateCitizenMap(lat, lon, condition, phsScore, accuracy) {
        if (!state.citizenMap) {
            state.citizenMap = L.map("citizen-leaflet-map", {
                zoomControl: true,
                attributionControl: false,
            }).setView([lat, lon], 16);

            L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                maxZoom: 19,
            }).addTo(state.citizenMap);
        } else {
            state.citizenMap.setView([lat, lon], 16);
            if (state.citizenMarker) state.citizenMap.removeLayer(state.citizenMarker);
            if (state.citizenCircle) state.citizenMap.removeLayer(state.citizenCircle);
        }

        let markerColor = "#10b981";
        if (condition === "CRITICAL") markerColor = "#ef4444";
        else if (condition === "POOR") markerColor = "#f97316";
        else if (condition === "MODERATE") markerColor = "#eab308";

        // Accuracy uncertainty circle
        if (accuracy) {
            state.citizenCircle = L.circle([lat, lon], {
                radius: accuracy,
                color: markerColor,
                fillColor: markerColor,
                fillOpacity: 0.15,
                weight: 1.5,
            }).addTo(state.citizenMap);
        }

        state.citizenMarker = L.circleMarker([lat, lon], {
            radius: 10,
            fillColor: markerColor,
            color: "#ffffff",
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9,
        }).addTo(state.citizenMap);

        state.citizenMarker.bindPopup(`<b>Reported Road Defect</b><br>RHS: ${phsScore} (${condition})<br>Accuracy: ±${accuracy || '--'}m`).openPopup();
        setTimeout(() => state.citizenMap.invalidateSize(), 300);
    }

    // ═══════════════════════════════════════════════════════════════════
    // 5. COMPLAINT SUBMISSION (CITIZEN)
    // ═══════════════════════════════════════════════════════════════════

    async function submitCitizenComplaint() {
        if (!state.analysisResult || !state.gpsData) return;

        el.btnSubmitComplaint.disabled = true;
        el.btnSubmitComplaint.innerHTML = `
            <div class="spinner" style="border-top-color:#fff; width:18px; height:18px;"></div>
            Submitting Complaint...
        `;

        const payload = {
            original_image: state.analysisResult.original_image,
            annotated_image: state.analysisResult.result_image,
            latitude: state.gpsData.latitude,
            longitude: state.gpsData.longitude,
            gps_accuracy: state.gpsData.accuracy,
            altitude: state.gpsData.altitude,
            heading: state.gpsData.heading,
            speed: state.gpsData.speed,
            timestamp: state.gpsData.timestamp,
            phs_data: state.analysisResult.summary,
            detections: state.analysisResult.detections,
            description: el.citizenInputDesc.value.trim(),
            citizen_phone: el.citizenInputPhone.value.trim(),
        };

        try {
            const resp = await fetch("/api/complaints", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            el.btnSubmitComplaint.disabled = false;
            el.btnSubmitComplaint.innerHTML = `
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                Submit Complaint
            `;

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || "Failed to submit complaint.");
            }

            const data = await resp.json();
            const complaint = data.complaint;
            state.submittedComplaintId = complaint.complaint_id;

            // Save to LocalStorage for "My Reports"
            saveReportToLocalStorage(complaint);

            // Show Confirmation View
            el.cardSubmitComplaint.style.display = "none";
            el.successComplaintId.textContent = complaint.complaint_id;
            el.submissionSuccessBox.style.display = "block";
            el.submissionSuccessBox.scrollIntoView({ behavior: "smooth", block: "center" });

            // Refresh notification polling immediately
            fetchNotificationsCount();
        } catch (err) {
            el.btnSubmitComplaint.disabled = false;
            el.btnSubmitComplaint.innerHTML = `Submit Complaint`;
            alert(`Submission Failed: ${err.message}`);
        }
    }

    function saveReportToLocalStorage(complaint) {
        try {
            const list = JSON.parse(localStorage.getItem("roadcare_my_reports") || "[]");
            list.unshift({
                complaint_id: complaint.complaint_id,
                created_at: complaint.created_at,
                condition: complaint.condition,
                phs: complaint.phs,
                defects: complaint.total_defects,
                status: complaint.status,
                image: complaint.original_image,
            });
            localStorage.setItem("roadcare_my_reports", JSON.stringify(list.slice(0, 30)));
            updateMyReportsBadge();
        } catch (e) {
            console.error("LocalStorage error:", e);
        }
    }

    function updateMyReportsBadge() {
        try {
            const list = JSON.parse(localStorage.getItem("roadcare_my_reports") || "[]");
            if (list.length > 0) {
                el.myReportsCountBadge.textContent = list.length;
                el.myReportsCountBadge.style.display = "inline-block";
            } else {
                el.myReportsCountBadge.style.display = "none";
            }
        } catch (e) {}
    }

    // ═══════════════════════════════════════════════════════════════════
    // 6. TRACK COMPLAINT / MY REPORTS (CITIZEN)
    // ═══════════════════════════════════════════════════════════════════

    async function searchAndTrackComplaint(cid) {
        const queryId = (cid || el.trackSearchInput.value).trim().toUpperCase();
        if (!queryId) return;

        el.trackedComplaintResult.innerHTML = `
            <div style="text-align:center; padding:2rem; color:var(--text-muted);">
                <div class="spinner" style="margin:0 auto 1rem;"></div>
                Looking up complaint ${queryId}...
            </div>
        `;
        el.trackedComplaintResult.style.display = "block";

        try {
            const resp = await fetch(`/api/complaints/${queryId}`);
            if (!resp.ok) {
                el.trackedComplaintResult.innerHTML = `
                    <div class="dash-card" style="text-align:center; border-color:var(--rose-danger);">
                        <p style="color:#f87171; font-weight:700;">Complaint '${queryId}' not found.</p>
                        <p class="text-sm" style="color:var(--text-dim);">Please check the Complaint ID and try again.</p>
                    </div>
                `;
                return;
            }

            const data = await resp.json();
            renderTrackedComplaintCard(data.complaint);
        } catch (err) {
            el.trackedComplaintResult.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
        }
    }

    function renderTrackedComplaintCard(c) {
        const statusClass = (c.status || "new").toLowerCase().replace(/\s+/g, "-");
        const condClass = (c.condition || "good").toLowerCase();

        let timelineHtml = "";
        (c.timeline || []).forEach(step => {
            timelineHtml += `
                <div class="timeline-step">
                    <div class="timeline-dot"></div>
                    <div class="timeline-content">
                        <div class="timeline-header">
                            <span>${step.event}</span>
                            <span class="timeline-time">${step.timestamp}</span>
                        </div>
                        <div class="timeline-actor">By: ${step.actor}</div>
                        ${step.notes ? `<div class="timeline-notes">${step.notes}</div>` : ""}
                    </div>
                </div>
            `;
        });

        el.trackedComplaintResult.innerHTML = `
            <div class="dash-card" style="border-color:rgba(6, 182, 212, 0.4);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:1rem;">
                    <div>
                        <span class="complaint-id-tag" style="font-size:1.3rem;">${c.complaint_id}</span>
                        <p class="text-sm" style="color:var(--text-muted);">Reported on ${c.created_at}</p>
                    </div>
                    <div>
                        <span class="status-badge ${statusClass}">${c.status}</span>
                        <span class="condition-badge ${condClass}">${c.condition} (RHS: ${c.phs})</span>
                    </div>
                </div>

                <div class="modal-images-grid" style="margin-bottom:1rem;">
                    <div class="modal-img-box">
                        <div class="modal-img-lbl">CITIZEN PHOTO</div>
                        <div class="img-frame"><img src="${c.original_image}" alt="Original"></div>
                    </div>
                    <div class="modal-img-box highlight">
                        <div class="modal-img-lbl">AI DEFECT ANALYSIS</div>
                        <div class="img-frame"><img src="${c.annotated_image}" alt="Annotated"></div>
                    </div>
                </div>

                <h4 style="font-size:0.85rem; text-transform:uppercase; color:var(--text-dim); margin-bottom:0.5rem;">Status Audit Timeline</h4>
                <div class="timeline-stepper">
                    ${timelineHtml}
                </div>
            </div>
        `;
    }

    function renderMyReportsList() {
        try {
            const list = JSON.parse(localStorage.getItem("roadcare_my_reports") || "[]");
            if (list.length === 0) {
                el.myReportsContainer.innerHTML = `
                    <p class="empty-state">No complaints submitted from this browser yet. Use the 'Report Road' tab to submit your first report.</p>
                `;
                return;
            }

            el.myReportsContainer.innerHTML = "";
            list.forEach(item => {
                const card = document.createElement("div");
                card.className = "report-item-card";
                const statusClass = (item.status || "new").toLowerCase().replace(/\s+/g, "-");
                card.innerHTML = `
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <span class="complaint-id-tag" style="font-size:0.95rem;">${item.complaint_id}</span>
                        <span class="status-badge ${statusClass}">${item.status}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:1rem;">
                        <span class="text-sm" style="color:var(--text-dim);">${item.created_at.split(' ')[0]}</span>
                        <button class="btn-table-action">View Details →</button>
                    </div>
                `;
                card.addEventListener("click", () => {
                    el.trackSearchInput.value = item.complaint_id;
                    searchAndTrackComplaint(item.complaint_id);
                });
                el.myReportsContainer.appendChild(card);
            });
        } catch (e) {}
    }

    // ═══════════════════════════════════════════════════════════════════
    // 7. OFFICER AUTHENTICATION & LOGIN
    // ═══════════════════════════════════════════════════════════════════

    async function handleOfficerLogin(e) {
        e.preventDefault();
        const email = el.loginEmail.value.trim();
        const password = el.loginPassword.value.trim();

        el.loginErrorMsg.style.display = "none";

        try {
            const resp = await fetch("/api/officer/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password }),
            });

            const data = await resp.json();
            if (!resp.ok || !data.success) {
                throw new Error(data.detail || "Authentication failed.");
            }

            // Save Token & User in storage (survives browser refresh)
            state.officerToken = data.token;
            state.officerUser = data.officer;
            localStorage.setItem("roadcare_officer_token", data.token);
            localStorage.setItem("roadcare_officer_user", JSON.stringify(data.officer));
            sessionStorage.setItem("roadcare_officer_token", data.token);
            sessionStorage.setItem("roadcare_officer_user", JSON.stringify(data.officer));

            showOfficerSession();
            switchPortal("officer");
        } catch (err) {
            el.loginErrorMsg.textContent = `❌ ${err.message}`;
            el.loginErrorMsg.style.display = "block";
        }
    }

    function showOfficerSession() {
        if (state.officerUser) {
            el.navOfficerName.textContent = state.officerUser.name || "Officer";
            el.officerSidebarName.textContent = state.officerUser.name || "Chief Road Officer";
            el.officerSessionPill.style.display = "flex";
        }
    }

    function handleOfficerLogout() {
        state.officerToken = null;
        state.officerUser = null;
        localStorage.removeItem("roadcare_officer_token");
        localStorage.removeItem("roadcare_officer_user");
        sessionStorage.removeItem("roadcare_officer_token");
        sessionStorage.removeItem("roadcare_officer_user");
        el.officerSessionPill.style.display = "none";
        switchPortal("officer");
    }

    // ═══════════════════════════════════════════════════════════════════
    // 8. OFFICER DASHBOARD & COMPLAINTS MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════

    async function loadOfficerDashboard() {
        try {
            const resp = await fetch("/api/dashboard/stats");
            const data = await resp.json();
            if (!data.success) return;

            const stats = data.stats;
            el.kpiTotal.textContent = stats.total_complaints;
            el.kpiNew.textContent = stats.new_complaints;
            el.kpiReview.textContent = stats.under_review;
            el.kpiCritical.textContent = stats.critical_roads;
            el.kpiResolved.textContent = stats.resolved;

            // Critical alerts banner
            if (stats.critical_alerts && stats.critical_alerts.length > 0) {
                const alertCount = stats.critical_alerts.length;
                el.dashPriorityText.textContent = `${alertCount} critical complaint(s) with severe road damage (RHS < 40) require urgent road crew inspection.`;
                el.dashPriorityBanner.style.display = "flex";
            } else {
                el.dashPriorityBanner.style.display = "none";
            }

            // Load Recent Complaints Table
            loadRecentComplaintsTable();
        } catch (err) {
            console.error("Dashboard stats error:", err);
        }
    }

    async function loadRecentComplaintsTable() {
        try {
            const resp = await fetch("/api/complaints?limit=6&sort_by=newest");
            const data = await resp.json();
            if (!data.success) return;

            renderComplaintsTable(data.complaints, el.tableRecentComplaints);
        } catch (err) {
            console.error("Recent complaints error:", err);
        }
    }

    async function loadAllComplaints() {
        const status = el.filterStatusSelect.value;
        const condition = el.filterConditionSelect.value;
        const sort_by = el.filterSortSelect.value;
        const search = el.filterSearchInput.value.trim();

        const params = new URLSearchParams({
            status,
            condition,
            sort_by,
            limit: "100",
        });
        if (search) params.set("search", search);

        try {
            const resp = await fetch(`/api/complaints?${params.toString()}`);
            const data = await resp.json();
            if (!data.success) return;

            renderComplaintsTable(data.complaints, el.tableAllComplaints);
        } catch (err) {
            console.error("Complaints list error:", err);
        }
    }

    function renderComplaintsTable(complaints, tbodyElement) {
        tbodyElement.innerHTML = "";

        if (!complaints || complaints.length === 0) {
            tbodyElement.innerHTML = `<tr><td colspan="9" class="empty-state">No complaints matching filter criteria.</td></tr>`;
            return;
        }

        complaints.forEach(c => {
            const tr = document.createElement("tr");

            const statusClass = (c.status || "new").toLowerCase().replace(/\s+/g, "-");
            const condClass = (c.condition || "good").toLowerCase();
            const priorityClass = (c.priority || "normal").toLowerCase();

            let pCount = c.pothole_count != null ? c.pothole_count : 0;
            let cCount = c.crack_count != null ? c.crack_count : 0;
            if (pCount === 0 && cCount === 0 && Array.isArray(c.detections) && c.detections.length > 0) {
                pCount = c.detections.filter(d => (d.class_name || "").toLowerCase().includes("pothole")).length;
                cCount = c.detections.filter(d => (d.class_name || "").toLowerCase().includes("crack")).length;
            }
            const defectStr = `${pCount} Pothole${pCount === 1 ? "" : "s"}, ${cCount} Crack${cCount === 1 ? "" : "s"}`;
            const phsScore = c.phs != null ? Number(c.phs).toFixed(1) : "--";

            tr.innerHTML = `
                <td><strong class="complaint-id-tag" style="font-size:0.85rem;">${c.complaint_id}</strong></td>
                <td><img src="${c.original_image}" alt="Thumb" class="table-thumb" onerror="this.src='/uploads/placeholder.jpg'"></td>
                <td>${defectStr}</td>
                <td><strong>${phsScore}</strong></td>
                <td><span class="condition-badge ${condClass}" style="font-size:0.68rem; padding:1px 6px;">${c.condition}</span></td>
                <td><span class="priority-badge ${priorityClass}">${c.priority}</span></td>
                <td><span class="status-badge ${statusClass}">${c.status}</span></td>
                <td style="font-size:0.75rem; color:var(--text-dim);">${(c.created_at || "").split(' ')[0]}</td>
                <td>
                    <div style="display:flex; gap:6px; align-items:center;">
                        <button class="btn-table-action" data-cid="${c.complaint_id}">Inspect</button>
                        <button class="btn-table-delete" data-cid="${c.complaint_id}" title="Remove / Delete Complaint">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                        </button>
                    </div>
                </td>
            `;

            tr.querySelector(".btn-table-action").addEventListener("click", () => {
                openComplaintDetailModal(c.complaint_id);
            });

            tr.querySelector(".btn-table-delete").addEventListener("click", (e) => {
                e.stopPropagation();
                deleteComplaintAsOfficer(c.complaint_id);
            });

            tbodyElement.appendChild(tr);
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // 9. COMPLAINT DETAIL MODAL & OFFICER ACTIONS
    // ═══════════════════════════════════════════════════════════════════

    async function openComplaintDetailModal(cid) {
        try {
            const resp = await fetch(`/api/complaints/${cid}`);
            if (!resp.ok) return;

            const data = await resp.json();
            const c = data.complaint;

            // Fill header
            el.modalCid.textContent = c.complaint_id;
            const statusClass = (c.status || "new").toLowerCase().replace(/\s+/g, "-");
            el.modalStatusBadge.textContent = c.status;
            el.modalStatusBadge.className = `status-badge ${statusClass}`;

            const priorityClass = (c.priority || "normal").toLowerCase();
            el.modalPriorityBadge.textContent = `AI Priority: ${c.priority}`;
            el.modalPriorityBadge.className = `priority-badge ${priorityClass}`;

            el.modalCreatedTime.textContent = `Reported: ${c.created_at}`;

            // Images
            el.modalOriginalImg.src = c.original_image;
            el.modalAnnotatedImg.src = c.annotated_image;

            // PHS & Defects
            el.modalPhsScore.textContent = c.phs != null ? Number(c.phs).toFixed(1) : "--";
            const condClass = (c.condition || "good").toLowerCase();
            el.modalConditionBadge.textContent = c.condition;
            el.modalConditionBadge.className = `condition-badge ${condClass}`;

            let pCount = c.pothole_count != null ? c.pothole_count : 0;
            let cCount = c.crack_count != null ? c.crack_count : 0;
            if (pCount === 0 && cCount === 0 && Array.isArray(c.detections) && c.detections.length > 0) {
                pCount = c.detections.filter(d => (d.class_name || "").toLowerCase().includes("pothole")).length;
                cCount = c.detections.filter(d => (d.class_name || "").toLowerCase().includes("crack")).length;
            }
            el.modalPotholesCount.textContent = pCount;
            el.modalCracksCount.textContent = cCount;
            el.modalTotalCount.textContent = c.total_defects || (pCount + cCount);

            // Detection list
            el.modalDetectionsList.innerHTML = "";
            (c.detections || []).forEach((d, idx) => {
                const row = document.createElement("div");
                row.className = "det-item-row";
                row.innerHTML = `
                    <span class="det-class-tag ${(d.class_name || '').toLowerCase()}">#${idx + 1} ${d.class_name}</span>
                    <span class="det-conf">${Math.round(d.confidence * 100)}% conf</span>
                `;
                el.modalDetectionsList.appendChild(row);
            });

            // Location & Map with Accuracy Uncertainty Circle
            const altStr = c.altitude != null ? `, Altitude: ${c.altitude.toFixed(1)}m` : "";
            el.modalLocText.textContent = `Lat: ${c.latitude.toFixed(6)}, Lon: ${c.longitude.toFixed(6)}, Accuracy: ±${c.gps_accuracy || '--'}m${altStr}`;
            el.modalGmapsLink.href = `https://www.google.com/maps?q=${c.latitude},${c.longitude}`;
            initOrUpdateModalMap(c.latitude, c.longitude, c.condition, c.phs, c.gps_accuracy);

            // Citizen notes
            el.modalCitizenDesc.textContent = c.description || "No additional remarks provided.";
            el.modalCitizenPhone.textContent = c.citizen_phone ? `Contact: ${c.citizen_phone}` : "Contact: Not provided";

            // Timeline
            renderTimelineInModal(c.timeline || []);

            // Setup Officer Action Buttons
            setupModalOfficerControls(c.complaint_id);

            // Show modal
            el.modalComplaintDetail.style.display = "flex";
        } catch (err) {
            console.error("Error opening complaint detail:", err);
        }
    }

    function initOrUpdateModalMap(lat, lon, condition, phs, accuracy) {
        if (!state.modalMap) {
            state.modalMap = L.map("modal-leaflet-map", {
                zoomControl: true,
                attributionControl: false,
            }).setView([lat, lon], 16);

            L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                maxZoom: 19,
            }).addTo(state.modalMap);
        } else {
            state.modalMap.setView([lat, lon], 16);
            if (state.modalMarker) state.modalMap.removeLayer(state.modalMarker);
            if (state.modalCircle) state.modalMap.removeLayer(state.modalCircle);
        }

        let markerColor = "#10b981";
        if (condition === "CRITICAL") markerColor = "#ef4444";
        else if (condition === "POOR") markerColor = "#f97316";
        else if (condition === "MODERATE") markerColor = "#eab308";

        // Accuracy circle
        if (accuracy) {
            state.modalCircle = L.circle([lat, lon], {
                radius: accuracy,
                color: markerColor,
                fillColor: markerColor,
                fillOpacity: 0.15,
                weight: 1.5,
            }).addTo(state.modalMap);
        }

        state.modalMarker = L.circleMarker([lat, lon], {
            radius: 9,
            fillColor: markerColor,
            color: "#ffffff",
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9,
        }).addTo(state.modalMap);

        setTimeout(() => state.modalMap.invalidateSize(), 300);
    }

    function renderTimelineInModal(timeline) {
        el.modalTimelineContainer.innerHTML = "";
        timeline.forEach(step => {
            const div = document.createElement("div");
            div.className = "timeline-step";
            div.innerHTML = `
                <div class="timeline-dot"></div>
                <div class="timeline-content">
                    <div class="timeline-header">
                        <span>${step.event}</span>
                        <span class="timeline-time">${step.timestamp}</span>
                    </div>
                    <div class="timeline-actor">Actor: ${step.actor}</div>
                    ${step.notes ? `<div class="timeline-notes">${step.notes}</div>` : ""}
                </div>
            `;
            el.modalTimelineContainer.appendChild(div);
        });
    }

    function setupModalOfficerControls(cid) {
        const buttons = el.modalOfficerControls.querySelectorAll(".btn-status");
        buttons.forEach(btn => {
            btn.onclick = async () => {
                const targetStatus = btn.dataset.status;
                const notes = el.modalOfficerNotes.value.trim();
                await updateStatusAsOfficer(cid, targetStatus, notes);
            };
        });

        if (el.btnDeleteComplaint) {
            el.btnDeleteComplaint.onclick = () => {
                deleteComplaintAsOfficer(cid);
            };
        }
    }

    async function deleteComplaintAsOfficer(cid) {
        if (!state.officerToken) {
            alert("Officer authentication required to remove complaints.");
            return;
        }

        const confirmed = confirm(`Are you sure you want to permanently remove complaint ${cid}?\nThis action cannot be undone.`);
        if (!confirmed) return;

        try {
            const resp = await fetch(`/api/complaints/${cid}`, {
                method: "DELETE",
                headers: {
                    "Authorization": `Bearer ${state.officerToken}`,
                },
            });

            const data = await resp.json().catch(() => ({}));
            if (!resp.ok || !data.success) {
                throw new Error(data.detail || data.message || "Failed to remove complaint.");
            }

            // Close modal if open with this complaint
            if (el.modalComplaintDetail && el.modalComplaintDetail.style.display !== "none" && el.modalCid.textContent === cid) {
                el.modalComplaintDetail.style.display = "none";
            }

            // Refresh dashboard data
            await loadOfficerDashboard();
            if (state.activeOfficerTab === "complaints") {
                await loadAllComplaints();
            }
            if (state.activeOfficerTab === "map") {
                await loadOfficerMap();
            }
            fetchNotificationsCount();
        } catch (err) {
            alert(`Error removing complaint: ${err.message}`);
        }
    }

    async function updateStatusAsOfficer(cid, newStatus, notes) {
        if (!state.officerToken) {
            alert("Officer authentication required.");
            return;
        }

        try {
            const resp = await fetch(`/api/complaints/${cid}/status`, {
                method: "PATCH",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${state.officerToken}`,
                },
                body: JSON.stringify({
                    status: newStatus,
                    officer_notes: notes || `Officer updated status to ${newStatus}`,
                }),
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || "Status update failed.");
            }

            el.modalOfficerNotes.value = "";
            // Reload modal data to reflect new timeline
            openComplaintDetailModal(cid);

            // Refresh dashboard and complaints tables
            loadOfficerDashboard();
            if (state.activeOfficerTab === "complaints") loadAllComplaints();
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 10. OFFICER MAP (LEAFLET MULTI-MARKER)
    // ═══════════════════════════════════════════════════════════════════

    async function loadOfficerMap() {
        try {
            const resp = await fetch("/api/map/complaints");
            const data = await resp.json();
            if (!data.success) return;

            const complaints = data.complaints || [];

            if (!state.officerMap) {
                state.officerMap = L.map("officer-leaflet-map", {
                    zoomControl: true,
                    attributionControl: false,
                }).setView([13.0827, 80.2707], 12);

                L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                    maxZoom: 19,
                }).addTo(state.officerMap);
            }

            // Clear old markers
            state.officerMarkers.forEach(m => state.officerMap.removeLayer(m));
            state.officerMarkers = [];

            const bounds = [];

            complaints.forEach(c => {
                let markerColor = "#10b981";
                if (c.condition === "CRITICAL") markerColor = "#ef4444";
                else if (c.condition === "POOR") markerColor = "#f97316";
                else if (c.condition === "MODERATE") markerColor = "#eab308";

                // Accuracy circle on officer map
                if (c.gps_accuracy) {
                    const circle = L.circle([c.latitude, c.longitude], {
                        radius: c.gps_accuracy,
                        color: markerColor,
                        fillColor: markerColor,
                        fillOpacity: 0.12,
                        weight: 1,
                    }).addTo(state.officerMap);
                    state.officerMarkers.push(circle);
                }

                const marker = L.circleMarker([c.latitude, c.longitude], {
                    radius: 9,
                    fillColor: markerColor,
                    color: "#ffffff",
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.9,
                }).addTo(state.officerMap);

                let pCount = c.pothole_count != null ? c.pothole_count : 0;
                let cCount = c.crack_count != null ? c.crack_count : 0;

                const popupHtml = `
                    <div style="font-family:var(--font-main); font-size:13px; min-width:180px;">
                        <strong style="color:#06b6d4;">${c.complaint_id}</strong><br>
                        <span>RHS: <b>${c.phs}</b> (${c.condition})</span><br>
                        <span>Status: <b>${c.status}</b></span><br>
                        <span>GPS Accuracy: <b>±${c.gps_accuracy || '--'}m</b></span><br>
                        <span>Defects: ${c.total_defects || (pCount + cCount)} (${pCount} Potholes, ${cCount} Cracks)</span><br>
                        <button onclick="window.__openDetail('${c.complaint_id}')" style="margin-top:6px; background:#06b6d4; color:#080d1a; border:none; padding:4px 8px; border-radius:4px; font-weight:700; cursor:pointer; width:100%;">Inspect Complaint</button>
                    </div>
                `;

                marker.bindPopup(popupHtml);
                state.officerMarkers.push(marker);
                bounds.push([c.latitude, c.longitude]);
            });

            if (bounds.length > 0) {
                state.officerMap.fitBounds(bounds, { padding: [40, 40] });
            }

            setTimeout(() => state.officerMap.invalidateSize(), 300);
        } catch (err) {
            console.error("Map load error:", err);
        }
    }

    // Expose inspect modal helper to window for popup clicks
    window.__openDetail = (cid) => {
        openComplaintDetailModal(cid);
    };

    // ═══════════════════════════════════════════════════════════════════
    // 11. OFFICER NOTIFICATIONS & REAL-TIME POLLING
    // ═══════════════════════════════════════════════════════════════════

    function startNotificationPolling() {
        fetchNotificationsCount();
        if (state.pollingTimer) clearInterval(state.pollingTimer);
        // Poll every 8 seconds
        state.pollingTimer = setInterval(fetchNotificationsCount, 8000);
    }

    async function fetchNotificationsCount() {
        try {
            const resp = await fetch("/api/notifications?limit=5");
            const data = await resp.json();
            if (!data.success) return;

            const unread = data.unread_count || 0;
            if (unread > 0) {
                el.navNotifBadge.textContent = unread;
                el.navNotifBadge.style.display = "inline-block";
                el.sidebarNotifBadge.textContent = unread;
                el.sidebarNotifBadge.style.display = "inline-block";
            } else {
                el.navNotifBadge.style.display = "none";
                el.sidebarNotifBadge.style.display = "none";
            }
        } catch (e) {}
    }

    async function loadNotificationsList() {
        try {
            const resp = await fetch("/api/notifications?limit=50");
            const data = await resp.json();
            if (!data.success) return;

            el.officerNotificationsContainer.innerHTML = "";
            const list = data.notifications || [];

            if (list.length === 0) {
                el.officerNotificationsContainer.innerHTML = `<p class="empty-state">No notifications recorded yet.</p>`;
                return;
            }

            list.forEach(n => {
                const item = document.createElement("div");
                item.className = `notif-item ${n.is_read ? '' : 'unread'}`;
                item.innerHTML = `
                    <div class="notif-icon">🔔</div>
                    <div class="notif-body">
                        <div class="notif-title">${n.message}</div>
                        <div class="notif-detail">${n.detail || ''}</div>
                        <div class="notif-time">${n.created_at}</div>
                    </div>
                `;
                item.addEventListener("click", async () => {
                    // Mark read
                    if (!n.is_read && state.officerToken) {
                        fetch(`/api/notifications/${n.id}/read`, {
                            method: "PATCH",
                            headers: { "Authorization": `Bearer ${state.officerToken}` },
                        });
                        item.classList.remove("unread");
                    }
                    if (n.complaint_id) {
                        openComplaintDetailModal(n.complaint_id);
                    }
                });
                el.officerNotificationsContainer.appendChild(item);
            });
        } catch (err) {
            console.error("Notifications list error:", err);
        }
    }

    async function markAllNotificationsRead() {
        if (!state.officerToken) return;
        try {
            await fetch("/api/notifications/read-all", {
                method: "POST",
                headers: { "Authorization": `Bearer ${state.officerToken}` },
            });
            loadNotificationsList();
            fetchNotificationsCount();
        } catch (e) {}
    }

    // ═══════════════════════════════════════════════════════════════════
    // 12. OFFICER ANALYTICS CHARTS
    // ═══════════════════════════════════════════════════════════════════

    async function loadAnalytics() {
        try {
            const resp = await fetch("/api/dashboard/stats");
            const data = await resp.json();
            if (!data.success) return;

            const stats = data.stats;
            const def = stats.defects_summary;
            const cond = stats.condition_distribution;
            const totalDef = Math.max(def.total, 1);
            const totalComplaints = Math.max(stats.total_complaints, 1);

            // Defect Bars
            const potholePct = Math.round((def.potholes / totalDef) * 100);
            const crackPct = Math.round((def.cracks / totalDef) * 100);

            el.analyticsDefectBars.innerHTML = `
                <div class="chart-bar-item">
                    <div class="chart-bar-header">
                        <span>Potholes</span>
                        <span>${def.potholes} (${potholePct}%)</span>
                    </div>
                    <div class="chart-bar-track">
                        <div class="chart-bar-fill red" style="width:${potholePct}%;"></div>
                    </div>
                </div>
                <div class="chart-bar-item">
                    <div class="chart-bar-header">
                        <span>Cracks</span>
                        <span>${def.cracks} (${crackPct}%)</span>
                    </div>
                    <div class="chart-bar-track">
                        <div class="chart-bar-fill cyan" style="width:${crackPct}%;"></div>
                    </div>
                </div>
            `;

            // Condition Bars
            const critPct = Math.round((cond.critical / totalComplaints) * 100);
            const poorPct = Math.round((cond.poor / totalComplaints) * 100);
            const modPct = Math.round((cond.moderate / totalComplaints) * 100);
            const goodPct = Math.round((cond.good / totalComplaints) * 100);

            el.analyticsConditionBars.innerHTML = `
                <div class="chart-bar-item">
                    <div class="chart-bar-header">
                        <span>Critical (RHS 0-39)</span>
                        <span>${cond.critical} (${critPct}%)</span>
                    </div>
                    <div class="chart-bar-track">
                        <div class="chart-bar-fill red" style="width:${critPct}%;"></div>
                    </div>
                </div>
                <div class="chart-bar-item">
                    <div class="chart-bar-header">
                        <span>Poor (RHS 40-59)</span>
                        <span>${cond.poor} (${poorPct}%)</span>
                    </div>
                    <div class="chart-bar-track">
                        <div class="chart-bar-fill orange" style="width:${poorPct}%;"></div>
                    </div>
                </div>
                <div class="chart-bar-item">
                    <div class="chart-bar-header">
                        <span>Moderate (RHS 60-74)</span>
                        <span>${cond.moderate} (${modPct}%)</span>
                    </div>
                    <div class="chart-bar-track">
                        <div class="chart-bar-fill yellow" style="width:${modPct}%;"></div>
                    </div>
                </div>
                <div class="chart-bar-item">
                    <div class="chart-bar-header">
                        <span>Good / Excellent (RHS 75-100)</span>
                        <span>${cond.good} (${goodPct}%)</span>
                    </div>
                    <div class="chart-bar-track">
                        <div class="chart-bar-fill green" style="width:${goodPct}%;"></div>
                    </div>
                </div>
            `;
        } catch (err) {
            console.error("Analytics load error:", err);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 13. EVENT LISTENERS
    // ═══════════════════════════════════════════════════════════════════

    function setupEventListeners() {
        // Portal switching
        el.btnPortalCitizen.addEventListener("click", () => switchPortal("citizen"));
        el.btnPortalOfficer.addEventListener("click", () => switchPortal("officer"));
        el.brandHomeBtn.addEventListener("click", () => {
            switchPortal("citizen");
            switchCitizenTab("report");
        });

        // Citizen subtabs
        el.cnavReport.addEventListener("click", () => switchCitizenTab("report"));
        el.cnavTrack.addEventListener("click", () => switchCitizenTab("track"));
        el.cnavAbout.addEventListener("click", () => switchCitizenTab("about"));

        // Step 1: Image Capture
        el.inputCamera.addEventListener("change", (e) => handleFileSelection(e.target.files[0]));
        el.inputUpload.addEventListener("change", (e) => handleFileSelection(e.target.files[0]));
        el.btnClearImg.addEventListener("click", clearSelectedImage);

        // Drag & Drop
        el.dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            el.dropZone.style.borderColor = "var(--cyan-primary)";
        });
        el.dropZone.addEventListener("dragleave", () => {
            el.dropZone.style.borderColor = "var(--border-color)";
        });
        el.dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            el.dropZone.style.borderColor = "var(--border-color)";
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleFileSelection(e.dataTransfer.files[0]);
            }
        });

        // Step 2: GPS Location Capture
        if (el.btnGetGps) el.btnGetGps.addEventListener("click", acquireLocation);
        if (el.btnReacquireGps) el.btnReacquireGps.addEventListener("click", reacquireLocation);

        // Step 3: Analyze
        el.btnAnalyze.addEventListener("click", runAiRoadAnalysis);

        // Step 5: Submit Complaint
        el.btnSubmitComplaint.addEventListener("click", submitCitizenComplaint);
        el.btnViewMyComplaint.addEventListener("click", () => {
            if (state.submittedComplaintId) {
                switchCitizenTab("track");
                el.trackSearchInput.value = state.submittedComplaintId;
                searchAndTrackComplaint(state.submittedComplaintId);
            }
        });
        el.btnNewReport.addEventListener("click", () => {
            clearSelectedImage();
            reacquireLocation();
            el.resultSection.style.display = "none";
            window.scrollTo({ top: 0, behavior: "smooth" });
        });


        // Track search
        el.btnTrackSearch.addEventListener("click", () => searchAndTrackComplaint());
        el.trackSearchInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") searchAndTrackComplaint();
        });

        // Officer Login
        el.formOfficerLogin.addEventListener("submit", handleOfficerLogin);
        el.btnOfficerLogout.addEventListener("click", handleOfficerLogout);
        el.btnSidebarSignout.addEventListener("click", handleOfficerLogout);
        el.btnQuickFillDemo.addEventListener("click", () => {
            el.loginEmail.value = "bharath4330";
            el.loginPassword.value = "";
            el.loginPassword.focus();
        });

        // Officer Sidebar Tab Switching
        el.snavBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                switchOfficerTab(btn.dataset.target);
            });
        });

        // Dashboard actions
        el.btnRefreshDash.addEventListener("click", loadOfficerDashboard);
        el.btnGotoAllComplaints.addEventListener("click", () => switchOfficerTab("tab-complaints"));

        // Complaints filters
        el.filterStatusSelect.addEventListener("change", loadAllComplaints);
        el.filterConditionSelect.addEventListener("change", loadAllComplaints);
        el.filterSortSelect.addEventListener("change", loadAllComplaints);
        el.filterSearchInput.addEventListener("input", debounce(loadAllComplaints, 300));

        // Notifications
        el.btnMarkAllRead.addEventListener("click", markAllNotificationsRead);

        // Modal close
        el.btnCloseDetailModal.addEventListener("click", () => {
            el.modalComplaintDetail.style.display = "none";
        });
        el.modalComplaintDetail.addEventListener("click", (e) => {
            if (e.target === el.modalComplaintDetail) {
                el.modalComplaintDetail.style.display = "none";
            }
        });
    }

    function debounce(fn, delay) {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => fn(...args), delay);
        };
    }

    // Run initialization
    init();
});
