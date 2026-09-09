"""
RoadHealth AI — Video Analysis Pipeline
Performs frame-sampled AI road distress inspection using the proposed YOLO11m-P2P3-CBAM model.
Features multi-frame temporal object tracking to eliminate duplicate defect counting and
computes specialized video inspection telemetry (Lateral Wheelpath Analysis, Road Segment Chainage Profiles,
and Surface Roughness / Ride Disturbance Indices).
"""

import os
import cv2
import time
import tempfile
import numpy as np
from pathlib import Path
import imageio

# Local pipeline modules
from inference_pipeline import run_inference
from detection_processing import process_detections
from feature_extraction import extract_features
from progression_analysis import compute_progression
from road_health import compute_road_health
from visualization import draw_detections


def get_video_metadata(video_path: str) -> dict:
    """Extract technical video properties using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "error": "Could not open video file",
            "duration": 0, "fps": 0, "total_frames": 0, "width": 0, "height": 0
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024) if os.path.exists(video_path) else 0.0

    cap.release()

    mins = int(duration_sec // 60)
    secs = int(duration_sec % 60)
    duration_str = f"{mins:02d}:{secs:02d}"

    return {
        "file_name": os.path.basename(video_path),
        "duration_sec": duration_sec,
        "duration_str": duration_str,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "resolution": f"{width} × {height}",
        "file_size_mb": round(file_size_mb, 2),
    }


# ══════════════════════════════════════════════════════════════════════
# MULTI-FRAME TEMPORAL OBJECT TRACKER (FIXES FALSE DUPLICATE COUNTING)
# ══════════════════════════════════════════════════════════════════════

def _compute_iou(boxA, boxB):
    """Compute Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return inter_area / float(boxA_area + boxB_area - inter_area)


def _classify_wheelpath(box, img_w):
    """Classify distress lateral position based on horizontal center relative to road width."""
    cx = (box[0] + box[2]) / 2.0
    norm_x = cx / max(1.0, float(img_w))
    if norm_x < 0.35:
        return "Left Wheelpath (LWP)"
    elif norm_x <= 0.65:
        return "Center Lane (CL)"
    else:
        return "Right Wheelpath (RWP)"


class RoadDefectTracker:
    """
    Multi-frame spatial-temporal object tracker designed for forward-moving road inspection video.
    Maintains persistent IDs across sampled frames, preventing duplicate counting of the same defect.
    """
    def __init__(self, width: int, height: int, max_missed_frames: int = 6):
        self.width = width
        self.height = height
        self.max_missed_frames = max_missed_frames
        
        self.pothole_counter = 0
        self.crack_counter = 0
        self.tracks = {}  # track_id -> track dict

    def update(self, detections: list, frame_num: int, timestamp_sec: float) -> list:
        """
        Associates current frame detections with active tracks.
        Returns features updated with assigned `track_id` and `track_label`.
        """
        updated_features = []
        unmatched_dets = list(range(len(detections)))
        
        # 1. Match against existing active tracks
        for track_id, track in list(self.tracks.items()):
            # If missed for too many frames, do not match
            if (frame_num - track["last_frame"]) > (self.max_missed_frames * 30):
                continue

            best_match_idx = -1
            best_score = float("inf")

            last_box = track["last_box"]
            last_cx = (last_box[0] + last_box[2]) / 2.0
            last_cy = (last_box[1] + last_box[3]) / 2.0

            for det_idx in unmatched_dets:
                det = detections[det_idx]
                if det["class_name"].lower() != track["class_name"].lower():
                    continue

                curr_box = det["box"]
                curr_cx = (curr_box[0] + curr_box[2]) / 2.0
                curr_cy = (curr_box[1] + curr_box[3]) / 2.0

                iou = _compute_iou(last_box, curr_box)
                
                # Normalized distance accounting for forward camera motion (y moves downward)
                norm_dx = abs(curr_cx - last_cx) / self.width
                norm_dy = (curr_cy - last_cy) / self.height  # positive means moved downward towards camera

                # Road vehicle motion affinity check:
                # Forward driving means distress moves downward (norm_dy >= -0.08)
                if norm_dy < -0.12:
                    continue  # Object moved backward too abruptly, likely different defect

                dist_score = float(np.sqrt((1.4 * norm_dx) ** 2 + max(0.0, norm_dy) ** 2))

                # Match criteria
                is_match = (
                    (iou > 0.15) or
                    (dist_score < 0.22 and norm_dy >= -0.06) or
                    (iou > 0.04 and dist_score < 0.30)
                )

                if is_match and dist_score < best_score:
                    best_score = dist_score
                    best_match_idx = det_idx

            # If match found, update existing track
            if best_match_idx != -1:
                det = detections[best_match_idx]
                unmatched_dets.remove(best_match_idx)

                track["last_box"] = det["box"]
                track["last_frame"] = frame_num
                track["last_timestamp"] = timestamp_sec
                track["duration_sec"] = round(timestamp_sec - track["first_timestamp"], 2)
                track["persistence_frames"] += 1
                track["observations"].append({
                    "frame": frame_num,
                    "timestamp_sec": timestamp_sec,
                    "box": det["box"],
                    "confidence": det.get("confidence", 0.0),
                    "severity": det.get("severity_score", 0.0),
                })
                track["max_severity"] = max(track["max_severity"], det.get("severity_score", 0.0))
                track["conf_sum"] += det.get("confidence", 0.0)

                # Update wheelpath based on detection
                wp = _classify_wheelpath(det["box"], self.width)
                track["wheelpath"] = wp

                det_copy = dict(det)
                det_copy["track_id"] = track_id
                det_copy["track_label"] = track["label"]
                det_copy["wheelpath"] = wp
                updated_features.append(det_copy)

        # 2. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            c_name = det["class_name"].lower()
            if c_name == "pothole":
                self.pothole_counter += 1
                t_label = f"P-{self.pothole_counter:02d}"
            else:
                self.crack_counter += 1
                t_label = f"C-{self.crack_counter:02d}"

            track_id = f"{c_name}_{len(self.tracks) + 1}"
            wp = _classify_wheelpath(det["box"], self.width)

            new_track = {
                "track_id": track_id,
                "label": t_label,
                "class_name": c_name,
                "first_frame": frame_num,
                "first_timestamp": timestamp_sec,
                "last_frame": frame_num,
                "last_timestamp": timestamp_sec,
                "duration_sec": 0.0,
                "persistence_frames": 1,
                "max_severity": det.get("severity_score", 0.0),
                "conf_sum": det.get("confidence", 0.0),
                "last_box": det["box"],
                "wheelpath": wp,
                "observations": [{
                    "frame": frame_num,
                    "timestamp_sec": timestamp_sec,
                    "box": det["box"],
                    "confidence": det.get("confidence", 0.0),
                    "severity": det.get("severity_score", 0.0),
                }],
            }
            self.tracks[track_id] = new_track

            det_copy = dict(det)
            det_copy["track_id"] = track_id
            det_copy["track_label"] = t_label
            det_copy["wheelpath"] = wp
            updated_features.append(det_copy)

        return updated_features

    def get_summary_inventory(self) -> list:
        """Returns structured inventory of all unique tracked physical road distresses."""
        inventory = []
        for tid, t in self.tracks.items():
            mean_conf = t["conf_sum"] / max(1, t["persistence_frames"])
            
            # Risk rating based on peak severity & persistence
            max_sev = t["max_severity"]
            if max_sev >= 0.70 or (max_sev >= 0.55 and t["persistence_frames"] >= 4):
                risk_level = "CRITICAL"
            elif max_sev >= 0.45:
                risk_level = "HIGH"
            elif max_sev >= 0.20:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"

            mins_start = int(t["first_timestamp"] // 60)
            secs_start = int(t["first_timestamp"] % 60)
            first_ts_str = f"{mins_start:02d}:{secs_start:02d}"

            mins_end = int(t["last_timestamp"] // 60)
            secs_end = int(t["last_timestamp"] % 60)
            last_ts_str = f"{mins_end:02d}:{secs_end:02d}"

            inventory.append({
                "track_id": t["label"],
                "class_name": t["class_name"].title(),
                "first_seen": first_ts_str,
                "first_sec": round(t["first_timestamp"], 2),
                "last_seen": last_ts_str,
                "last_sec": round(t["last_timestamp"], 2),
                "duration_sec": max(0.1, round(t["duration_sec"], 2)),
                "persistence_frames": t["persistence_frames"],
                "wheelpath_position": t["wheelpath"],
                "max_severity": round(max_sev, 3),
                "mean_confidence": round(mean_conf, 3),
                "risk_rating": risk_level,
            })

        # Sort by first seen timestamp
        inventory.sort(key=lambda x: x["first_sec"])
        return inventory


# ══════════════════════════════════════════════════════════════════════
# SPECIALIZED VIDEO ROAD INSPECTION TELEMETRY METRICS
# ══════════════════════════════════════════════════════════════════════

def _compute_lateral_wheelpath_analysis(tracked_inventory: list) -> dict:
    """
    Evaluates distress distribution across vehicle wheelpaths vs center lane.
    Wheelpath defects (LWP / RWP) cause immediate tire/suspension impact.
    """
    lwp_defects = [d for d in tracked_inventory if "Left Wheelpath" in d["wheelpath_position"]]
    cl_defects = [d for d in tracked_inventory if "Center Lane" in d["wheelpath_position"]]
    rwp_defects = [d for d in tracked_inventory if "Right Wheelpath" in d["wheelpath_position"]]

    total_unique = len(tracked_inventory)
    wheelpath_count = len(lwp_defects) + len(rwp_defects)
    wp_hazard_ratio = (wheelpath_count / total_unique * 100.0) if total_unique > 0 else 0.0

    def _wp_stats(subset, name):
        p_count = sum(1 for d in subset if d["class_name"].lower() == "pothole")
        c_count = sum(1 for d in subset if d["class_name"].lower() == "crack")
        max_s = max([d["max_severity"] for d in subset], default=0.0)
        
        if max_s >= 0.70 or p_count >= 2:
            status = "HIGH HAZARD"
            desc = "Direct tire impact zone with critical defect depth; high blowout and suspension fatigue risk."
        elif max_s >= 0.35 or (p_count + c_count) > 0:
            status = "MODERATE RISK"
            desc = "Noticeable pavement distress in travel track; requires scheduled surface rehabilitation."
        else:
            status = "NOMINAL"
            desc = "No severe distress detected along this lateral track."

        return {
            "name": name,
            "total_defects": len(subset),
            "potholes": p_count,
            "cracks": c_count,
            "max_severity": round(max_s, 3),
            "status": status,
            "description": desc,
        }

    counts = {"Left Wheelpath": len(lwp_defects), "Center Lane": len(cl_defects), "Right Wheelpath": len(rwp_defects)}
    dom_lane = max(counts, key=counts.get) if total_unique > 0 else "None"

    return {
        "left_wheelpath": _wp_stats(lwp_defects, "Left Wheelpath (LWP)"),
        "center_lane": _wp_stats(cl_defects, "Center Lane / Mid-Track (CL)"),
        "right_wheelpath": _wp_stats(rwp_defects, "Right Wheelpath (RWP)"),
        "wheelpath_hazard_ratio": round(wp_hazard_ratio, 1),
        "dominant_distress_lane": dom_lane,
        "direct_impact_hazards": wheelpath_count,
    }


def _compute_chronological_segment_profile(frame_results: list, duration_sec: float, segment_window_sec: float = 5.0) -> list:
    """
    Partitions the inspected video stream into sequential road chainage segments (e.g. 5-second blocks).
    Computes localized segment health, damage variation, and maintenance recommendation per segment.
    """
    if not frame_results or duration_sec <= 0:
        return []

    num_segments = max(1, int(np.ceil(duration_sec / segment_window_sec)))
    segments = []

    for i in range(num_segments):
        start_t = i * segment_window_sec
        end_t = min(duration_sec, (i + 1) * segment_window_sec)
        
        m_s = int(start_t // 60)
        s_s = int(start_t % 60)
        m_e = int(end_t // 60)
        s_e = int(end_t % 60)
        t_range_str = f"{m_s:02d}:{s_s:02d} – {m_e:02d}:{s_e:02d}"

        seg_frames = [
            f for f in frame_results 
            if start_t <= f["timestamp_sec"] < end_t or (i == num_segments - 1 and f["timestamp_sec"] <= end_t)
        ]

        if seg_frames:
            seg_phs = float(np.mean([f["phs"] for f in seg_frames]))
            seg_dmg = float(np.mean([f["damage_score"] for f in seg_frames]))
            max_dmg = float(np.max([f["damage_score"] for f in seg_frames]))
            pothole_frames = sum(1 for f in seg_frames if f["potholes"] > 0)
            crack_frames = sum(1 for f in seg_frames if f["cracks"] > 0)
            peak_p = max([f["potholes"] for f in seg_frames], default=0)
            peak_c = max([f["cracks"] for f in seg_frames], default=0)
        else:
            seg_phs = 100.0
            seg_dmg = 0.0
            max_dmg = 0.0
            pothole_frames = 0
            crack_frames = 0
            peak_p = 0
            peak_c = 0

        # Segment Condition Classification
        if seg_phs >= 85:
            cond = "VERY GOOD"
            action = "Routine surveillance; pavement intact."
        elif seg_phs >= 70:
            cond = "GOOD"
            action = "Nominal condition; preventive monitoring."
        elif seg_phs >= 50:
            cond = "FAIR"
            action = "Crack sealing and surface leveling recommended."
        elif seg_phs >= 30:
            cond = "POOR"
            action = "High priority patching required; localized distress."
        else:
            cond = "CRITICAL"
            action = "Emergency asphalt resurfacing & structural repair required."

        segments.append({
            "segment_id": f"Segment {i + 1}",
            "time_range": t_range_str,
            "start_sec": round(start_t, 1),
            "end_sec": round(end_t, 1),
            "segment_phs": round(seg_phs, 1),
            "segment_damage": round(seg_dmg, 3),
            "peak_damage": round(max_dmg, 3),
            "condition": cond,
            "active_pothole_frames": pothole_frames,
            "active_crack_frames": crack_frames,
            "peak_potholes_in_frame": peak_p,
            "peak_cracks_in_frame": peak_c,
            "action_recommendation": action,
        })

    return segments


def _compute_surface_roughness_index(tracked_inventory: list, duration_sec: float, frames_with_defects: int, total_frames: int) -> dict:
    """
    Computes dynamic Road Surface Roughness and Ride Disturbance Index (RDI / Estimated IRI Proxy).
    Based on unique pothole volume, crack encounter rate, and wheelpath hit probability.
    """
    unique_potholes = [d for d in tracked_inventory if d["class_name"].lower() == "pothole"]
    unique_cracks = [d for d in tracked_inventory if d["class_name"].lower() == "crack"]
    
    wp_potholes = sum(1 for d in unique_potholes if "Wheelpath" in d["wheelpath_position"])
    cl_potholes = len(unique_potholes) - wp_potholes

    max_p_sev = max([d["max_severity"] for d in unique_potholes], default=0.0)
    max_c_sev = max([d["max_severity"] for d in unique_cracks], default=0.0)

    # Ride Disturbance Score (0.0 to 10.0 scale)
    base_rdi = (
        (wp_potholes * 2.2) +
        (cl_potholes * 1.2) +
        (len(unique_cracks) * 0.7) +
        (max_p_sev * 3.5) +
        (max_c_sev * 1.5)
    )
    # Factor in percentage of road timeline affected
    distress_ratio = (frames_with_defects / max(1, total_frames))
    rdi_score = min(10.0, base_rdi * (0.6 + 0.8 * distress_ratio))
    rdi_score = max(0.5 if (unique_potholes or unique_cracks) else 0.0, rdi_score)

    # Estimated International Roughness Index (IRI in m/km proxy)
    est_iri = round(1.8 + (rdi_score / 10.0) * 6.5, 2)

    if est_iri < 2.5:
        ride_quality = "SMOOTH RIDE"
        vib_hazard = "MINIMAL"
        ride_desc = "Optimal vehicular ride comfort; zero to negligible vehicle suspension oscillation."
    elif est_iri < 4.2:
        ride_quality = "MODERATE RIDE"
        vib_hazard = "LOW–MEDIUM"
        ride_desc = "Minor surface roughness; slight vehicle cabin vibration without immediate hazard."
    elif est_iri < 6.5:
        ride_quality = "ROUGH / BUMPY"
        vib_hazard = "HIGH"
        ride_desc = "Severe road surface irregularities; high vehicular vibration, driver discomfort, and tire stress."
    else:
        ride_quality = "HAZARDOUS / DESTRUCTIVE"
        vib_hazard = "CRITICAL"
        ride_desc = "Severe structural pavement failure; extreme mechanical shock and vehicle damage risk."

    return {
        "ride_disturbance_index": round(rdi_score, 2),
        "estimated_iri_m_km": est_iri,
        "ride_quality": ride_quality,
        "vibration_hazard": vib_hazard,
        "description": ride_desc,
    }


def _draw_hud_telemetry(img_bgr, ts_str, u_potholes, u_cracks, frame_p, frame_c, frame_phs):
    """Draws a sleek semi-transparent HUD telemetry banner on top of the frame."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    
    # Top overlay bar
    overlay = out.copy()
    bar_h = max(38, int(h * 0.068))
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (11, 18, 32), -1)
    cv2.addWeighted(overlay, 0.84, out, 0.16, 0, out)
    
    # Accent indicator line
    cv2.line(out, (0, bar_h), (w, bar_h), (59, 130, 246), 2)
    
    # Text strings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.40, w / 1650)
    thick = max(1, int(w / 800))
    
    txt_left = f"RoadHealth AI | Video Inspection | {ts_str}"
    txt_right = f"Unique: {u_potholes} Pothole(s) / {u_cracks} Crack(s) | RHS: {frame_phs:.0f}"
    
    cv2.putText(out, txt_left, (14, int(bar_h * 0.65)), font, scale, (34, 211, 238), thick, cv2.LINE_AA)
    
    # Measure text for right alignment
    (tw, _), _ = cv2.getTextSize(txt_right, font, scale, thick)
    rx = max(10, w - tw - 14)
    cv2.putText(out, txt_right, (rx, int(bar_h * 0.65)), font, scale, (248, 250, 252), thick, cv2.LINE_AA)
    
    return out


# ══════════════════════════════════════════════════════════════════════
# MAIN VIDEO ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════

def analyze_video(
    video_path: str,
    model,
    sample_interval: int = 5,
    generate_annotated_video: bool = True,
    progress_callback=None,
) -> dict:
    """
    Executes frame-sampled pavement inspection on a video file.
    Reuses the loaded YOLO11m-P2P3-CBAM model without reloading per frame.
    Integrates RoadDefectTracker to guarantee accurate unique defect counting.
    Generates a full H.264 annotated video for direct in-browser playback and download.
    """
    t_start = time.time()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize Temporal Tracker
    tracker = RoadDefectTracker(width=width, height=height, max_missed_frames=6)

    # Containers for temporal analysis
    analyzed_frames_count = 0
    frame_results = []
    detection_observations = []
    timeline = []
    
    # Time-series data points
    ts_seconds = []
    ts_damage_scores = []
    ts_phs_scores = []
    ts_pothole_counts = []
    ts_crack_counts = []

    # Representative frame buffers
    peak_damage = -1.0
    peak_frame_orig = None
    peak_frame_annotated = None
    peak_frame_info = None

    first_defect_frame_orig = None
    first_defect_frame_annotated = None
    first_defect_info = None

    # Setup Video Writer for Full Annotated Output Video
    video_writer = None
    temp_output_path = None
    if generate_annotated_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="_annotated.mp4")
        temp_output_path = temp_file.name
        temp_file.close()
        try:
            video_writer = imageio.get_writer(
                temp_output_path,
                fps=fps,
                codec='libx264',
                quality=7,
                pixelformat='yuv420p',
                macro_block_size=1,
            )
        except Exception:
            # Fallback to OpenCV VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    current_frame_idx = 0
    cached_features = []
    cached_phs = 100.0
    cached_potholes = 0
    cached_cracks = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        timestamp_sec = current_frame_idx / fps if fps > 0 else 0.0
        mins = int(timestamp_sec // 60)
        secs = int(timestamp_sec % 60)
        ts_str = f"{mins:02d}:{secs:02d}"

        if current_frame_idx % sample_interval == 0:
            analyzed_frames_count += 1

            if progress_callback is not None:
                progress_callback(
                    analyzed_frames_count,
                    max(1, total_frames // sample_interval),
                    f"Analyzing frame {current_frame_idx + 1} / {total_frames} ({ts_str})"
                )

            # 1. Run model detection on frame
            raw_dets, nms_dets = run_inference(model, frame_bgr)
            final_dets = process_detections(nms_dets, frame_bgr)
            features = extract_features(final_dets, frame_bgr)

            # 2. Update Temporal Object Tracker
            tracked_features = tracker.update(features, current_frame_idx + 1, timestamp_sec)

            # 3. Health Analysis for frame
            prog_records, highest_risk, highest_level = compute_progression(tracked_features, width, height)
            frame_health = compute_road_health(tracked_features, {}, prog_records)

            potholes_in_frame = [d for d in tracked_features if d["class_name"].lower() == "pothole"]
            cracks_in_frame = [d for d in tracked_features if d["class_name"].lower() == "crack"]
            n_potholes = len(potholes_in_frame)
            n_cracks = len(cracks_in_frame)
            frame_damage = frame_health["overall_damage"]
            frame_phs = frame_health["pavement_health_score"]

            cached_features = tracked_features
            cached_phs = frame_phs
            cached_potholes = n_potholes
            cached_cracks = n_cracks

            # Store time-series
            ts_seconds.append(round(timestamp_sec, 2))
            ts_damage_scores.append(round(frame_damage, 4))
            ts_phs_scores.append(round(frame_phs, 2))
            ts_pothole_counts.append(n_potholes)
            ts_crack_counts.append(n_cracks)

            # Record frame detection observations
            for det in tracked_features:
                det_record = {
                    "frame_number": current_frame_idx + 1,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "timestamp": ts_str,
                    "track_id": det.get("track_label", "N/A"),
                    "class": det["class_name"].title(),
                    "confidence": round(det["confidence"], 4),
                    "severity": round(det.get("severity_score", 0), 4),
                    "wheelpath": det.get("wheelpath", "Center Lane"),
                    "x1": round(det["box"][0], 1),
                    "y1": round(det["box"][1], 1),
                    "x2": round(det["box"][2], 1),
                    "y2": round(det["box"][3], 1),
                }
                detection_observations.append(det_record)

            # Append to timeline if distress detected
            if n_potholes > 0 or n_cracks > 0:
                defect_types = []
                if n_potholes > 0:
                    defect_types.append(f"{n_potholes} Pothole(s)")
                if n_cracks > 0:
                    defect_types.append(f"{n_cracks} Crack(s)")

                timeline_entry = {
                    "timestamp": ts_str,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "frame": current_frame_idx + 1,
                    "defects": " + ".join(defect_types),
                    "damage_score": round(frame_damage, 3),
                    "phs": round(frame_phs, 1),
                    "active_tracks": ", ".join([d.get("track_label", "") for d in tracked_features if d.get("track_label")]),
                }
                timeline.append(timeline_entry)

                if first_defect_frame_orig is None:
                    first_defect_frame_orig = frame_bgr.copy()
                    first_defect_frame_annotated = draw_detections(frame_bgr, tracked_features)
                    first_defect_info = timeline_entry

            # Check peak damage frame
            if frame_damage > peak_damage or peak_frame_orig is None:
                peak_damage = frame_damage
                peak_frame_orig = frame_bgr.copy()
                peak_frame_annotated = draw_detections(frame_bgr, tracked_features)
                peak_frame_info = {
                    "frame": current_frame_idx + 1,
                    "timestamp": ts_str,
                    "damage_score": round(frame_damage, 3),
                    "phs": round(frame_phs, 1),
                    "potholes": n_potholes,
                    "cracks": n_cracks,
                }

            frame_results.append({
                "frame": current_frame_idx + 1,
                "timestamp_sec": timestamp_sec,
                "timestamp": ts_str,
                "potholes": n_potholes,
                "cracks": n_cracks,
                "damage_score": frame_damage,
                "phs": frame_phs,
            })

        # Render frame for full video output with HUD telemetry
        if video_writer is not None:
            annotated_frame = draw_detections(frame_bgr, cached_features)
            annotated_frame = _draw_hud_telemetry(
                annotated_frame,
                ts_str,
                tracker.pothole_counter,
                tracker.crack_counter,
                cached_potholes,
                cached_cracks,
                cached_phs
            )
            if hasattr(video_writer, "append_data"):
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_writer.append_data(rgb_frame)
            else:
                video_writer.write(annotated_frame)

        current_frame_idx += 1

    cap.release()

    # Close and finalize video writer
    annotated_video_bytes = None
    if video_writer is not None:
        if hasattr(video_writer, "close"):
            video_writer.close()
        elif hasattr(video_writer, "release"):
            video_writer.release()
        
        if temp_output_path and os.path.exists(temp_output_path):
            with open(temp_output_path, "rb") as vf:
                annotated_video_bytes = vf.read()

    t_end = time.time()
    elapsed = t_end - t_start

    # =========================================================
    # VIDEO-LEVEL AGGREGATION & SPECIALIZED METRICS
    # =========================================================
    duration_sec = total_frames / fps if fps > 0 else 0.0
    mins = int(duration_sec // 60)
    secs = int(duration_sec % 60)
    duration_str = f"{mins:02d}:{secs:02d}"

    # Get Unique Tracked Defect Inventory
    tracked_inventory = tracker.get_summary_inventory()
    
    unique_potholes = sum(1 for d in tracked_inventory if d["class_name"].lower() == "pothole")
    unique_cracks = sum(1 for d in tracked_inventory if d["class_name"].lower() == "crack")
    total_unique_defects = len(tracked_inventory)

    frames_with_potholes = sum(1 for f in frame_results if f["potholes"] > 0)
    frames_with_cracks = sum(1 for f in frame_results if f["cracks"] > 0)
    frames_with_defects = sum(1 for f in frame_results if f["potholes"] > 0 or f["cracks"] > 0)

    pothole_obs = [d for d in detection_observations if d["class"].lower() == "pothole"]
    crack_obs = [d for d in detection_observations if d["class"].lower() == "crack"]

    total_pothole_obs = len(pothole_obs)
    total_crack_obs = len(crack_obs)
    total_obs = len(detection_observations)

    max_pothole_sev = max([d["max_severity"] for d in tracked_inventory if d["class_name"].lower() == "pothole"], default=0.0)
    max_crack_sev = max([d["max_severity"] for d in tracked_inventory if d["class_name"].lower() == "crack"], default=0.0)
    avg_confidence = float(np.mean([d["mean_confidence"] for d in tracked_inventory])) if tracked_inventory else 0.0

    # Calculate overall video damage & PHS
    if ts_damage_scores:
        mean_damage = float(np.mean(ts_damage_scores))
        max_damage = float(np.max(ts_damage_scores))
        video_damage_score = min(1.0, 0.60 * mean_damage + 0.40 * max_damage)
        video_phs = max(0.0, min(100.0, 100.0 * (1.0 - video_damage_score)))
    else:
        video_damage_score = 0.0
        video_phs = 100.0

    # Road condition classification
    if video_phs >= 85:
        road_cond = "VERY GOOD"
    elif video_phs >= 70:
        road_cond = "GOOD"
    elif video_phs >= 50:
        road_cond = "FAIR"
    elif video_phs >= 30:
        road_cond = "POOR"
    else:
        road_cond = "VERY POOR"

    # Maintenance Priority
    if video_phs < 30 or max_pothole_sev >= 0.70 or unique_potholes >= 3:
        maint_prio = "CRITICAL"
        maint_rec = "Emergency asphalt resurfacing and immediate structural pothole repairs required."
    elif video_phs < 50 or unique_potholes > 0:
        maint_prio = "HIGH"
        maint_rec = "Scheduled hot-mix asphalt patching and wheelpath structural verification required."
    elif video_phs < 70 or unique_cracks > 0:
        maint_prio = "MEDIUM"
        maint_rec = "Targeted crack sealing and preventive surface maintenance recommended within 1–2 weeks."
    elif video_phs < 85:
        maint_prio = "LOW"
        maint_rec = "Routine surveillance and preventive drainage maintenance recommended."
    else:
        maint_prio = "ROUTINE"
        maint_rec = "No structural repairs required. Continue routine periodic monitoring."

    # Compute Specialized Video Engineering Metrics
    lateral_analysis = _compute_lateral_wheelpath_analysis(tracked_inventory)
    segment_profile = _compute_chronological_segment_profile(frame_results, duration_sec, segment_window_sec=5.0)
    roughness_info = _compute_surface_roughness_index(tracked_inventory, duration_sec, frames_with_defects, analyzed_frames_count)

    # Distress Encounter Rate (DPM = Defects Per Minute)
    distress_rate_per_min = round((total_unique_defects / max(0.1, duration_sec)) * 60.0, 1) if duration_sec > 0 else 0.0
    active_distress_duration_pct = round((frames_with_defects / max(1, analyzed_frames_count)) * 100.0, 1)

    return {
        "metadata": {
            "file_name": os.path.basename(video_path),
            "duration_sec": duration_sec,
            "duration_str": duration_str,
            "fps": fps,
            "total_frames": total_frames,
            "analyzed_frames": analyzed_frames_count,
            "sample_interval": sample_interval,
            "resolution": f"{width} × {height}",
            "elapsed_time_sec": round(elapsed, 2),
        },
        "summary": {
            "video_pavement_health_score": round(video_phs, 2),
            "video_damage_score": round(video_damage_score, 4),
            "road_condition": road_cond,
            "maintenance_priority": maint_prio,
            "maintenance_recommendation": maint_rec,
            "unique_potholes": unique_potholes,
            "unique_cracks": unique_cracks,
            "total_unique_defects": total_unique_defects,
            "frames_with_potholes": frames_with_potholes,
            "frames_with_cracks": frames_with_cracks,
            "frames_with_defects": frames_with_defects,
            "total_pothole_observations": total_pothole_obs,
            "total_crack_observations": total_crack_obs,
            "total_observations": total_obs,
            "max_pothole_severity": round(max_pothole_sev, 4),
            "max_crack_severity": round(max_crack_sev, 4),
            "avg_confidence": round(avg_confidence, 4),
            "distress_rate_per_min": distress_rate_per_min,
            "active_distress_duration_pct": active_distress_duration_pct,
            "wheelpath_hazard_ratio": lateral_analysis["wheelpath_hazard_ratio"],
            "dominant_distress_lane": lateral_analysis["dominant_distress_lane"],
            "estimated_iri_m_km": roughness_info["estimated_iri_m_km"],
            "ride_quality": roughness_info["ride_quality"],
            "ride_disturbance_index": roughness_info["ride_disturbance_index"],
        },
        "tracked_inventory": tracked_inventory,
        "lateral_wheelpath_analysis": lateral_analysis,
        "segment_health_profile": segment_profile,
        "surface_roughness": roughness_info,
        "temporal": {
            "timeline": timeline,
            "timestamps_sec": ts_seconds,
            "damage_series": ts_damage_scores,
            "phs_series": ts_phs_scores,
            "pothole_series": ts_pothole_counts,
            "crack_series": ts_crack_counts,
        },
        "detection_observations": detection_observations,
        "representative_frames": {
            "peak_orig": peak_frame_orig,
            "peak_annotated": peak_frame_annotated,
            "peak_info": peak_frame_info,
            "first_orig": first_defect_frame_orig,
            "first_annotated": first_defect_frame_annotated,
            "first_info": first_defect_info,
        },
        "annotated_video_bytes": annotated_video_bytes,
        "annotated_video_path": temp_output_path,
    }
