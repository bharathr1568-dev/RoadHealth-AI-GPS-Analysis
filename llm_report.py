"""
RoadHealth AI — Groq LLM Report Generator
Generates a professional AI-powered final report using Groq API
with multimodal visual reasoning on the prediction image/video.
"""

import io
import re
import base64
import requests

import cv2
import numpy as np
from PIL import Image as PILImage

from config import GROQ_API_KEY


# ── Groq config ───────────────────────────────────────────────
GROQ_MODEL = "qwen/qwen3.6-27b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_IMAGE_SIZE = 560


def _prepare_image_base64(cv2_img_bgr):
    """Convert a BGR OpenCV image to a compressed base64 JPEG for Groq vision."""
    rgb = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(rgb)

    # Resize for lower token usage
    w, h = pil_img.size
    scale = min(1.0, MAX_IMAGE_SIZE / max(w, h))
    if scale < 1.0:
        pil_img = pil_img.resize(
            (int(w * scale), int(h * scale)),
            PILImage.Resampling.LANCZOS,
        )

    # JPEG compression
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=45, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ===============================================================
# 1. IMAGE REPORT PIPELINE (EXISTING — 100% PRESERVED)
# ===============================================================

def _build_engineering_context(features, zone_damage, progression_records, report_summary):
    """Build compact engineering context text from analysis data."""
    rs = report_summary
    potholes = [d for d in features if d["class_name"] == "pothole"]
    cracks = [d for d in features if d["class_name"] == "crack"]

    if not cracks:
        progression_text = (
            "No qualifying crack was detected by the engineering pipeline. "
            "Progression risk was therefore not estimated."
        )
    else:
        highest = rs.get("highest_crack_progression_risk")
        level = rs.get("crack_progression_level", "N/A")
        progression_text = (
            f"Highest progression risk score: {highest}; "
            f"risk level: {level}. "
            f"This is an image-based risk score, not a calibrated probability."
        )

    # Zone damage summary — only zones with damage
    zone_lines = []
    for z_id in sorted(zone_damage.keys()):
        z = zone_damage[z_id]
        if z["damage_score"] > 0:
            zone_lines.append(
                f"  Zone Z{z_id}: Damage={z['damage_score']:.3f}, "
                f"Risk={z['risk_level']}, "
                f"Potholes={z['potholes']}, Cracks={z['cracks']}"
            )
    zone_summary = "\n".join(zone_lines) if zone_lines else "  No zone damage detected."

    # Defect details
    defect_lines = []
    for i, d in enumerate(features, 1):
        defect_lines.append(
            f"  #{i} {d['class_name'].title()}: "
            f"Conf={d['confidence']:.3f}, "
            f"Severity={d.get('severity_score', 0):.3f}, "
            f"Area%={d.get('bbox_area_ratio', 0)*100:.2f}"
        )
    defect_summary = "\n".join(defect_lines) if defect_lines else "  No defects detected."

    return f"""
Potholes detected        : {rs['detected_potholes']}
Cracks detected          : {rs['detected_cracks']}
Pavement Health Score    : {rs['pavement_health_score']:.2f}/100
Computed road condition  : {rs['road_condition']}
Critical zone            : {rs['critical_zone']}
Critical zone score      : {rs['critical_zone_damage_score']:.3f}
Maintenance priority     : {rs['maintenance_priority']}
Overall damage score     : {rs['overall_damage']:.4f}
Existing recommendation  : {rs['maintenance_recommendation']}

Defect Details:
{defect_summary}

Spatial Zone Damage (4×4 Grid):
{zone_summary}

Progression:
{progression_text}
"""


def _build_prompt(engineering_context: str) -> str:
    """Build the Groq report prompt for image pavement analysis."""
    return f"""You are preparing a professional road-condition inspection report from the supplied road image and automated pavement-analysis data.

Your response must contain ONLY the final engineering report.

The report is intended for a municipal road-maintenance department.

============================================================
CORE REQUIREMENT & PROHIBITIONS
============================================================

The generation rules below are internal constraints. NEVER include, reference, or expose them in your output.

Do NOT output:
- chain-of-thought, reasoning, analysis, planning, scratch work
- self-correction, internal deliberation, decision-making process
- cost calculations, assumptions, or alternative estimates
- prompt interpretation, rule checking, criteria verification, or instruction checking
- meta-commentary, drafting notes, or progress statements
- XML reasoning tags (<think>, <analysis>, <reasoning>)

Do NOT write statements such as:
- "Rule 1 says..."
- "According to Rule 2..."
- "Following the required format..."
- "Based on the instructions..."
- "According to the prompt..."
- "The prompt requires..."
- "The prompt says..."
- "Checking the criteria..."
- "Let's analyze..."
- "First, I will..."
- "Drafting..."
- "Self-correction..."
- "Internal calculation..."
- "Cost logic..."
- "Reasoning..."
- "Analysis..."
- "Thinking Process..."
- "I analyzed..."
- "I determined..."
- "I think..."
- "I believe..."
- "I will..."
- "Let's..."
- "Wait, looking at..."
- "After reviewing..."
- "(Since priority is...)"
- "implied as..."

Do NOT answer: "How did the AI arrive at this conclusion?"
Write with direct, objective, natural third-person engineering language (e.g., "The inspected road is classified as FAIR with a Pavement Health Score of 55.85/100.", "Pothole #1 in Zone Z10 represents the primary structural hazard requiring scheduled repair.").

============================================================
VERY IMPORTANT OUTPUT CONTRACT
============================================================

The response MUST start directly with:
## ROAD HEALTH STATUS

There must be NOTHING before this heading.

The response MUST end directly with:
*This report is generated using AI-assisted pavement distress analysis. Field verification by a licensed road engineer is recommended before executing major repair works.*

There must be NOTHING after this verification note.

The output between those two boundaries is the FINAL REPORT.

============================================================
DATA INTEGRITY & AUTHORITATIVE DATA
============================================================

Use the supplied engineering data as the authoritative numerical information:
- Use exact values. Do NOT modify them or fabricate detections, severity values, Pavement Health Score, zone scores, or progression values.
- Use the exact road condition supplied (VERY GOOD, GOOD, FAIR, POOR, VERY POOR).
- Maintenance priority & required action mapping:
  HIGH or CRITICAL → IMMEDIATE
  MEDIUM → SOON
  LOW or ROUTINE → ROUTINE MONITORING
  Output only the resulting action without explaining the mapping rule or writing "(Since priority is ...)".

============================================================
ENGINEERING DATA
============================================================

{engineering_context}

============================================================
FINAL REPORT FORMAT
============================================================

## ROAD HEALTH STATUS

**Condition:** [VERY GOOD / GOOD / FAIR / POOR / VERY POOR]

**Pavement Health Score:** [value]/100

**Maintenance Priority:** [LOW / MEDIUM / HIGH / CRITICAL]

**Required Action:** [IMMEDIATE / SOON / ROUTINE MONITORING]

[One concise professional engineering summary paragraph.]

## CRITICAL DAMAGE IDENTIFIED

[Describe the most important detected defect using the supplied defect ID, severity, confidence, area, and zone information in natural engineering terms.]

## POTHOLE REPAIR SPECIFICATION

- **Priority Defect:** [defect or None]
- **Repair Urgency:** [Immediate / Scheduled / Routine / N/A]
- **Recommended Method:** [method or N/A]
- **Supplementary Inspection:** [recommendation or N/A]

## CRACK ASSESSMENT

[Concise engineering assessment of detected and visible cracking.]

## CRACK-TO-POTHOLE DETERIORATION RISK

**Risk Level:** [LOW / MEDIUM / HIGH / INSUFFICIENT EVIDENCE]

[1–2 concise engineering sentences.]

## MAINTENANCE ACTION PLAN

**IMMEDIATE (0–48 hours)**

- [Required action or "No immediate intervention required."]

**SHORT-TERM (1–2 weeks)**

- [Specific repair and inspection recommendations.]

**PREVENTIVE (1–3 months)**

- [Monitoring and preventive maintenance recommendations.]

## FINAL ENGINEERING DECISION

**Road Status:** [condition]

**Priority:** [priority]

**Action:** [required action]

**Primary Concern:** [single most important concern]

---

*This report is generated using AI-assisted pavement distress analysis. Field verification by a licensed road engineer is recommended before executing major repair works.*
"""


def clean_groq_report(text: str) -> str:
    """
    Convert the raw Groq response into the final user-facing
    clean engineering report for image analysis.
    """
    if not text:
        return ""

    text = str(text).strip()

    # 1. Remove complete reasoning blocks
    for tag in ["think", "analysis", "reasoning"]:
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(rf"</?{tag}>", "", text, flags=re.IGNORECASE)

    # 2. If drafting or refinement markers exist, discard earlier drafts
    draft_markers = [
        "Refining the content:",
        "Refining the report:",
        "Refined Report:",
        "Final Report:",
        "Final Output:",
        "Draft Report:",
        "Drafting text:",
    ]
    for dm in draft_markers:
        pos = text.rfind(dm)
        if pos != -1:
            text = text[pos + len(dm):]

    # 3. Find the actual report start marker (use last occurrence if duplicated in draft)
    marker = "## ROAD HEALTH STATUS"
    marker_position = text.rfind(marker)
    if marker_position >= 0:
        text = text[marker_position:]
    else:
        alt_marker = "ROAD HEALTH STATUS"
        alt_pos = text.rfind(alt_marker)
        if alt_pos >= 0:
            text = "## " + text[alt_pos:]

    # 4. Remove anything after the engineering verification note
    verification_marker = (
        "*This report is generated using AI-assisted pavement distress analysis."
    )
    verification_position = text.find(verification_marker)
    if verification_position >= 0:
        text = text[:verification_position] + (
            "*This report is generated using AI-assisted pavement distress analysis. "
            "Field verification by a licensed road engineer is recommended before "
            "executing major repair works.*"
        )

    # 5. Remove forbidden reasoning phrases, drafting notes, and meta-rule statements
    forbidden_phrases = [
        "Thinking Process:",
        "Self-Correction:",
        "Self-Correction on Cost:",
        "Draft Report Sections:",
        "Drafting the Report:",
        "Review against Constraints:",
        "Final Review:",
        "Mental check:",
        "Refinement:",
        "Deconstruct and Analyze Input Data:",
        "Analyze Visual Image:",
        "Cost Logic:",
        "Cost Logic (Internal - do not output):",
        "Internal reasoning:",
        "Internal calculation:",
        "the prompt says",
        "prompt says",
        "according to the prompt",
        "as per the prompt",
        "prompt rule",
        "stick to the rule",
        "let's assume",
        "let us assume",
        "i should mention",
        "need to estimate",
        "rough estimate",
        "rough guess",
        "this confirms",
        "is the big one",
        "focus on pothole",
    ]

    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if any(phrase.lower() in stripped.lower() for phrase in forbidden_phrases):
            continue
        
        # Strip inline annotations like "(Since priority is MEDIUM)" or "(implied as ...)"
        cleaned_line = re.sub(r"\(Since[^\)]*\)", "", line, flags=re.IGNORECASE)
        cleaned_line = re.sub(r"\(Wait,[^\)]*\)", "", cleaned_line, flags=re.IGNORECASE)
        cleaned_line = re.sub(r"\(implied as[^\)]*\)", "", cleaned_line, flags=re.IGNORECASE)
        cleaned_line = re.sub(r"\(The prompt says[^\)]*\)", "", cleaned_line, flags=re.IGNORECASE)
        
        cleaned_lines.append(cleaned_line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _call_groq(headers, content, model, max_tokens=2500):
    """Send a request to Groq and return the raw report text with system constraints."""
    system_prompt = (
        "You are a licensed municipal pavement engineering inspector. "
        "Output ONLY the final formal engineering report in clean markdown. "
        "NEVER write thinking process, draft notes, self-corrections, rule explanations, or phrases like 'the prompt says', 'let us assume', 'wait'. "
        "Output direct, final engineering statements only."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        GROQ_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Error code: {response.status_code} - {response.text}"
        )

    response_json = response.json()

    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(
            f"Could not extract Groq report: {response_json}"
        )


# Text-only fallback model if vision model is unavailable
FALLBACK_MODEL = "openai/gpt-oss-120b"


def generate_llm_report(features, zone_damage, progression_records, report_summary,
                         predicted_img_bgr=None):
    """
    Generate an AI-powered final report using Groq multimodal API.
    Sends the prediction image + engineering data to Groq vision model.
    Cleans the raw response with clean_groq_report() before returning.
    Falls back to resilient text models if vision fails.
    Returns the cleaned final report text (markdown formatted).
    """
    engineering_context = _build_engineering_context(
        features, zone_damage, progression_records, report_summary
    )
    prompt_text = _build_prompt(engineering_context)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # 1. Try multimodal (vision) first
    if predicted_img_bgr is not None:
        try:
            image_url = _prepare_image_base64(predicted_img_bgr)
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            raw_report = _call_groq(headers, content, GROQ_MODEL, max_tokens=3000)
            cleaned = clean_groq_report(raw_report)
            if cleaned and "## ROAD HEALTH STATUS" in cleaned:
                return cleaned
        except Exception:
            pass

    # 2. Text-only candidate model cascade
    candidate_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "groq/compound-mini",
        GROQ_MODEL,
        FALLBACK_MODEL,
    ]

    last_error = None
    content = [{"type": "text", "text": prompt_text}]

    for model_name in candidate_models:
        try:
            raw_report = _call_groq(headers, content, model_name, max_tokens=2500)
            cleaned = clean_groq_report(raw_report)
            if cleaned and "## ROAD HEALTH STATUS" in cleaned:
                return cleaned
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise last_error

    return ""


# ===============================================================
# 2. VIDEO REPORT PIPELINE (DEDICATED — SEPARATE FROM IMAGE)
# ===============================================================

def _build_video_engineering_context(video_results: dict) -> str:
    """Build structured engineering context text from aggregated video inspection data."""
    meta = video_results["metadata"]
    sm = video_results["summary"]
    temporal = video_results["temporal"]
    tracked_inv = video_results.get("tracked_inventory", [])
    lat_analysis = video_results.get("lateral_wheelpath_analysis", {})
    seg_profile = video_results.get("segment_health_profile", [])
    roughness = video_results.get("surface_roughness", {})

    # Tracked defect inventory lines
    inv_lines = []
    for d in tracked_inv:
        inv_lines.append(
            f"  Track {d['track_id']} ({d['class_name']}): Sighted {d['first_seen']}–{d['last_seen']} "
            f"({d['duration_sec']:.1f}s, {d['persistence_frames']} frames) | Position: {d['wheelpath_position']} | "
            f"MaxSeverity: {d['max_severity']:.3f} | Conf: {d['mean_confidence']:.3f} | Risk: {d['risk_rating']}"
        )
    inv_text = "\n".join(inv_lines) if inv_lines else "  No persistent physical defects tracked."

    # Lateral wheelpath lines
    wp_lines = []
    if lat_analysis:
        for wp_key, wp_name in [
            ("left_wheelpath", "Left Wheelpath (LWP)"),
            ("center_lane", "Center Lane (CL)"),
            ("right_wheelpath", "Right Wheelpath (RWP)"),
        ]:
            wp_d = lat_analysis.get(wp_key, {})
            wp_lines.append(
                f"  {wp_name}: {wp_d.get('total_defects', 0)} defects ({wp_d.get('potholes', 0)}P / {wp_d.get('cracks', 0)}C) | "
                f"MaxSeverity={wp_d.get('max_severity', 0.0):.3f} | Status={wp_d.get('status', 'NOMINAL')}"
            )
    wp_text = "\n".join(wp_lines) if wp_lines else "  No lateral wheelpath distress."

    # Segment progression lines
    seg_lines = []
    for s in seg_profile[:8]:
        seg_lines.append(
            f"  {s['segment_id']} ({s['time_range']}): PHS={s['segment_phs']:.1f}/100 | Condition={s['condition']} | "
            f"Defects: {s['active_pothole_frames']}P-frames, {s['active_crack_frames']}C-frames | Rec: {s['action_recommendation']}"
        )
    seg_text = "\n".join(seg_lines) if seg_lines else "  No segment telemetry available."

    # Timeline excerpt
    timeline_lines = []
    for item in temporal["timeline"][:12]:
        timeline_lines.append(
            f"  {item['timestamp']} (Frame {item['frame']}): {item['defects']} | "
            f"Damage={item['damage_score']:.3f} | PHS={item['phs']:.1f}"
        )
    timeline_text = "\n".join(timeline_lines) if timeline_lines else "  No significant damage events detected across timeline."

    return f"""
Video Filename           : {meta['file_name']}
Video Duration           : {meta['duration_str']} ({meta['duration_sec']:.1f} seconds)
Video Resolution         : {meta['resolution']} @ {meta['fps']} FPS
Total Video Frames       : {meta['total_frames']}
Analyzed Sampled Frames  : {meta['analyzed_frames']} (sampled every {meta['sample_interval']} frames)
Video Health Score (PHS) : {sm['video_pavement_health_score']:.2f}/100
Road Condition           : {sm['road_condition']}
Maintenance Priority     : {sm['maintenance_priority']}
Overall Damage Score     : {sm['video_damage_score']:.4f}

UNIQUE DISTRESS SUMMARY (De-duplicated Multi-Frame Tracking):
- Confirmed Unique Potholes : {sm['unique_potholes']} (tracked across {sm['total_pothole_observations']} frame sightings)
- Confirmed Unique Cracks   : {sm['unique_cracks']} (tracked across {sm['total_crack_observations']} frame sightings)
- Total Unique Defect Count : {sm['total_unique_defects']}
- Distress Encounter Rate   : {sm.get('distress_rate_per_min', 0.0):.1f} Defects/Minute ({sm.get('active_distress_duration_pct', 0.0):.1f}% active video duration)

LATERAL WHEELPATH & RIDE QUALITY TELEMETRY:
- Wheelpath Hazard Ratio    : {sm.get('wheelpath_hazard_ratio', 0.0):.1f}% ({sm.get('dominant_distress_lane', 'CL')})
- Estimated Roughness (IRI) : {sm.get('estimated_iri_m_km', 2.0):.2f} m/km ({sm.get('ride_quality', 'SMOOTH')})
- Ride Disturbance Index    : {sm.get('ride_disturbance_index', 0.0):.1f} / 10.0
{wp_text}

UNIQUE TRACKED DEFECT INVENTORY:
{inv_text}

CHRONOLOGICAL ROAD SEGMENT DETERIORATION PROFILE:
{seg_text}

DETECTION TIMELINE EXCERPT:
{timeline_text}
"""


def _build_video_prompt(video_context: str) -> str:
    """Build the Groq video report prompt according to the exact engineering report specification."""
    return f"""You are preparing a professional municipal road-condition inspection report from the supplied road video inspection data and representative frame.

Your response must contain ONLY the final engineering report.

The report is intended for a municipal road-maintenance department.

============================================================
CORE REQUIREMENT & PROHIBITIONS
============================================================

The generation rules below are internal constraints. NEVER include, reference, or expose them in your output.

Do NOT output:
- chain-of-thought, reasoning, analysis, planning, scratch work
- self-correction, internal deliberation, decision-making process
- cost calculations, assumptions, or alternative estimates
- prompt interpretation, rule checking, criteria verification, or instruction checking
- meta-commentary, drafting notes, or progress statements
- XML reasoning tags (<think>, <analysis>, <reasoning>)

Do NOT write statements such as:
- "Rule 1 says..."
- "According to Rule 2..."
- "Following the required format..."
- "Based on the instructions..."
- "According to the prompt..."
- "The prompt requires..."
- "Checking the criteria..."
- "Let's analyze..."
- "First, I will..."
- "Drafting..."
- "Self-correction..."
- "Internal calculation..."
- "Cost logic..."
- "Reasoning..."
- "Analysis..."
- "Thinking Process..."
- "I analyzed..."
- "I determined..."
- "I think..."
- "I believe..."
- "I will..."
- "Let's..."
- "After reviewing..."

Do NOT answer: "How did the AI arrive at this conclusion?"
Write with direct, objective, third-person engineering language (e.g., "The corridor is classified as POOR.", "Multi-frame tracking confirmed 21 unique potholes.", "Immediate corrective action is recommended.").

============================================================
VERY IMPORTANT OUTPUT CONTRACT
============================================================

The response MUST start directly with:
## VIDEO ROAD HEALTH STATUS

There must be NOTHING before this heading.

The response MUST end directly with:
*This report is generated using AI-assisted pavement distress analysis. Field verification by a licensed road engineer is recommended before executing major repair works.*

There must be NOTHING after this verification note.

The output between those two boundaries is the FINAL REPORT.

============================================================
DATA INTEGRITY & AUTHORITATIVE DATA
============================================================

Use the supplied engineering data as the authoritative numerical information:
- Use exact values. Do NOT modify them or fabricate detections, unique defects, persistence durations, severity values, IRI, Pavement Health Score, zone scores, or progression values.
- If unique tracking data is supplied, report unique defect counts and IDs (e.g., P-01, C-01) and persistence durations exactly. Do not count repeated frame sightings as unique defects.
- Use the exact road condition supplied (VERY GOOD, GOOD, FAIR, POOR, VERY POOR).
- Maintenance priority & required action mapping:
  HIGH or CRITICAL → IMMEDIATE
  MEDIUM → SOON
  LOW or ROUTINE → ROUTINE MONITORING
  Output only the resulting action without explaining the mapping rule.
- Report exact IRI roughness if supplied (e.g., "**IRI:** 8.30 m/km — HAZARDOUS").
- Report exact Lateral Wheelpath (LWP, CL, RWP) defect counts and status if supplied.
- Report concise Chronological Segment Progression using supplied segment data without showing calculations.

============================================================
SUPPLIED VIDEO ENGINEERING DATA
============================================================

{video_context}

============================================================
FINAL REPORT FORMAT
============================================================

## VIDEO ROAD HEALTH STATUS

**Road Condition:** [VERY GOOD / GOOD / FAIR / POOR / VERY POOR]

**Pavement Health Score:** [value]/100

**Maintenance Priority:** [LOW / MEDIUM / HIGH / CRITICAL]

**Required Action:** [IMMEDIATE / SOON / ROUTINE MONITORING]

**IRI:** [value and classification if available]

**Video Duration:** [duration]

**Analyzed Frames:** [count]

[Write ONE concise professional paragraph summarizing the overall road condition.]

## UNIQUE ROAD DEFECT INVENTORY

[Describe confirmed unique physical defects from the tracking data including unique pothole count, unique crack count, important defect IDs, persistence duration, severity, and relevant zone/lane.]

## LATERAL WHEELPATH & RIDE QUALITY

[Summarize LWP, CL and RWP defect counts and hazard levels when provided by the engineering data. Include IRI and its supplied classification when available.]

## CHRONOLOGICAL SEGMENT PROGRESSION

[Summarize the condition of the video segments using the supplied segment data. Focus on where deterioration is concentrated without exposing calculations.]

## CRITICAL DAMAGE IDENTIFIED

[Identify the most important physical defects requiring attention. Reference actual defect IDs, severity, location and persistence where available.]

## MAINTENANCE ACTION PLAN

**IMMEDIATE (0–48 hours)**

- [Immediate engineering action.]
- [Traffic control if supported by the supplied hazard information.]

**SHORT-TERM (1–2 weeks)**

- [Targeted repairs and inspections.]

**PREVENTIVE (1–3 months)**

- [Monitoring, drainage assessment and rehabilitation planning.]

## FINAL ENGINEERING DECISION

**Road Status:** [condition]

**Priority:** [priority]

**Action:** [action]

**Primary Concern:** [single most important concern]

---

*This report is generated using AI-assisted pavement distress analysis. Field verification by a licensed road engineer is recommended before executing major repair works.*
"""


def clean_video_groq_report(text: str) -> str:
    """Return only the final user-facing video engineering report."""

    if not text:
        return ""

    text = str(text).strip()

    # Remove complete reasoning blocks.
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(
        r"<analysis>.*?</analysis>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(
        r"<reasoning>.*?</reasoning>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove orphan reasoning tags.
    text = re.sub(
        r"</?(think|analysis|reasoning)>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Find the actual report beginning.
    start_marker = "## VIDEO ROAD HEALTH STATUS"

    start = text.find(start_marker)

    if start != -1:
        text = text[start:]

    # Cut anything after the final verification statement.
    end_marker = (
        "*This report is generated using AI-assisted pavement distress analysis."
    )

    end = text.find(end_marker)

    if end != -1:
        text = text[:end] + (
            "*This report is generated using AI-assisted pavement distress analysis. "
            "Field verification by a licensed road engineer is recommended before "
            "executing major repair works.*"
        )

    # Remove known reasoning headings if any survive.
    forbidden_lines = [
        "Thinking Process:",
        "Deconstruct and Analyze",
        "Analyze the Request:",
        "Analyze Visual Image:",
        "Draft Report:",
        "Drafting text:",
        "Draft Report Sections:",
        "Self-Correction:",
        "Self-Correction on Cost:",
        "Cost Logic:",
        "Cost Logic (Internal - do not output):",
        "Mental check:",
        "Refinement:",
        "Final Review:",
        "Review against Constraints:",
        "Internal reasoning:",
        "Internal calculation:",
        "Reasoning:",
        "Analysis:",
    ]

    cleaned_lines = []

    for line in text.splitlines():

        stripped = line.strip()

        if any(
            phrase.lower() in stripped.lower()
            for phrase in forbidden_lines
        ):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Remove excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def generate_video_llm_report(video_results: dict, peak_annotated_img_bgr=None) -> str:
    """
    Generate an AI-powered final report for video inspection data.
    Uses multimodal vision if peak frame is available, otherwise text-only fallback.
    Returns the cleaned final video report text.
    """
    video_context = _build_video_engineering_context(video_results)
    prompt_text = _build_video_prompt(video_context)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    # 1. Try multimodal vision with peak annotated frame
    if peak_annotated_img_bgr is not None:
        try:
            image_url = _prepare_image_base64(peak_annotated_img_bgr)
            content = [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            raw_report = _call_groq(headers, content, GROQ_MODEL, max_tokens=3000)
            cleaned = clean_video_groq_report(raw_report)
            if cleaned and "## VIDEO ROAD HEALTH STATUS" in cleaned:
                return cleaned
        except Exception:
            pass

    # 2. Text model cascading fallback
    candidate_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "groq/compound-mini",
        GROQ_MODEL,
        FALLBACK_MODEL,
    ]

    last_error = None
    content = [{"type": "text", "text": prompt_text}]

    for model_name in candidate_models:
        try:
            raw_report = _call_groq(headers, content, model_name, max_tokens=2500)
            cleaned = clean_video_groq_report(raw_report)
            if cleaned and "## VIDEO ROAD HEALTH STATUS" in cleaned:
                return cleaned
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise last_error

    return ""
