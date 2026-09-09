"""
RoadHealth AI — Video PDF Report Generator
Generates a multi-page static engineering report for video analysis using ReportLab.
Features unique defect inventory tracking, lateral wheelpath analysis, and chronological segment profiles.
NOTE: NEVER embeds video binary data; embeds only static representative frames, analytics charts, and data tables.
"""

import io
import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.platypus.flowables import KeepTogether

from config import MODEL_INFO

# ── Colours ────────────────────────────────────────────────────
BRAND_DARK  = colors.HexColor("#050816")
BRAND_BLUE  = colors.HexColor("#0f3460")
BRAND_ACCENT = colors.HexColor("#3B82F6")
BRAND_CYAN  = colors.HexColor("#22D3EE")
BRAND_LIGHT = colors.HexColor("#f8f9fa")
WHITE = colors.white
BLACK = colors.black

PAGE_W, PAGE_H = A4


def _cv2_to_reportlab_image(cv2_img, max_width=6.5*inch, max_height=4.2*inch):
    """Convert a BGR OpenCV image to a ReportLab Image flowable."""
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    img_w, img_h = pil.size
    ratio = min(max_width / img_w, max_height / img_h)
    return Image(buf, width=img_w * ratio, height=img_h * ratio)


def _make_styles():
    """Create custom paragraph styles for the report."""
    base = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "VTitle", parent=base["Title"],
        fontSize=22, textColor=BRAND_DARK, spaceAfter=4,
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    )
    styles["subtitle"] = ParagraphStyle(
        "VSubtitle", parent=base["Title"],
        fontSize=11, textColor=BRAND_BLUE, spaceAfter=12,
        alignment=TA_CENTER, fontName="Helvetica",
    )
    styles["heading"] = ParagraphStyle(
        "VHeading", parent=base["Heading1"],
        fontSize=13, textColor=BRAND_DARK, spaceBefore=9, spaceAfter=5,
        fontName="Helvetica-Bold",
    )
    styles["subheading"] = ParagraphStyle(
        "VSubheading", parent=base["Heading2"],
        fontSize=9.5, textColor=BRAND_BLUE, spaceBefore=5, spaceAfter=3,
        fontName="Helvetica-Bold",
    )
    styles["body"] = ParagraphStyle(
        "VBody", parent=base["Normal"],
        fontSize=8.5, textColor=BRAND_DARK, leading=11.5,
        alignment=TA_LEFT, fontName="Helvetica",
    )
    styles["table_cell"] = ParagraphStyle(
        "VCell", parent=base["Normal"],
        fontSize=7.8, leading=9.5, fontName="Helvetica",
    )
    styles["table_header"] = ParagraphStyle(
        "VTH", parent=base["Normal"],
        fontSize=8, leading=10, fontName="Helvetica-Bold", textColor=WHITE,
    )
    styles["meta"] = ParagraphStyle(
        "VMeta", parent=base["Normal"],
        fontSize=8, textColor=colors.HexColor("#64748B"), leading=10,
        alignment=TA_CENTER, fontName="Helvetica",
    )
    return styles


def generate_video_pdf_report(
    video_results: dict,
    chart_imgs_list: list = None,
    ai_report_text: str = None,
) -> bytes:
    """
    Generate a complete multi-page static engineering report for video analysis.
    Returns the PDF as raw bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16*mm, rightMargin=16*mm,
        topMargin=16*mm, bottomMargin=16*mm,
    )

    styles = _make_styles()
    elements = []
    meta = video_results["metadata"]
    sm = video_results["summary"]
    temporal = video_results["temporal"]
    tracked_inv = video_results.get("tracked_inventory", [])
    lat_analysis = video_results.get("lateral_wheelpath_analysis", {})
    seg_profile = video_results.get("segment_health_profile", [])
    roughness = video_results.get("surface_roughness", {})
    rep_frames = video_results.get("representative_frames", {})

    # ===========================================================
    # HEADER & METADATA
    # ===========================================================
    elements.append(Paragraph("ROADHEALTH AI — VIDEO INSPECTION REPORT", styles["title"]))
    elements.append(Paragraph(
        "Explainable AI Framework for Multi-Frame Road Distress Tracking &amp; Video Infrastructure Intelligence",
        styles["subtitle"]
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=BRAND_ACCENT, spaceAfter=8))

    gen_time = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    elements.append(Paragraph(
        f"Generated: {gen_time} &nbsp;|&nbsp; Model: <b>{MODEL_INFO['name']}</b> &nbsp;|&nbsp; "
        f"Tracking Engine: <b>RoadDefectTracker Multi-Frame IoU</b>",
        styles["meta"]
    ))
    elements.append(Spacer(1, 8))

    # ===========================================================
    # EXECUTIVE SUMMARY CARDS TABLE
    # ===========================================================
    phs_val = f"{sm['video_pavement_health_score']:.1f} / 100"
    iri_val = f"{sm.get('estimated_iri_m_km', 2.0):.2f} m/km"
    summary_data = [
        [
            Paragraph("<b>Video Health Score</b>", styles["table_header"]),
            Paragraph("<b>Road Condition</b>", styles["table_header"]),
            Paragraph("<b>Maintenance Priority</b>", styles["table_header"]),
            Paragraph("<b>Roughness (IRI Proxy)</b>", styles["table_header"]),
        ],
        [
            Paragraph(f"<font size=11><b>{phs_val}</b></font>", styles["table_cell"]),
            Paragraph(f"<font size=10><b>{sm['road_condition']}</b></font>", styles["table_cell"]),
            Paragraph(f"<font size=10><b>{sm['maintenance_priority']}</b></font>", styles["table_cell"]),
            Paragraph(f"<font size=10><b>{iri_val}</b><br/>{sm.get('ride_quality', 'SMOOTH')}</font>", styles["table_cell"]),
        ],
    ]
    t_summary = Table(summary_data, colWidths=[1.6*inch, 1.6*inch, 1.8*inch, 1.6*inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BRAND_BLUE),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0, 1), (-1, 1), BRAND_LIGHT),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_summary)
    elements.append(Spacer(1, 10))

    # ===========================================================
    # VIDEO TECHNICAL SPECIFICATIONS TABLE
    # ===========================================================
    elements.append(Paragraph("1. Video Stream Specifications &amp; Defect Counts", styles["heading"]))
    spec_data = [
        [
            Paragraph("<b>Property</b>", styles["table_header"]),
            Paragraph("<b>Value</b>", styles["table_header"]),
            Paragraph("<b>Property</b>", styles["table_header"]),
            Paragraph("<b>Value</b>", styles["table_header"]),
        ],
        [
            Paragraph("Video Filename", styles["table_cell"]),
            Paragraph(str(meta['file_name']), styles["table_cell"]),
            Paragraph("Sampling Frequency", styles["table_cell"]),
            Paragraph(f"1 frame every {meta['sample_interval']} frames", styles["table_cell"]),
        ],
        [
            Paragraph("Video Duration", styles["table_cell"]),
            Paragraph(f"{meta['duration_str']} ({meta['duration_sec']:.1f}s)", styles["table_cell"]),
            Paragraph("Total Video Frames", styles["table_cell"]),
            Paragraph(f"{meta['total_frames']:,} frames", styles["table_cell"]),
        ],
        [
            Paragraph("Resolution &amp; FPS", styles["table_cell"]),
            Paragraph(f"{meta['resolution']} @ {meta['fps']} FPS", styles["table_cell"]),
            Paragraph("Sampled Frames Analyzed", styles["table_cell"]),
            Paragraph(f"{meta['analyzed_frames']} frames", styles["table_cell"]),
        ],
        [
            Paragraph("Unique Potholes", styles["table_cell"]),
            Paragraph(f"<b>{sm['unique_potholes']}</b> ({sm['total_pothole_observations']} frame sightings)", styles["table_cell"]),
            Paragraph("Unique Cracks", styles["table_cell"]),
            Paragraph(f"<b>{sm['unique_cracks']}</b> ({sm['total_crack_observations']} frame sightings)", styles["table_cell"]),
        ],
        [
            Paragraph("Wheelpath Hazard Ratio", styles["table_cell"]),
            Paragraph(f"<b>{sm.get('wheelpath_hazard_ratio', 0.0):.1f}%</b> ({sm.get('dominant_distress_lane', 'CL')})", styles["table_cell"]),
            Paragraph("Distress Encounter Rate", styles["table_cell"]),
            Paragraph(f"<b>{sm.get('distress_rate_per_min', 0.0):.1f} DPM</b> ({sm.get('active_distress_duration_pct', 0.0):.1f}% active)", styles["table_cell"]),
        ],
    ]
    t_specs = Table(spec_data, colWidths=[1.6*inch, 1.7*inch, 1.6*inch, 1.7*inch])
    t_specs.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BRAND_DARK),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, BRAND_LIGHT]),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))
    elements.append(t_specs)
    elements.append(Spacer(1, 10))

    # ===========================================================
    # REPRESENTATIVE STATIC ANNOTATED SNAPSHOT
    # ===========================================================
    if rep_frames.get("peak_annotated") is not None:
        elements.append(Paragraph("2. Critical Representative Distress Snapshot", styles["heading"]))
        elements.append(Paragraph(
            f"Peak Distress Frame Snapshot — Frame #{rep_frames['peak_info']['frame']} at "
            f"Timestamp {rep_frames['peak_info']['timestamp']} "
            f"(Damage Score: {rep_frames['peak_info']['damage_score']:.3f}, Active Potholes: {rep_frames['peak_info']['potholes']}, Active Cracks: {rep_frames['peak_info']['cracks']})",
            styles["body"]
        ))
        elements.append(Spacer(1, 4))
        try:
            img_flowable = _cv2_to_reportlab_image(rep_frames["peak_annotated"], max_width=6.5*inch, max_height=3.0*inch)
            elements.append(img_flowable)
        except Exception:
            pass
        elements.append(Spacer(1, 10))

    # ===========================================================
    # LATERAL WHEELPATH & IMPACT BREAKDOWN
    # ===========================================================
    elements.append(Paragraph("3. Lateral Wheelpath &amp; Vehicular Impact Distribution", styles["heading"]))
    wp_rows = [
        [
            Paragraph("<b>Lateral Lane Position</b>", styles["table_header"]),
            Paragraph("<b>Total Defects</b>", styles["table_header"]),
            Paragraph("<b>Potholes</b>", styles["table_header"]),
            Paragraph("<b>Cracks</b>", styles["table_header"]),
            Paragraph("<b>Peak Severity</b>", styles["table_header"]),
            Paragraph("<b>Impact Status</b>", styles["table_header"]),
        ]
    ]
    if lat_analysis:
        for wp_key, wp_name in [
            ("left_wheelpath", "Left Wheelpath (LWP)"),
            ("center_lane", "Center Lane (CL)"),
            ("right_wheelpath", "Right Wheelpath (RWP)"),
        ]:
            wp_d = lat_analysis.get(wp_key, {})
            wp_rows.append([
                Paragraph(f"<b>{wp_name}</b>", styles["table_cell"]),
                Paragraph(str(wp_d.get("total_defects", 0)), styles["table_cell"]),
                Paragraph(str(wp_d.get("potholes", 0)), styles["table_cell"]),
                Paragraph(str(wp_d.get("cracks", 0)), styles["table_cell"]),
                Paragraph(f"{wp_d.get('max_severity', 0.0):.3f}", styles["table_cell"]),
                Paragraph(str(wp_d.get("status", "NOMINAL")), styles["table_cell"]),
            ])

        t_wp = Table(wp_rows, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 1.0*inch, 1.1*inch])
        t_wp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BRAND_BLUE),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, BRAND_LIGHT]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(t_wp)
    elements.append(Spacer(1, 10))

    # ===========================================================
    # UNIQUE TRACKED DEFECT INVENTORY TABLE
    # ===========================================================
    elements.append(Paragraph("4. Unique Physical Defect Inventory (De-duplicated)", styles["heading"]))
    if tracked_inv:
        inv_rows = [
            [
                Paragraph("<b>Track ID</b>", styles["table_header"]),
                Paragraph("<b>Class</b>", styles["table_header"]),
                Paragraph("<b>First Seen</b>", styles["table_header"]),
                Paragraph("<b>Last Seen</b>", styles["table_header"]),
                Paragraph("<b>Dwell Time</b>", styles["table_header"]),
                Paragraph("<b>Wheelpath Position</b>", styles["table_header"]),
                Paragraph("<b>Max Severity</b>", styles["table_header"]),
                Paragraph("<b>Risk Rating</b>", styles["table_header"]),
            ]
        ]
        for d in tracked_inv[:15]:
            inv_rows.append([
                Paragraph(f"<b>{d['track_id']}</b>", styles["table_cell"]),
                Paragraph(d["class_name"], styles["table_cell"]),
                Paragraph(d["first_seen"], styles["table_cell"]),
                Paragraph(d["last_seen"], styles["table_cell"]),
                Paragraph(f"{d['duration_sec']:.1f}s ({d['persistence_frames']}f)", styles["table_cell"]),
                Paragraph(d["wheelpath_position"], styles["table_cell"]),
                Paragraph(f"{d['max_severity']:.3f}", styles["table_cell"]),
                Paragraph(f"<b>{d['risk_rating']}</b>", styles["table_cell"]),
            ])

        t_inv = Table(inv_rows, colWidths=[0.8*inch, 0.8*inch, 0.7*inch, 0.7*inch, 1.0*inch, 1.2*inch, 0.7*inch, 0.7*inch])
        t_inv.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BRAND_DARK),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, BRAND_LIGHT]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(t_inv)
    else:
        elements.append(Paragraph("No tracked physical distresses observed across the video.", styles["body"]))

    elements.append(Spacer(1, 10))

    # ===========================================================
    # CHRONOLOGICAL ROAD SEGMENT DETERIORATION PROFILE
    # ===========================================================
    if seg_profile:
        elements.append(Paragraph("5. Chronological Road Segment Deterioration Profile", styles["heading"]))
        seg_rows = [
            [
                Paragraph("<b>Segment</b>", styles["table_header"]),
                Paragraph("<b>Time Interval</b>", styles["table_header"]),
                Paragraph("<b>Segment RHS</b>", styles["table_header"]),
                Paragraph("<b>Condition</b>", styles["table_header"]),
                Paragraph("<b>Active Defect Frames</b>", styles["table_header"]),
                Paragraph("<b>Maintenance Recommendation</b>", styles["table_header"]),
            ]
        ]
        for s in seg_profile[:10]:
            seg_rows.append([
                Paragraph(f"<b>{s['segment_id']}</b>", styles["table_cell"]),
                Paragraph(s["time_range"], styles["table_cell"]),
                Paragraph(f"{s['segment_phs']:.1f}", styles["table_cell"]),
                Paragraph(f"<b>{s['condition']}</b>", styles["table_cell"]),
                Paragraph(f"{s['active_pothole_frames']}P / {s['active_crack_frames']}C", styles["table_cell"]),
                Paragraph(s["action_recommendation"], styles["table_cell"]),
            ])

        t_seg = Table(seg_rows, colWidths=[0.9*inch, 1.1*inch, 0.9*inch, 0.9*inch, 1.2*inch, 1.6*inch])
        t_seg.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BRAND_BLUE),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, BRAND_LIGHT]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        elements.append(t_seg)
        elements.append(Spacer(1, 10))

    # ===========================================================
    # CHARTS (IF PROVIDED)
    # ===========================================================
    if chart_imgs_list:
        elements.append(Paragraph("6. Quantitative Progression Charts", styles["heading"]))
        for c_img in chart_imgs_list[:2]:
            try:
                buf_c = io.BytesIO(c_img)
                pil_c = PILImage.open(buf_c)
                w, h = pil_c.size
                ratio = min(6.5*inch / w, 2.8*inch / h)
                buf_c.seek(0)
                elements.append(Image(buf_c, width=w * ratio, height=h * ratio))
                elements.append(Spacer(1, 5))
            except Exception:
                pass

    # ===========================================================
    # AI ENGINEERING FINAL REPORT (GROQ)
    # ===========================================================
    if ai_report_text:
        elements.append(PageBreak())
        elements.append(Paragraph("7. AI-Powered Video Engineering Report", styles["heading"]))
        elements.append(Paragraph("Synthesized via Groq AI &amp; YOLO11m-P2P3-CBAM Multimodal Reasoning", styles["subheading"]))
        elements.append(Spacer(1, 6))

        for block in ai_report_text.split("\n\n"):
            block_s = block.strip()
            if not block_s:
                continue
            if block_s.startswith("## "):
                elements.append(Paragraph(f"<b>{block_s[3:]}</b>", styles["heading"]))
            elif block_s.startswith("### "):
                elements.append(Paragraph(f"<b>{block_s[4:]}</b>", styles["subheading"]))
            elif block_s.startswith("- "):
                for line in block_s.split("\n"):
                    elements.append(Paragraph(f"• {line.lstrip('- ')}", styles["body"]))
            else:
                elements.append(Paragraph(block_s.replace("\n", "<br/>"), styles["body"]))
            elements.append(Spacer(1, 4))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()
