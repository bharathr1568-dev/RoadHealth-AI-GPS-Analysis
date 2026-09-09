"""
RoadHealth AI — PDF Report Generator
Professional multi-page engineering report using ReportLab.
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
BRAND_DARK  = colors.HexColor("#1a1a2e")
BRAND_BLUE  = colors.HexColor("#0f3460")
BRAND_ACCENT = colors.HexColor("#e94560")
BRAND_LIGHT = colors.HexColor("#f5f5f5")
WHITE = colors.white
BLACK = colors.black

PAGE_W, PAGE_H = A4


# ── Helpers ────────────────────────────────────────────────────

def _cv2_to_reportlab_image(cv2_img, max_width=6.5*inch, max_height=4.5*inch):
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
    """Create the custom paragraph styles for the report."""
    base = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle(
        "RHTitle", parent=base["Title"],
        fontSize=28, textColor=BRAND_DARK, spaceAfter=6,
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    )
    styles["subtitle"] = ParagraphStyle(
        "RHSubtitle", parent=base["Title"],
        fontSize=13, textColor=BRAND_BLUE, spaceAfter=16,
        alignment=TA_CENTER, fontName="Helvetica",
    )
    styles["heading"] = ParagraphStyle(
        "RHHeading", parent=base["Heading1"],
        fontSize=15, textColor=BRAND_DARK, spaceBefore=12, spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    styles["subheading"] = ParagraphStyle(
        "RHSubheading", parent=base["Heading2"],
        fontSize=11, textColor=BRAND_BLUE, spaceBefore=6, spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    styles["body"] = ParagraphStyle(
        "RHBody", parent=base["Normal"],
        fontSize=9.5, leading=13.5, textColor=BLACK,
        fontName="Helvetica", alignment=TA_JUSTIFY,
    )
    styles["body_center"] = ParagraphStyle(
        "RHBodyCenter", parent=styles["body"], alignment=TA_CENTER,
    )
    styles["small"] = ParagraphStyle(
        "RHSmall", parent=base["Normal"],
        fontSize=8, leading=10, textColor=colors.grey, fontName="Helvetica",
    )
    styles["metric_value"] = ParagraphStyle(
        "RHMetric", parent=base["Normal"],
        fontSize=20, textColor=BRAND_ACCENT, fontName="Helvetica-Bold",
        alignment=TA_CENTER,
    )
    styles["metric_label"] = ParagraphStyle(
        "RHMetricLabel", parent=base["Normal"],
        fontSize=8.5, textColor=BRAND_BLUE, fontName="Helvetica",
        alignment=TA_CENTER,
    )
    return styles


def _hr():
    return HRFlowable(width="100%", thickness=1, color=BRAND_BLUE, spaceAfter=8, spaceBefore=4)


def _metric_table(data_pairs, styles, col_count=4):
    """Build a row of metric cards. data_pairs = [(value, label), ...]."""
    cells_val = []
    cells_lbl = []
    for val, lbl in data_pairs:
        cells_val.append(Paragraph(str(val), styles["metric_value"]))
        cells_lbl.append(Paragraph(lbl, styles["metric_label"]))
    # Pad to col_count
    while len(cells_val) % col_count != 0:
        cells_val.append(Paragraph("", styles["metric_value"]))
        cells_lbl.append(Paragraph("", styles["metric_label"]))
    rows = []
    for i in range(0, len(cells_val), col_count):
        rows.append(cells_val[i:i+col_count])
        rows.append(cells_lbl[i:i+col_count])
    col_w = 6.2 * inch / col_count
    t = Table(rows, colWidths=[col_w]*col_count)
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
    ]))
    return t


# ── Page numbers ───────────────────────────────────────────────

def _page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(PAGE_W / 2, 0.4 * inch, f"RoadHealth AI — Page {doc.page}")
    canvas.drawRightString(PAGE_W - inch, 0.4 * inch, datetime.now().strftime("%Y-%m-%d %H:%M"))
    canvas.restoreState()


# ═══════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ═══════════════════════════════════════════════════════════════

def generate_pdf_report(
    original_img_bgr,
    predicted_img_bgr,
    zone_img_bgr,
    features,
    zone_damage,
    progression_records,
    report_summary,
    charts_img_list=None,
    ai_report_text=None,
):
    """
    Generate a complete, non-redundant multi-page PDF report.
    Returns bytes of the PDF.
    """
    buf = io.BytesIO()
    # Margins set to 1 inch (72 pt) left/right, 0.8 inch top/bottom.
    # Printable width: 595.27 - 144 = 451.27 points (approx 6.26 inches)
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=inch, rightMargin=inch,
        topMargin=0.8*inch, bottomMargin=0.8*inch,
    )
    S = _make_styles()
    story = []
    rs = report_summary

    # ── PAGE 1: COVER & EXECUTIVE SUMMARY ──────────────────────
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("ROADHEALTH AI", S["title"]))
    story.append(Paragraph(
        "AI-Based Smart Road Health Monitoring &amp; Predictive Maintenance",
        S["subtitle"],
    ))
    story.append(Spacer(1, 0.05*inch))
    story.append(_hr())
    story.append(Spacer(1, 0.05*inch))
    
    # Metadata block
    now = datetime.now().strftime("%B %d, %Y — %H:%M:%S")
    meta_text = (
        f"<b>Inspection Date:</b> {now} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Model:</b> {MODEL_INFO['name']} ({MODEL_INFO['parameters']:,} params)<br/>"
        f"<b>Classes:</b> Pothole, Crack &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>Task:</b> {MODEL_INFO['task']}"
    )
    story.append(Paragraph(meta_text, S["body_center"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(_hr())
    story.append(Spacer(1, 0.05*inch))
    
    story.append(Paragraph("EXECUTIVE SUMMARY", S["heading"]))
    
    # Core Health Metrics
    story.append(_metric_table([
        (f'{rs["pavement_health_score"]:.1f}', "RHS / 100"),
        (rs["road_condition"], "Condition"),
        (rs["maintenance_priority"], "Priority"),
        (rs["critical_zone"], "Critical Zone"),
    ], S))
    story.append(Spacer(1, 0.08*inch))
    
    # Core Counts / Analysis
    prog_level = rs.get("crack_progression_level", "N/A")
    story.append(_metric_table([
        (rs["detected_potholes"], "Potholes"),
        (rs["detected_cracks"], "Cracks"),
        (rs["total_defects"], "Total Defects"),
        (prog_level, "Progression Risk"),
    ], S))
    story.append(Spacer(1, 0.15*inch))
    
    # Key Recommendation Card
    rec_box = [
        [Paragraph(f"<b>Key Engineering Recommendation:</b> {rs['maintenance_recommendation']}", S["body"])]
    ]
    rec_table = Table(rec_box, colWidths=[6.2*inch])
    rec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4f4")),
        ("BORDER", (0, 0), (-1, -1), 1, BRAND_ACCENT),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(rec_table)
    story.append(PageBreak())

    # ── PAGE 2: VISUAL DISTRESS & SPATIAL RISK ANALYSIS ─────────────
    story.append(Paragraph("VISUAL DISTRESS &amp; SPATIAL ANALYSIS", S["heading"]))
    story.append(_hr())
    
    # Helper to construct image cells with neat borders and captions
    def _make_image_cell(img_bgr, caption, width=3.0*inch, height=2.2*inch):
        try:
            rl_img = _cv2_to_reportlab_image(img_bgr, max_width=width, max_height=height)
            return [
                Table([[rl_img]], colWidths=[width], style=TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cccccc")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ])),
                Spacer(1, 4),
                Paragraph(f"<b>{caption}</b>", S["body_center"])
            ]
        except Exception as e:
            return [Paragraph(f"Error loading image: {e}", S["body"])]

    cell_original = _make_image_cell(original_img_bgr, "Figure 1: Original Road Surface")
    cell_predicted = _make_image_cell(predicted_img_bgr, "Figure 2: AI Distress Detection")
    cell_zone_map = _make_image_cell(zone_img_bgr, "Figure 3: 4×4 Spatial Zone Damage")
    
    # Damaged Zones Summary list
    raw_damaged_zones = []
    for z_id in sorted(zone_damage.keys()):
        z = zone_damage[z_id]
        if z["damage_score"] > 0:
            raw_damaged_zones.append((z_id, z))
            
    # Sort by damage score descending
    raw_damaged_zones.sort(key=lambda x: x[1]["damage_score"], reverse=True)
    
    damaged_zones_rows = []
    for z_id, z in raw_damaged_zones:
        defects_list = []
        if z["potholes"] > 0:
            defects_list.append(f"{z['potholes']} Pot")
        if z["cracks"] > 0:
            defects_list.append(f"{z['cracks']} Crk")
        defects_str = ", ".join(defects_list) if defects_list else "None"
        
        risk_text = z["risk_level"]
        if risk_text == "CRITICAL":
            risk_styled = f"<font color='{BRAND_ACCENT.hexval()}'><b>CRITICAL</b></font>"
        elif risk_text == "HIGH":
            risk_styled = f"<font color='#f39c12'><b>HIGH</b></font>"
        elif risk_text == "MEDIUM":
            risk_styled = f"<font color='#3498db'><b>MEDIUM</b></font>"
        else:
            risk_styled = f"<font color='#2ecc71'><b>LOW</b></font>"
            
        damaged_zones_rows.append([
            Paragraph(f"Z{z_id}", S["small"]),
            Paragraph(f"{z['damage_score']:.3f}", S["small"]),
            Paragraph(risk_styled, S["small"]),
            Paragraph(defects_str, S["small"])
        ])

    if damaged_zones_rows:
        # Show top 5 damaged zones
        top_damaged = damaged_zones_rows[:5]
        zone_header = [
            Paragraph("<b>Zone</b>", S["small"]),
            Paragraph("<b>Damage</b>", S["small"]),
            Paragraph("<b>Risk</b>", S["small"]),
            Paragraph("<b>Defects</b>", S["small"])
        ]
        zone_rows = [zone_header] + top_damaged
        zt = Table(zone_rows, colWidths=[0.55*inch, 0.6*inch, 0.75*inch, 1.0*inch])
        zt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BRAND_LIGHT]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        
        cell_zone_summary = [
            Paragraph("<b>TOP DAMAGED SPATIAL ZONES</b>", S["subheading"]),
            Spacer(1, 4),
            Paragraph("Grid zones exhibiting elevated structural damage:", S["small"]),
            Spacer(1, 4),
            zt
        ]
    else:
        cell_zone_summary = [
            Paragraph("<b>TOP DAMAGED SPATIAL ZONES</b>", S["subheading"]),
            Spacer(1, 10),
            Paragraph("No spatial grid damage detected. All zones are operating at optimal health.", S["body"])
        ]
        
    grid_data = [
        [cell_original, cell_predicted],
        [cell_zone_map, cell_zone_summary]
    ]
    grid_table = Table(grid_data, colWidths=[3.1*inch, 3.1*inch])
    grid_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    
    story.append(grid_table)
    story.append(PageBreak())

    # ── PAGE 3: DEFECT INVENTORY & PROGRESSION RISK ─────────────
    story.append(Paragraph("DEFECT INVENTORY &amp; PROGRESSION RISK", S["heading"]))
    story.append(_hr())
    
    story.append(Paragraph("<b>Distress Inventory:</b> Details of all detected road anomalies.", S["subheading"]))
    story.append(Spacer(1, 4))
    
    if features:
        header = [
            Paragraph("<font color='white'><b>#</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Class</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Conf</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Severity</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Area%</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Edge Den</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Elong</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Bounding Box (X1,Y1,X2,Y2)</b></font>", S["small"])
        ]
        rows = [header]
        for i, d in enumerate(features, 1):
            box_str = f"{d['box'][0]:.0f}, {d['box'][1]:.0f}, {d['box'][2]:.0f}, {d['box'][3]:.0f}"
            rows.append([
                Paragraph(str(i), S["small"]),
                Paragraph(d["class_name"].title(), S["small"]),
                Paragraph(f"{d['confidence']:.3f}", S["small"]),
                Paragraph(f"{d.get('severity_score', 0):.3f}", S["small"]),
                Paragraph(f"{d.get('bbox_area_ratio', 0)*100:.2f}%", S["small"]),
                Paragraph(f"{d.get('edge_density', 0):.2f}", S["small"]),
                Paragraph(f"{d.get('elongation_score', 0):.2f}", S["small"]),
                Paragraph(box_str, S["small"])
            ])
        t = Table(rows, colWidths=[0.3*inch, 0.7*inch, 0.6*inch, 0.6*inch, 0.7*inch, 0.8*inch, 0.7*inch, 1.8*inch], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_BLUE),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BRAND_LIGHT]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No pavement defects were detected by the AI inspection pipeline.", S["body"]))
        
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("<b>Crack-to-Pothole Progression Risk:</b> Proximity-based deterioration metrics.", S["subheading"]))
    story.append(Spacer(1, 4))
    
    if progression_records:
        header = [
            Paragraph("<font color='white'><b>Crack ID</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Risk Score</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Risk Level</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Distance to Pothole</b></font>", S["small"]),
            Paragraph("<font color='white'><b>Same Zone</b></font>", S["small"])
        ]
        rows = [header]
        for r in progression_records:
            dist = r["nearest_pothole_distance_px"]
            dist_str = f"{dist:.1f} px" if dist is not None else "N/A"
            same_zone_str = "Yes" if r["same_zone_as_pothole"] else "No"
            
            lvl = r["risk_level"]
            if lvl == "HIGH":
                lvl_styled = f"<font color='{BRAND_ACCENT.hexval()}'><b>HIGH</b></font>"
            elif lvl == "MEDIUM":
                lvl_styled = f"<font color='#3498db'><b>MEDIUM</b></font>"
            else:
                lvl_styled = f"<font color='#2ecc71'><b>LOW</b></font>"
                
            rows.append([
                Paragraph(f"#{r['crack_id']}", S["small"]),
                Paragraph(f"{r['progression_risk_score']:.3f}", S["small"]),
                Paragraph(lvl_styled, S["small"]),
                Paragraph(dist_str, S["small"]),
                Paragraph(same_zone_str, S["small"])
            ])
        t_prog = Table(rows, colWidths=[1.0*inch, 1.2*inch, 1.0*inch, 1.8*inch, 1.2*inch])
        t_prog.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_ACCENT),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BRAND_LIGHT]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t_prog)
    else:
        story.append(Paragraph("No cracks detected — crack-to-pothole progression risk not estimated.", S["body"]))
        
    story.append(PageBreak())

    # ── PAGE 4: HEALTH METHODOLOGY & MODEL DETAILS ─────────────
    story.append(Paragraph("HEALTH ASSESSMENT METHODOLOGY &amp; MODEL DETAILS", S["heading"]))
    story.append(_hr())
    
    story.append(Paragraph("<b>Road Health Index Formulation</b>", S["subheading"]))
    formula_html = (
        "The road condition is evaluated based on the computed <b>Road Health Score (RHS)</b>, "
        "formulated dynamically from detected defect parameters:<br/><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>RHS = 100 × (1 − Overall Damage Score)</b><br/><br/>"
        "Where the <b>Overall Damage Score</b> is a weighted formulation prioritizing hazard safety:<br/><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Overall Damage = 0.55 × Pothole Damage + 0.20 × Crack Damage + 0.25 × Max Zone Damage</b>"
    )
    story.append(Paragraph(formula_html, S["body"]))
    story.append(Spacer(1, 0.08*inch))
    
    story.append(Paragraph("<b>Component Weights &amp; Rationale:</b>", S["body"]))
    rationale_text = (
        "• <b>Pothole Severity (55%):</b> Potholes pose the highest threat to vehicle safety and structural integrity.<br/>"
        "• <b>Max Zone Damage (25%):</b> Captures localized distress hotspots across the 4×4 spatial grid layout.<br/>"
        "• <b>Crack Severity (20%):</b> Incorporates linear cracking and fatigue, signaling early structural wear."
    )
    story.append(Paragraph(rationale_text, S["body"]))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph("<b>AI Model &amp; Inspection Specifications</b>", S["subheading"]))
    info_data = [
        [Paragraph("<font color='white'><b>Model Specification Property</b></font>", S["body"]),
         Paragraph("<font color='white'><b>Value / System Config</b></font>", S["body"])],
        [Paragraph("Model Name", S["body"]), Paragraph(MODEL_INFO["name"], S["body"])],
        [Paragraph("Architecture File", S["body"]), Paragraph(MODEL_INFO["yaml"], S["body"])],
        [Paragraph("Parameters Count", S["body"]), Paragraph(f"{MODEL_INFO['parameters']:,}", S["body"])],
        [Paragraph("GFLOPs", S["body"]), Paragraph(str(MODEL_INFO["gflops"]), S["body"])],
        [Paragraph("Layer Count", S["body"]), Paragraph(str(MODEL_INFO["layers"]), S["body"])],
        [Paragraph("Inspection Task Type", S["body"]), Paragraph(MODEL_INFO["task"], S["body"])],
        [Paragraph("Target Classes", S["body"]), Paragraph("Pothole, Crack", S["body"])],
        [Paragraph("Training Dataset", S["body"]), Paragraph(MODEL_INFO["dataset"], S["body"])],
    ]
    t_model = Table(info_data, colWidths=[2.5*inch, 3.7*inch])
    t_model.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, BRAND_LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_model)
    story.append(PageBreak())

    # ── PAGE 5: CHARTS (optional) ─────────────────────────────
    if charts_img_list:
        story.append(Paragraph("VISUAL ANALYTICS", S["heading"]))
        story.append(_hr())
        for chart_bytes in charts_img_list:
            img = Image(io.BytesIO(chart_bytes), width=6*inch, height=2.8*inch)
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
        story.append(PageBreak())

    # ── PAGE 6: AI-POWERED FINAL REPORT (optional) ──────────────
    if ai_report_text:
        story.append(Paragraph("AI-POWERED FINAL REPORT", S["heading"]))
        story.append(_hr())
        story.append(Paragraph(
            "<i>Generated by Groq AI · Qwen 3.6 27B Vision — based on detected analysis data.</i>",
            S["small"],
        ))
        story.append(Spacer(1, 0.1*inch))

        # Parse markdown-like text into paragraphs
        for line in ai_report_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 0.05*inch))
            elif stripped.startswith("## "):
                heading_text = stripped[3:].strip()
                story.append(Spacer(1, 0.08*inch))
                story.append(Paragraph(heading_text, S["subheading"]))
            elif stripped.startswith("### "):
                heading_text = stripped[4:].strip()
                story.append(Paragraph(f"<b>{heading_text}</b>", S["body"]))
            elif stripped.startswith("- **") or stripped.startswith("* **"):
                # Bold list item
                item_text = stripped[2:].strip()
                item_text = item_text.replace("**", "<b>", 1).replace("**", "</b>", 1)
                story.append(Paragraph(f"• {item_text}", S["body"]))
            elif stripped.startswith("- ") or stripped.startswith("* "):
                item_text = stripped[2:].strip()
                story.append(Paragraph(f"• {item_text}", S["body"]))
            else:
                text = stripped
                while "**" in text:
                    text = text.replace("**", "<b>", 1).replace("**", "</b>", 1)
                story.append(Paragraph(text, S["body"]))

    # ── PAGE: CONCLUSION & SIGN-OFF (Appended dynamically) ─────
    conclusion_story = []
    conclusion_story.append(Spacer(1, 0.15*inch))
    conclusion_story.append(_hr())
    conclusion_story.append(Paragraph("ENGINEERING CONCLUSION &amp; SIGN-OFF", S["heading"]))
    conclusion_story.append(Spacer(1, 0.05*inch))
    
    conclusion_text = (
        f"This report summarizes the automated pavement distress assessment. The AI detection system "
        f"identified <b>{rs['detected_potholes']} pothole(s)</b> and <b>{rs['detected_cracks']} crack(s)</b>. "
        f"The overall Pavement Health Score is computed as <b>{rs['pavement_health_score']:.1f}/100</b>, "
        f"classifying the road section condition as <b>{rs['road_condition']}</b> with a <b>{rs['maintenance_priority']}</b> "
        f"maintenance priority. The most critical spatial segment is zone <b>{rs['critical_zone']}</b> "
        f"(score: <b>{rs['critical_zone_damage_score']:.3f}</b>). Recommended action: {rs['maintenance_recommendation']}"
    )
    conclusion_story.append(Paragraph(conclusion_text, S["body"]))
    conclusion_story.append(Spacer(1, 0.2*inch))
    
    # Signature fields
    sig_data = [
        [Paragraph("<b>Prepared By:</b>", S["body"]), Paragraph("<b>Approved By:</b>", S["body"])],
        [Paragraph("<br/><br/>_____________________________________<br/>RoadHealth AI Automated Analyzer", S["body"]),
         Paragraph("<br/><br/>_____________________________________<br/>Certified Field Engineer (Sign &amp; Seal)", S["body"])],
        [Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", S["body"]), Paragraph("Date: ________________________", S["body"])]
    ]
    sig_table = Table(sig_data, colWidths=[3.1*inch, 3.1*inch])
    sig_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    conclusion_story.append(sig_table)
    conclusion_story.append(Spacer(1, 0.15*inch))
    conclusion_story.append(Paragraph(
        "<i>AI-generated assessment should be supplemented by professional "
        "field inspection for final engineering decisions.</i>",
        S["small"],
    ))
    
    story.append(KeepTogether(conclusion_story))

    # Build PDF
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return buf.getvalue()
