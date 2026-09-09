"""
RoadHealth AI — Design System & Custom UI Components
Provides dark-mode glassmorphic styling, animations, and clean HTML generators.
"""

import re

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-dark: #050816;
    --bg-dark-secondary: #07111F;
    --bg-card: rgba(15, 23, 42, 0.75);
    --bg-card-hover: rgba(30, 41, 59, 0.85);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-highlight: rgba(59, 130, 246, 0.35);
    --border-cyan: rgba(34, 211, 238, 0.4);
    --accent-blue: #3B82F6;
    --accent-cyan: #22D3EE;
    --accent-violet: #8B5CF6;
    --accent-emerald: #10B981;
    --accent-amber: #F59E0B;
    --accent-rose: #F43F5E;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
}

html, body, [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #0d1527 0%, #050816 65%, #02040a 100%) !important;
    background-attachment: fixed !important;
}

html {
    scroll-behavior: smooth;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #050816;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050816 0%, #0a1124 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid var(--border-subtle) !important;
}

.stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(96, 165, 250, 0.4) !important;
    box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.35) !important;
}

.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #60A5FA 0%, #2563EB 100%) !important;
    box-shadow: 0 6px 20px 0 rgba(59, 130, 246, 0.55) !important;
    transform: translateY(-2px) !important;
}

.stButton > button[kind="secondary"], .stButton > button[data-testid="baseButton-secondary"] {
    background: rgba(15, 23, 42, 0.8) !important;
    color: #f8fafc !important;
    border: 1px solid var(--border-subtle) !important;
    backdrop-filter: blur(8px) !important;
}

.stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="baseButton-secondary"]:hover {
    background: rgba(30, 41, 59, 0.95) !important;
    border-color: var(--border-highlight) !important;
    transform: translateY(-2px) !important;
    color: #38bdf8 !important;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px dashed rgba(59, 130, 246, 0.35) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-cyan) !important;
    background: rgba(15, 23, 42, 0.85) !important;
    box-shadow: 0 0 25px rgba(34, 211, 238, 0.15) !important;
}

[data-testid="stDataFrame"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-subtle) !important;
    overflow: hidden !important;
}

.rh-navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: rgba(5, 8, 22, 0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border-subtle);
    border-radius: 14px;
    margin-bottom: 30px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
}

.rh-nav-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
}

.rh-nav-logo-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #3B82F6 0%, #22D3EE 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 0 16px rgba(59, 130, 246, 0.5);
}

.rh-nav-title {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.rh-nav-badge {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
    background: rgba(34, 211, 238, 0.12);
    border: 1px solid rgba(34, 211, 238, 0.3);
    color: #22D3EE;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.rh-nav-links {
    display: flex;
    align-items: center;
    gap: 24px;
}

.rh-nav-link {
    font-size: 0.9rem;
    font-weight: 500;
    color: #94A3B8;
    text-decoration: none;
    transition: color 0.2s ease;
    padding: 6px 10px;
    border-radius: 6px;
}

.rh-nav-link:hover {
    color: #38BDF8;
    background: rgba(255, 255, 255, 0.04);
}

.rh-hero-container {
    padding: 20px 0 40px 0;
}

.rh-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38BDF8;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 6px 14px;
    border-radius: 30px;
    margin-bottom: 18px;
}

.rh-pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22D3EE;
    box-shadow: 0 0 10px #22D3EE;
    display: inline-block;
    animation: rh-pulse 2s infinite;
}

@keyframes rh-pulse {
    0% { transform: scale(0.95); opacity: 0.7; box-shadow: 0 0 0 0 rgba(34, 211, 238, 0.7); }
    70% { transform: scale(1.1); opacity: 1; box-shadow: 0 0 0 8px rgba(34, 211, 238, 0); }
    100% { transform: scale(0.95); opacity: 0.7; box-shadow: 0 0 0 0 rgba(34, 211, 238, 0); }
}

.rh-hero-title {
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1.08;
    letter-spacing: -0.035em;
    background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 14px;
}

.rh-hero-title-accent {
    background: linear-gradient(135deg, #60A5FA 0%, #22D3EE 50%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.rh-hero-tagline {
    font-size: 1.35rem;
    font-weight: 600;
    line-height: 1.4;
    color: #CBD5E1;
    margin-bottom: 18px;
    letter-spacing: -0.01em;
}

.rh-hero-desc {
    font-size: 1.02rem;
    line-height: 1.65;
    color: #94A3B8;
    max-width: 620px;
    margin-bottom: 24px;
}

.rh-researcher-badge {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid var(--border-subtle);
    padding: 10px 18px;
    border-radius: 12px;
    margin-bottom: 28px;
}

.rh-researcher-icon {
    font-size: 1.2rem;
}

.rh-researcher-text {
    font-size: 0.84rem;
    color: #CBD5E1;
    line-height: 1.35;
}

.rh-researcher-text strong {
    color: #FFFFFF;
}

.rh-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.rh-card:hover {
    border-color: var(--border-highlight);
    background: var(--bg-card-hover);
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
}

.rh-card-accent-top {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3B82F6, #22D3EE, #8B5CF6);
}

.rh-section-header {
    margin: 48px 0 24px 0;
    text-align: center;
}

.rh-section-eyebrow {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #22D3EE;
    margin-bottom: 8px;
}

.rh-section-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #F8FAFC;
    margin-bottom: 12px;
}

.rh-section-subtitle {
    font-size: 1.05rem;
    color: #94A3B8;
    max-width: 680px;
    margin: 0 auto;
    line-height: 1.6;
}

.rh-hud-preview {
    background: #090e1f;
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 18px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), inset 0 0 30px rgba(59, 130, 246, 0.1);
}

.rh-hud-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #94A3B8;
}

.rh-hud-canvas {
    position: relative;
    width: 100%;
    height: 280px;
    background: linear-gradient(180deg, #0e172e 0%, #070d1d 100%);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.rh-hud-scanline {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #22D3EE, transparent);
    box-shadow: 0 0 12px #22D3EE;
    animation: rh-scan 3.5s linear infinite;
}

@keyframes rh-scan {
    0% { top: 0%; opacity: 0; }
    15% { opacity: 1; }
    85% { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

.rh-hud-grid {
    position: absolute;
    inset: 0;
    background-image: 
        linear-gradient(rgba(59, 130, 246, 0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59, 130, 246, 0.08) 1px, transparent 1px);
    background-size: 35px 35px;
}

.rh-hud-bbox-pothole {
    position: absolute;
    top: 45%;
    left: 28%;
    width: 140px;
    height: 90px;
    border: 2px solid #22D3EE;
    background: rgba(34, 211, 238, 0.12);
    border-radius: 6px;
    box-shadow: 0 0 15px rgba(34, 211, 238, 0.3);
}

.rh-hud-label-pothole {
    position: absolute;
    top: -24px;
    left: -2px;
    background: #22D3EE;
    color: #050816;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px 4px 0 0;
    letter-spacing: 0.02em;
}

.rh-hud-bbox-crack {
    position: absolute;
    top: 18%;
    left: 60%;
    width: 110px;
    height: 140px;
    border: 2px solid #A78BFA;
    background: rgba(167, 139, 250, 0.12);
    border-radius: 6px;
    box-shadow: 0 0 15px rgba(167, 139, 250, 0.3);
}

.rh-hud-label-crack {
    position: absolute;
    top: -24px;
    left: -2px;
    background: #A78BFA;
    color: #050816;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px 4px 0 0;
    letter-spacing: 0.02em;
}

.rh-hud-footer {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 14px;
}

.rh-hud-metric-pill {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 8px 10px;
    text-align: center;
}

.rh-hud-pill-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    font-weight: 700;
    color: #38BDF8;
}

.rh-hud-pill-lbl {
    font-size: 0.65rem;
    text-transform: uppercase;
    color: #64748B;
    margin-top: 2px;
}

.rh-pipe-flow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    overflow-x: auto;
    padding: 18px 8px;
}

.rh-pipe-node {
    flex: 1;
    min-width: 105px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
    transition: all 0.25s ease;
}

.rh-pipe-node:hover {
    border-color: var(--accent-cyan);
    background: rgba(30, 41, 59, 0.9);
    transform: translateY(-2px);
}

.rh-pipe-node-active {
    border-color: rgba(34, 211, 238, 0.6);
    background: rgba(34, 211, 238, 0.08);
    box-shadow: 0 0 20px rgba(34, 211, 238, 0.15);
}

.rh-pipe-icon {
    font-size: 1.3rem;
    margin-bottom: 6px;
}

.rh-pipe-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.2;
}

.rh-pipe-arrow {
    color: #38BDF8;
    font-size: 1.1rem;
    font-weight: bold;
}

.rh-stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    transition: all 0.2s ease;
}

.rh-stat-card:hover {
    border-color: var(--border-highlight);
    transform: translateY(-2px);
}

.rh-stat-val {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}

.rh-stat-lbl {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-top: 8px;
}

.rh-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.rh-badge-critical { background: rgba(244, 63, 94, 0.18); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.4); }
.rh-badge-high { background: rgba(249, 115, 22, 0.18); color: #FDBA74; border: 1px solid rgba(249, 115, 22, 0.4); }
.rh-badge-medium { background: rgba(245, 158, 11, 0.18); color: #FDE68A; border: 1px solid rgba(245, 158, 11, 0.4); }
.rh-badge-low { background: rgba(16, 185, 129, 0.18); color: #A7F3D0; border: 1px solid rgba(16, 185, 129, 0.4); }
.rh-badge-excellent { background: rgba(16, 185, 129, 0.18); color: #A7F3D0; border: 1px solid rgba(16, 185, 129, 0.4); }
.rh-badge-good { background: rgba(59, 130, 246, 0.18); color: #BFDBFE; border: 1px solid rgba(59, 130, 246, 0.4); }
.rh-badge-fair { background: rgba(245, 158, 11, 0.18); color: #FDE68A; border: 1px solid rgba(245, 158, 11, 0.4); }
.rh-badge-poor { background: rgba(244, 63, 94, 0.18); color: #FDA4AF; border: 1px solid rgba(244, 63, 94, 0.4); }
.rh-badge-routine { background: rgba(99, 102, 241, 0.18); color: #C7D2FE; border: 1px solid rgba(99, 102, 241, 0.4); }

.rh-compare-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
    height: 100%;
}

.rh-compare-card-proposed {
    border-color: rgba(34, 211, 238, 0.35);
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(13, 27, 49, 0.85) 100%);
    box-shadow: 0 10px 30px rgba(34, 211, 238, 0.08);
}

.rh-compare-title {
    font-size: 1.15rem;
    font-weight: 800;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.rh-compare-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 14px;
    font-size: 0.92rem;
    line-height: 1.5;
    color: #CBD5E1;
}

.rh-compare-icon-neg {
    color: #F87171;
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 2px;
}

.rh-compare-icon-pos {
    color: #34D399;
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 2px;
}

.rh-footer {
    border-top: 1px solid var(--border-subtle);
    padding: 40px 0 20px 0;
    margin-top: 60px;
}

.rh-footer-grid {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 24px;
    margin-bottom: 24px;
}

.rh-footer-col-main {
    max-width: 420px;
}

.rh-footer-col-links {
    display: flex;
    gap: 32px;
}

.rh-footer-bottom {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    padding-top: 16px;
    font-size: 0.78rem;
    color: #64748B;
}

/* ── RoadHealth Map UI Elements ── */
.rh-map-legend-container {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 10px 16px;
    backdrop-filter: blur(8px);
}

.rh-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}

.rh-legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    box-shadow: 0 0 8px currentColor;
}

.rh-assessment-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(16px);
}

.rh-streetview-preview-box {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(34, 211, 238, 0.3);
    margin-bottom: 16px;
    background: #07111F;
}

.rh-streetview-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(5, 8, 22, 0.85);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #38BDF8;
    z-index: 2;
    display: flex;
    align-items: center;
    gap: 6px;
}
</style>
"""


def render_html(html_str: str) -> str:
    """
    Renders clean, robust HTML for Streamlit markdown.
    Removes HTML comments and strips all leading/trailing whitespace from every line,
    guaranteeing that Markdown NEVER interprets lines with 4+ spaces as indented code blocks!
    """
    # 1. Remove all HTML comments
    cleaned = re.sub(r'<!--.*?-->', '', html_str, flags=re.DOTALL)
    # 2. Strip every line of leading and trailing whitespace
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    # 3. Join with newline (every line starts at column 0)
    return "\n".join(lines)


def get_navbar_html() -> str:
    return render_html("""
<div class="rh-navbar">
<div class="rh-nav-brand">
<div class="rh-nav-logo-icon">🛣️</div>
<div>
<span class="rh-nav-title">RoadHealth AI</span>
<span class="rh-nav-badge">P2P3·CBAM</span>
</div>
</div>
<div class="rh-nav-links">
<a href="#home" class="rh-nav-link">Home</a>
<a href="#research" class="rh-nav-link">Research</a>
<a href="#architecture" class="rh-nav-link">Architecture</a>
<a href="#performance" class="rh-nav-link">Performance</a>
<a href="#explainability" class="rh-nav-link">Explainability</a>
</div>
</div>
""")


def get_hero_html() -> str:
    return render_html("""
<div class="rh-hero-container">
<div class="rh-eyebrow">
<span class="rh-pulse-dot"></span>
AI-POWERED PAVEMENT INTELLIGENCE
</div>
<div class="rh-hero-title">
RoadHealth <span class="rh-hero-title-accent">AI</span>
</div>
<div class="rh-hero-tagline">
Explainable AI for Intelligent Pavement Health Monitoring
</div>
<div class="rh-hero-desc">
An advanced computer-vision research platform integrating <strong>YOLO11m-P2P3-CBAM</strong> 
with fine-grained detail enhancement, dual-attention spatial refinement, 4×4 zone risk mapping, 
and automated engineering maintenance decisions.
</div>
<div class="rh-researcher-badge">
<div class="rh-researcher-icon">🎓</div>
<div class="rh-researcher-text">
<strong>Researcher: Bharath</strong><br>
Integrated M.Tech Computer Science &amp; Engineering (Data Science)<br>
Vellore Institute of Technology (VIT), Vellore
</div>
</div>
</div>
""")


def get_hud_mockup_html() -> str:
    return render_html("""
<div class="rh-hud-preview">
<div class="rh-hud-header">
<span>● LIVE SENSOR FEED · CH-01</span>
<span style="color:#22D3EE;">YOLO11m-P2P3-CBAM · INFERENCE READY</span>
</div>
<div class="rh-hud-canvas">
<div class="rh-hud-grid"></div>
<div class="rh-hud-scanline"></div>
<div class="rh-hud-bbox-pothole">
<div class="rh-hud-label-pothole">POTHOLE 94.8% · Z14</div>
</div>
<div class="rh-hud-bbox-crack">
<div class="rh-hud-label-crack">CRACK 89.2% · Z07</div>
</div>
<div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); pointer-events:none; opacity:0.35;">
<svg width="60" height="60" viewBox="0 0 60 60">
<circle cx="30" cy="30" r="24" fill="none" stroke="#22D3EE" stroke-width="1.5" stroke-dasharray="4 4"/>
<line x1="30" y1="0" x2="30" y2="15" stroke="#22D3EE" stroke-width="1.5"/>
<line x1="30" y1="45" x2="30" y2="60" stroke="#22D3EE" stroke-width="1.5"/>
<line x1="0" y1="30" x2="15" y2="30" stroke="#22D3EE" stroke-width="1.5"/>
<line x1="45" y1="30" x2="60" y2="30" stroke="#22D3EE" stroke-width="1.5"/>
</svg>
</div>
</div>
<div class="rh-hud-footer">
<div class="rh-hud-metric-pill">
<div class="rh-hud-pill-val">68.3</div>
<div class="rh-hud-pill-lbl">GFLOPs</div>
</div>
<div class="rh-hud-metric-pill">
<div class="rh-hud-pill-val" style="color:#22D3EE;">88.4%</div>
<div class="rh-hud-pill-lbl">Precision</div>
</div>
<div class="rh-hud-metric-pill">
<div class="rh-hud-pill-val" style="color:#A78BFA;">84.20%</div>
<div class="rh-hud-pill-lbl">mAP@50</div>
</div>
<div class="rh-hud-metric-pill">
<div class="rh-hud-pill-val" style="color:#34D399;">Active</div>
<div class="rh-hud-pill-lbl">CBAM Attn</div>
</div>
</div>
</div>
""")


def get_horizontal_pipeline_html() -> str:
    return render_html("""
<div class="rh-pipe-flow">
<div class="rh-pipe-node">
<div class="rh-pipe-icon">📷</div>
<div class="rh-pipe-title">ROAD IMAGE</div>
</div>
<div class="rh-pipe-arrow">→</div>
<div class="rh-pipe-node">
<div class="rh-pipe-icon">🧠</div>
<div class="rh-pipe-title">YOLO11m</div>
</div>
<div class="rh-pipe-arrow">→</div>
<div class="rh-pipe-node rh-pipe-node-active">
<div class="rh-pipe-icon">⚡</div>
<div class="rh-pipe-title">P2→P3 DETAIL</div>
</div>
<div class="rh-pipe-arrow">→</div>
<div class="rh-pipe-node rh-pipe-node-active">
<div class="rh-pipe-icon">🎯</div>
<div class="rh-pipe-title">CBAM ATTENTION</div>
</div>
<div class="rh-pipe-arrow">→</div>
<div class="rh-pipe-node">
<div class="rh-pipe-icon">🔍</div>
<div class="rh-pipe-title">DAMAGE DETECTION</div>
</div>
<div class="rh-pipe-arrow">→</div>
<div class="rh-pipe-node">
<div class="rh-pipe-icon">📐</div>
<div class="rh-pipe-title">SEVERITY</div>
</div>
<div class="rh-pipe-arrow">→</div>
<div class="rh-pipe-node">
<div class="rh-pipe-icon">🗺️</div>
<div class="rh-pipe-title">ZONE ANALYSIS</div>
</div>
<div class="rh-pipe-arrow">→</div>
<div class="rh-pipe-node">
<div class="rh-pipe-icon">📊</div>
<div class="rh-pipe-title">RHS SCORE</div>
</div>
<div class="rh-pipe-arrow">→</div>
<div class="rh-pipe-node" style="border-color:rgba(52, 211, 153, 0.5); background:rgba(52, 211, 153, 0.08);">
<div class="rh-pipe-icon">🛡️</div>
<div class="rh-pipe-title">DECISION</div>
</div>
</div>
""")


def get_architecture_diagram_html() -> str:
    return render_html("""
<div style="background: rgba(10, 15, 30, 0.85); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 18px; padding: 28px; box-shadow: 0 15px 35px rgba(0,0,0,0.5);">
<div style="display: flex; justify-content: center; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 20px;">
<div style="background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 12px 20px; text-align: center; min-width: 170px;">
<div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Input Stream</div>
<div style="font-size: 0.98rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">Road Image (640×640)</div>
</div>
<div style="color: #38BDF8; font-size: 1.3rem;">↓</div>
<div style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 12px; padding: 12px 24px; text-align: center; min-width: 220px;">
<div style="font-size: 0.72rem; color: #60A5FA; text-transform: uppercase; font-weight: 700;">Feature Extractor</div>
<div style="font-size: 1.05rem; font-weight: 800; color: #FFFFFF; margin-top: 2px;">YOLO11m Backbone</div>
<div style="font-size: 0.7rem; color: #94A3B8; margin-top: 2px;">Multi-scale feature maps (P2, P3, P4, P5)</div>
</div>
</div>
<div style="text-align: center; color: #38BDF8; font-size: 1.3rem; margin-bottom: 14px;">↓</div>
<div style="background: linear-gradient(180deg, rgba(14, 30, 60, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%); border: 2px solid rgba(34, 211, 238, 0.45); border-radius: 16px; padding: 22px; position: relative; margin-bottom: 20px; box-shadow: 0 0 30px rgba(34, 211, 238, 0.12);">
<div style="position: absolute; top: -12px; left: 24px; background: linear-gradient(90deg, #3B82F6, #22D3EE); color: #050816; font-size: 0.72rem; font-weight: 800; padding: 3px 12px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.06em;">
★ Proposed Research Contribution
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 10px; align-items: center;">
<div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 14px; text-align: center;">
<div style="font-size: 0.72rem; color: #38BDF8; font-weight: 700;">STAGE 1: P2 EXTRACTION</div>
<div style="font-weight: 700; font-size: 0.92rem; color: #FFFFFF; margin-top: 4px;">P2 High-Res Layer</div>
<div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">Captures shallow micro-cracks &amp; fine damage boundaries</div>
</div>
<div style="text-align: center; color: #22D3EE; font-size: 1.2rem; font-weight: bold;">→</div>
<div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(34, 211, 238, 0.4); border-radius: 12px; padding: 14px; text-align: center;">
<div style="font-size: 0.72rem; color: #22D3EE; font-weight: 700;">STAGE 2: DETAIL BRANCH</div>
<div style="font-weight: 700; font-size: 0.92rem; color: #FFFFFF; margin-top: 4px;">P2→P3 Enhancement</div>
<div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">Downsample Conv + Depthwise Conv (SiLU activation)</div>
</div>
<div style="text-align: center; color: #A78BFA; font-size: 1.2rem; font-weight: bold;">→</div>
<div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(167, 139, 250, 0.45); border-radius: 12px; padding: 14px; text-align: center;">
<div style="font-size: 0.72rem; color: #A78BFA; font-weight: 700;">STAGE 3: CBAM ATTENTION</div>
<div style="font-weight: 700; font-size: 0.92rem; color: #FFFFFF; margin-top: 4px;">Channel + Spatial Attn</div>
<div style="font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">Adaptive pooling MLP + 7×7 Spatial Conv filtering</div>
</div>
</div>
</div>
<div style="text-align: center; color: #38BDF8; font-size: 1.3rem; margin-bottom: 14px;">↓</div>
<div style="display: flex; justify-content: center; align-items: center; gap: 16px; flex-wrap: wrap;">
<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%); border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 12px; padding: 12px 20px; text-align: center; min-width: 200px;">
<div style="font-size: 0.72rem; color: #60A5FA; text-transform: uppercase; font-weight: 700;">Refined Fusion</div>
<div style="font-size: 0.95rem; font-weight: 800; color: #FFFFFF; margin-top: 2px;">YOLO Detection Head</div>
</div>
<div style="color: #34D399; font-size: 1.3rem;">→</div>
<div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 12px 22px; text-align: center; min-width: 210px;">
<div style="font-size: 0.72rem; color: #34D399; text-transform: uppercase; font-weight: 700;">Pavement Outputs</div>
<div style="font-size: 0.95rem; font-weight: 800; color: #A7F3D0; margin-top: 2px;">Pothole &amp; Crack Detections</div>
</div>
</div>
</div>
""")


def get_why_cbam_comparison_html() -> str:
    return render_html("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
<div class="rh-compare-card">
<div class="rh-compare-title" style="color: #F87171;">
<span>⚠️</span> Conventional Object Detection
</div>
<div class="rh-compare-item">
<span class="rh-compare-icon-neg">✕</span>
<div><strong>Fine Cracks Lost in Downsampling:</strong> Standard deep feature strides (P3-P5) aggressively downsample image details, losing thin hairline cracks.</div>
</div>
<div class="rh-compare-item">
<span class="rh-compare-icon-neg">✕</span>
<div><strong>Small &amp; Shallow Damage Missed:</strong> Incipient distress points fail to activate higher-level feature channels without high-resolution assistance.</div>
</div>
<div class="rh-compare-item">
<span class="rh-compare-icon-neg">✕</span>
<div><strong>Background Noise &amp; Artifacts:</strong> Road texture shadows, oil spots, and asphalt variations cause false-positive detections.</div>
</div>
<div class="rh-compare-item">
<span class="rh-compare-icon-neg">✕</span>
<div><strong>Loss of Spatial Boundary Definition:</strong> Conventional heads lack localized attention to isolate exact boundary contours.</div>
</div>
</div>
<div class="rh-compare-card rh-compare-card-proposed">
<div class="rh-compare-title" style="color: #22D3EE;">
<span>✨</span> Proposed YOLO11m-P2P3-CBAM
</div>
<div class="rh-compare-item">
<span class="rh-compare-icon-pos">✓</span>
<div><strong>P2→P3 Detail Enhancement:</strong> Retains high-resolution shallow features from P2 (stride 4) and fuses them into P3 for fine crack preservation.</div>
</div>
<div class="rh-compare-item">
<span class="rh-compare-icon-pos">✓</span>
<div><strong>CBAM Channel Attention:</strong> Learns 'WHAT' feature channels carry true road distress signatures while suppressing noisy asphalt textures.</div>
</div>
<div class="rh-compare-item">
<span class="rh-compare-icon-pos">✓</span>
<div><strong>CBAM Spatial Attention:</strong> Discovers 'WHERE' localized damage boundaries are located, sharpening bounding box accuracy.</div>
</div>
<div class="rh-compare-item">
<span class="rh-compare-icon-pos">✓</span>
<div><strong>Multi-Scale Feature Refinement:</strong> Seamlessly combines macro pothole geometry with micro hairline crack representations.</div>
</div>
</div>
</div>
""")


def get_research_contributions_html() -> str:
    contributions = [
        ("01", "P2→P3 Detail Enhancement", "Directly taps shallow P2 feature maps to preserve fine-grained pavement textures, hairline cracks, and localized edge boundaries before deep downsampling."),
        ("02", "CBAM Attention Refinement", "Applies dual Channel and Spatial Attention to suppress background road noise (shadows, tire marks) and amplify true distress patterns."),
        ("03", "Severity Assessment", "Calculates composite engineering damage severity per defect utilizing bounding box geometry, edge density, and morphological proxies."),
        ("04", "Zone Damage Analysis", "Subdivides the road surface into a 4×4 spatial matrix to pinpoint critical distress clusters, safety floors, and localized risk zones."),
        ("05", "Road Health Score (RHS)", "Formulates an explainable composite RHS metric (0–100) combining defect counts, severity metrics, and spatial concentration weights."),
        ("06", "Crack→Pothole Progression Risk", "Models the probability of crack deterioration into full structural potholes using distance proximity, morphology, and zone density."),
        ("07", "Explainable Maintenance Intelligence", "Translates raw computer-vision detections into prioritized civil engineering decisions and automated multimodal AI reports."),
    ]
    
    cards_html = ""
    for num, title, desc in contributions:
        cards_html += f"""
<div class="rh-card" style="min-height: 180px;">
<div class="rh-card-accent-top"></div>
<div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 800; color: #22D3EE; margin-bottom: 8px;">{num}</div>
<div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">{title}</div>
<div style="font-size: 0.84rem; color: #94A3B8; line-height: 1.5;">{desc}</div>
</div>
"""
    
    return render_html(f"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px;">
{cards_html}
</div>
""")


def get_intelligence_pipeline_html() -> str:
    stages = [
        ("1. Image Input", "High-resolution road surface imagery captured via mobile, drone, or survey cameras."),
        ("2. AI Detection", "YOLO11m-P2P3-CBAM detects potholes and cracks with high precision and attention focus."),
        ("3. Severity Extraction", "Computes area ratios, edge densities, and morphological distress metrics."),
        ("4. Spatial Mapping", "Evaluates 4×4 grid cells to isolate high-risk damage clusters and safety floors."),
        ("5. RHS Computation", "Aggregates overall distress into a standard 0–100 Road Health Score index."),
        ("6. Risk Modeling", "Calculates crack-to-pothole progression risk to identify impending road failures."),
        ("7. Maintenance Action", "Synthesizes actionable engineering priorities, budget estimates, and automated reports."),
    ]
    
    cards_html = ""
    for title, desc in stages:
        cards_html += f"""
<div class="rh-stat-card" style="text-align: left; padding: 16px;">
<div style="font-size: 0.92rem; font-weight: 700; color: #38BDF8; margin-bottom: 4px;">{title}</div>
<div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.4;">{desc}</div>
</div>
"""
    
    return render_html(f"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px;">
{cards_html}
</div>
""")


def get_explainability_framework_html() -> str:
    return render_html("""
<div style="background: rgba(15, 23, 42, 0.75); border: 1px solid var(--border-subtle); border-radius: 18px; padding: 26px;">
<div style="display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;">
<div style="flex: 1; min-width: 130px; text-align: center; padding: 10px;">
<div style="font-size: 1.4rem;">🎯</div>
<div style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">Detection</div>
<div style="font-size: 0.72rem; color: #94A3B8;">BBoxes &amp; Classes</div>
</div>
<div style="color: #38BDF8; font-size: 1.1rem; font-weight: bold;">→</div>
<div style="flex: 1; min-width: 130px; text-align: center; padding: 10px;">
<div style="font-size: 1.4rem;">📐</div>
<div style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">Severity</div>
<div style="font-size: 0.72rem; color: #94A3B8;">Depth &amp; Area Density</div>
</div>
<div style="color: #38BDF8; font-size: 1.1rem; font-weight: bold;">→</div>
<div style="flex: 1; min-width: 130px; text-align: center; padding: 10px;">
<div style="font-size: 1.4rem;">🗺️</div>
<div style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">Zone Mapping</div>
<div style="font-size: 0.72rem; color: #94A3B8;">4×4 Spatial Grid</div>
</div>
<div style="color: #38BDF8; font-size: 1.1rem; font-weight: bold;">→</div>
<div style="flex: 1; min-width: 130px; text-align: center; padding: 10px;">
<div class="rh-pipe-icon" style="font-size: 1.4rem;">📊</div>
<div style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">RHS Score</div>
<div style="font-size: 0.72rem; color: #94A3B8;">0–100 Health Index</div>
</div>
<div style="color: #38BDF8; font-size: 1.1rem; font-weight: bold;">→</div>
<div style="flex: 1; min-width: 130px; text-align: center; padding: 10px;">
<div style="font-size: 1.4rem;">🚨</div>
<div style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF; margin-top: 4px;">Priority</div>
<div style="font-size: 0.72rem; color: #94A3B8;">Emergency / Routine</div>
</div>
<div style="color: #38BDF8; font-size: 1.1rem; font-weight: bold;">→</div>
<div style="flex: 1; min-width: 130px; text-align: center; padding: 10px;">
<div style="font-size: 1.4rem;">📋</div>
<div style="font-size: 0.85rem; font-weight: 700; color: #34D399; margin-top: 4px;">AI Report</div>
<div style="font-size: 0.72rem; color: #94A3B8;">Civil Engineering LLM</div>
</div>
</div>
</div>
""")


def get_footer_html() -> str:
    return render_html("""
<div class="rh-footer">
<div class="rh-footer-grid">
<div class="rh-footer-col-main">
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
<div style="width: 28px; height: 28px; border-radius: 8px; background: linear-gradient(135deg, #3B82F6, #22D3EE); display: flex; align-items: center; justify-content: center; font-size: 0.95rem;">🛣️</div>
<span style="font-size: 1.15rem; font-weight: 800; color: #FFFFFF;">RoadHealth AI</span>
</div>
<div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5; margin-bottom: 12px;">
Explainable AI Framework for Pavement Damage Assessment and Infrastructure Deterioration Prediction.
</div>
<div style="font-size: 0.8rem; color: #64748B;">
Academic Research Project · Vellore Institute of Technology (VIT)
</div>
</div>
<div class="rh-footer-col-links">
<div>
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #CBD5E1; margin-bottom: 8px;">Researcher</div>
<div style="font-size: 0.84rem; color: #94A3B8; line-height: 1.6;">
<strong>Bharath</strong><br>
Integrated M.Tech CSE (Data Science)<br>
School of Computer Science &amp; Engineering<br>
VIT Vellore, India
</div>
</div>
<div>
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #CBD5E1; margin-bottom: 8px;">Model Spec</div>
<div style="font-size: 0.84rem; color: #94A3B8; line-height: 1.6;">
YOLO11m-P2P3-CBAM<br>
20.39M Parameters<br>
68.3 GFLOPs<br>
900 Test Set Validated
</div>
</div>
</div>
</div>
<div class="rh-footer-bottom">
<span>© 2026 RoadHealth AI · All Research Rights Reserved</span>
<span>Proposed Model: YOLO11m-P2P3-CBAM</span>
</div>
</div>
""")


def metric_card_html(value: str, label: str) -> str:
    return render_html(f"""
<div class="rh-stat-card">
<div class="rh-stat-val">{value}</div>
<div class="rh-stat-lbl">{label}</div>
</div>
""")


def risk_badge_html(level: str) -> str:
    cls = str(level).lower().replace(" ", "-")
    return f'<span class="rh-badge rh-badge-{cls}">{level}</span>'


def get_map_header_html() -> str:
    return render_html("""
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin-bottom: 20px;">
    <div style="display: flex; align-items: center; gap: 12px;">
        <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #10B981 0%, #22D3EE 100%); display: flex; align-items: center; justify-content: center; font-size: 1.4rem; box-shadow: 0 0 20px rgba(34, 211, 238, 0.4);">🗺️</div>
        <div>
            <h2 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">RoadHealth Route Map Monitoring</h2>
            <p style="margin: 0; font-size: 0.88rem; color: #94A3B8;">OpenFreeMap + MapLibre GL + KartaView Street-Level Pavement Intelligence</p>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 8px; background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(34, 211, 238, 0.3); border-radius: 30px; padding: 6px 14px; font-size: 0.78rem; color: #38BDF8; font-weight: 600;">
        <span class="rh-pulse-dot"></span>
        <span>YOLO11m-P2P3-CBAM · Image Paper Deployment</span>
    </div>
</div>
""")


def get_map_legend_html() -> str:
    return render_html("""
<div class="rh-map-legend-container">
    <div style="font-size: 0.76rem; font-weight: 700; color: #CBD5E1; text-transform: uppercase; letter-spacing: 0.05em; margin-right: 4px;">
        Pavement Health Score:
    </div>
    <div class="rh-legend-item" style="color: #A7F3D0;">
        <div class="rh-legend-dot" style="background: #10B981; color: #10B981;"></div>
        <span>🟢 80–100 Healthy</span>
    </div>
    <div class="rh-legend-item" style="color: #FDE68A;">
        <div class="rh-legend-dot" style="background: #F59E0B; color: #F59E0B;"></div>
        <span>🟡 60–79 Moderate</span>
    </div>
    <div class="rh-legend-item" style="color: #FDBA74;">
        <div class="rh-legend-dot" style="background: #F97316; color: #F97316;"></div>
        <span>🟠 40–59 Poor</span>
    </div>
    <div class="rh-legend-item" style="color: #FDA4AF;">
        <div class="rh-legend-dot" style="background: #EF4444; color: #EF4444;"></div>
        <span>🔴 0–39 Critical</span>
    </div>
</div>
""")

