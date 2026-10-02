"""
BoneSight — Clinical Musculoskeletal Imaging Platform
======================================================
Professional Radiology & Deep-Learning Bone Radiograph Analysis Platform.

Features:
- Light medical-grade design system (#123B5D, #1F6F8B, #F5F8FA)
- Dedicated Landing Page & Diagnostic Radiology Workspace
- Real musculoskeletal X-ray visual assets & clinical viewer frame
- Multi-task anatomical classification & calibrated binary fracture assessment
- Disciplined factual clinical reporting
- Pure frontend decoupling (FastAPI POST /predict integration)
- Complete removal of experimental localization from the user interface
"""

import os
import io
import base64
import requests
import streamlit as st
from PIL import Image, ImageOps

# Page configuration
st.set_page_config(
    page_title="BoneSight | Clinical Bone X-Ray Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Backend URL configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# --- Asset Helpers ---
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

def get_asset_path(filename: str) -> str:
    return os.path.join(ASSETS_DIR, filename)

def get_base64_file(filepath: str) -> str:
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

def pil_to_base64_jpeg(pil_img: Image.Image, quality: int = 95) -> str:
    """Converts a PIL Image to a base64-encoded JPEG data string for controlled DOM rendering."""
    rgb_img = pil_img.convert("RGB")
    buf = io.BytesIO()
    rgb_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Pre-load hero base64
HERO_XRAY_PATH = get_asset_path("hero_xray.jpg")
hero_b64 = get_base64_file(HERO_XRAY_PATH)

# --- Session State Management ---
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "landing"
if "uploaded_bytes" not in st.session_state:
    st.session_state["uploaded_bytes"] = None
if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "view_inverted" not in st.session_state:
    st.session_state["view_inverted"] = False

def navigate_to(page: str):
    st.session_state["current_page"] = page

def reset_analysis():
    st.session_state["uploaded_bytes"] = None
    st.session_state["uploaded_filename"] = None
    st.session_state["prediction_result"] = None
    st.session_state["view_inverted"] = False

def load_preset(filename: str, display_name: str):
    path = get_asset_path(filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.session_state["uploaded_bytes"] = f.read()
        st.session_state["uploaded_filename"] = display_name
        st.session_state["prediction_result"] = None
        st.session_state["view_inverted"] = False
        st.session_state["current_page"] = "analysis"

def render_html(html_str: str):
    """
    Renders HTML in Streamlit safely by stripping all leading indentation
    from each line. This completely prevents markdown parsers (marked.js)
    from interpreting 4+ spaces as an indented code block (<pre><code>).
    """
    clean = "\n".join(line.lstrip() for line in html_str.strip().splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)

# Check backend health
def check_backend_online() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False

is_backend_online = check_backend_online()

# --- CLINICAL MEDICAL-GRADE DESIGN SYSTEM (CSS) ---
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Root Design Tokens */
    :root {
        --primary-navy: #123B5D;
        --secondary-blue: #1F6F8B;
        --accent-cyan: #3C8DAD;
        --bg-clinical: #F5F8FA;
        --surface-white: #FFFFFF;
        --border-light: #D9E3E8;
        --border-subtle: #E2E8F0;
        --text-primary: #1C2D37;
        --text-secondary: #5A6E7C;
        --text-muted: #82929E;
        --medical-green: #2E7D5B;
        --medical-green-bg: #E8F5E9;
        --medical-green-border: #A5D6A7;
        --medical-red: #C94C4C;
        --medical-red-bg: #FFEBEE;
        --medical-red-border: #EF9A9A;
        --viewport-bg: #0B131B;
    }

    /* Streamlit Global Overrides */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: var(--bg-clinical) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1240px !important;
    }

    /* Custom Navigation Bar */
    .med-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: var(--surface-white);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 12px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(18, 59, 93, 0.04);
    }
    .med-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        text-decoration: none;
    }
    .med-brand-icon {
        width: 34px;
        height: 34px;
        background: linear-gradient(135deg, #123B5D 0%, #1F6F8B 100%);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .med-brand-text {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--primary-navy);
        letter-spacing: -0.02em;
    }
    .med-brand-badge {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--secondary-blue);
        background: #EBF3F6;
        padding: 3px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .med-nav-right {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .med-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
        padding: 4px 10px;
        background: var(--bg-clinical);
        border: 1px solid var(--border-light);
        border-radius: 20px;
    }
    .status-dot-online {
        width: 8px;
        height: 8px;
        background-color: var(--medical-green);
        border-radius: 50%;
    }
    .status-dot-offline {
        width: 8px;
        height: 8px;
        background-color: var(--medical-red);
        border-radius: 50%;
    }

    /* Hero Section */
    .hero-container {
        display: flex;
        align-items: center;
        background-color: var(--surface-white);
        border: 1px solid var(--border-light);
        border-radius: 10px;
        padding: 48px;
        margin-bottom: 28px;
        box-shadow: 0 2px 8px rgba(18, 59, 93, 0.04);
    }
    .hero-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
        color: var(--primary-navy);
        margin-bottom: 16px;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        line-height: 1.6;
        margin-bottom: 28px;
        max-width: 520px;
    }

    /* Radiology Workstation Viewport Frame */
    .radiology-frame {
        background-color: var(--viewport-bg);
        border: 1px solid #1E2E3D;
        border-radius: 6px;
        padding: 10px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        color: #8C9BAE;
        font-family: 'IBM Plex Sans', monospace;
        font-size: 0.75rem;
    }
    .viewport-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1B2836;
        padding-bottom: 6px;
        margin-bottom: 8px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #728498;
    }
    .viewport-image-wrapper {
        position: relative;
        overflow: hidden;
        border-radius: 4px;
        background: #000000;
        display: flex;
        align-items: center;
        justify-content: center;
        max-height: 440px;
    }
    .viewport-image-wrapper img {
        width: 100%;
        height: auto;
        max-height: 420px;
        object-fit: contain;
        display: block;
    }
    .viewport-crosshair {
        position: absolute;
        top: 12px;
        right: 12px;
        color: #4A637D;
        font-size: 0.8rem;
    }
    .viewport-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #1B2836;
        padding-top: 6px;
        margin-top: 8px;
        color: #62778D;
    }

    /* Capability Cards */
    .section-heading {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--primary-navy);
        margin: 32px 0 16px 0;
        letter-spacing: -0.01em;
    }
    .capability-card {
        background-color: var(--surface-white);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(18, 59, 93, 0.03);
        height: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .capability-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(18, 59, 93, 0.06);
    }
    .cap-icon {
        font-size: 1.6rem;
        margin-bottom: 12px;
        color: var(--secondary-blue);
    }
    .cap-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--primary-navy);
        margin-bottom: 8px;
    }
    .cap-desc {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* Process Step Cards */
    .step-badge {
        font-family: 'IBM Plex Sans', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--secondary-blue);
        background: #EBF3F6;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 10px;
    }

    /* Diagnostic Results Workspace Cards */
    .result-card {
        background-color: var(--surface-white);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(18, 59, 93, 0.03);
    }
    .result-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    .result-value-lg {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-navy);
        line-height: 1.2;
    }
    .status-badge-fracture {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 6px;
        background-color: var(--medical-red-bg);
        border: 1px solid var(--medical-red-border);
        color: var(--medical-red);
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.02em;
    }
    .status-badge-normal {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 6px;
        background-color: var(--medical-green-bg);
        border: 1px solid var(--medical-green-border);
        color: var(--medical-green);
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.02em;
    }
    
    /* Factual Summary Panel */
    .summary-box {
        background-color: #F8FAFC;
        border-left: 4px solid var(--secondary-blue);
        border-top: 1px solid var(--border-light);
        border-right: 1px solid var(--border-light);
        border-bottom: 1px solid var(--border-light);
        border-radius: 0 6px 6px 0;
        padding: 16px 20px;
        margin: 12px 0 6px 0;
    }
    .summary-text {
        font-size: 1.05rem;
        font-style: italic;
        color: var(--primary-navy);
        line-height: 1.5;
    }
    .summary-meta {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 6px;
    }

    /* Disclaimer Box */
    .disclaimer-box {
        background-color: #FAFBFD;
        border: 1px solid var(--border-light);
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.8rem;
        color: var(--text-muted);
        line-height: 1.5;
        margin-top: 16px;
    }

    /* Progress bar styling */
    .conf-bar-bg {
        width: 100%;
        height: 8px;
        background-color: #E2E8F0;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }
    .conf-bar-fill-primary {
        height: 100%;
        background-color: var(--secondary-blue);
        border-radius: 4px;
    }
    .conf-bar-fill-red {
        height: 100%;
        background-color: var(--medical-red);
        border-radius: 4px;
    }
    .conf-bar-fill-green {
        height: 100%;
        background-color: var(--medical-green);
        border-radius: 4px;
    }

    /* X-Ray Image Viewport Centering & Framing */
    div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        background-color: #060B10 !important;
        border: 1px solid #1E2E3D !important;
        border-radius: 6px !important;
        padding: 12px !important;
        margin-bottom: 12px !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stImage"] img {
        max-height: 480px !important;
        width: auto !important;
        max-width: 100% !important;
        object-fit: contain !important;
        margin: 0 auto !important;
        border-radius: 3px !important;
    }

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="baseButton-primary"] {
        background-color: var(--primary-navy) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        border: 1px solid var(--primary-navy) !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: var(--secondary-blue) !important;
        border-color: var(--secondary-blue) !important;
        color: #FFFFFF !important;
    }

    /* Secondary Action Buttons */
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="baseButton-secondary"],
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: var(--primary-navy) !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        border: 1px solid var(--border-light) !important;
        padding: 0.55rem 1.4rem !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover,
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: #F0F4F8 !important;
        border-color: var(--secondary-blue) !important;
        color: var(--secondary-blue) !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: var(--primary-navy) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--primary-navy) !important;
    }
    
    /* Preset buttons */
    .preset-row {
        display: flex;
        gap: 8px;
        margin-bottom: 14px;
        align-items: center;
    }

    /* =========================================================================
       FIXED-SIZE RADIOLOGY VIEWPORT FRAME (Issue 2, 3 & 4)
       Eliminates layout shifting caused by variable image aspect ratios.
       Maintains strict 420px fixed height for all radiographs (landscape,
       portrait, square) with letterboxed dark clinical presentation.
       ========================================================================= */
    .radiology-fixed-viewer {
        width: 100%;
        height: 420px;
        background-color: var(--viewport-bg);
        border: 1px solid #1E2E3D;
        border-radius: 6px;
        padding: 10px 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        color: #8C9BAE;
        font-family: 'IBM Plex Sans', monospace;
        font-size: 0.75rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        overflow: hidden;
    }
    .radiology-fixed-viewer .viewport-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1B2836;
        padding-bottom: 6px;
        margin-bottom: 6px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #728498;
        flex-shrink: 0;
    }
    .radiology-fixed-viewer .viewport-stage {
        flex: 1;
        width: 100%;
        min-height: 0; /* Ensures flex child respects boundary */
        background-color: #000000;
        border-radius: 4px;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.6);
    }
    .radiology-fixed-viewer .viewport-stage img {
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
        object-fit: contain;
        display: block;
        margin: auto;
        border-radius: 2px;
        user-select: none;
        -webkit-user-drag: none;
    }
    .radiology-fixed-viewer .viewport-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #1B2836;
        padding-top: 6px;
        margin-top: 6px;
        color: #62778D;
        flex-shrink: 0;
    }
    .radiology-standby-stage {
        flex: 1;
        width: 100%;
        min-height: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 16px;
    }

    /* Suppress Streamlit native element toolbar to prevent floating overlaps (Issue 3) */
    div[data-testid="stElementToolbar"] {
        display: none !important;
    }

    /* =========================================================================
       CLINICAL FILE UPLOADER CONTROLS (Issue 1)
       Targets Streamlit's native st.file_uploader components specifically.
       ========================================================================= */
    /* 1. Main Upload/Browse Files Button (Before upload) */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploadDropzone"] button,
    section[data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploadDropzone"] button,
    [data-testid="stFileUploader"] section button {
        background-color: #FFFFFF !important;
        color: #123B5D !important;
        border: 1px solid #B8CBD5 !important;
        border-radius: 7px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.45rem 1.25rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15), 0 1px 2px rgba(18, 59, 93, 0.08) !important;
        cursor: pointer !important;
        transition: background-color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease, color 0.15s ease !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploadDropzone"] button:hover,
    section[data-testid="stFileUploaderDropzone"] button:hover,
    section[data-testid="stFileUploadDropzone"] button:hover,
    [data-testid="stFileUploader"] section button:hover {
        background-color: #EBF3F7 !important;
        border-color: #1F6F8B !important;
        color: #123B5D !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.22) !important;
    }

    [data-testid="stFileUploaderDropzone"] button:focus,
    section[data-testid="stFileUploaderDropzone"] button:focus {
        outline: 2px solid #1F6F8B !important;
        outline-offset: 1px !important;
    }

    /* Text & label elements inside the uploader button */
    [data-testid="stFileUploaderDropzone"] button p,
    [data-testid="stFileUploaderDropzone"] button span,
    [data-testid="stFileUploaderDropzone"] button div,
    section[data-testid="stFileUploaderDropzone"] button p,
    section[data-testid="stFileUploaderDropzone"] button span,
    section[data-testid="stFileUploaderDropzone"] button div {
        color: #123B5D !important;
        font-weight: 600 !important;
    }

    /* Upload Icon inside the uploader button */
    [data-testid="stFileUploaderDropzone"] button span[data-testid="stIconMaterial"],
    [data-testid="stFileUploaderDropzone"] button [class*="Icon"],
    [data-testid="stFileUploaderDropzone"] button svg,
    section[data-testid="stFileUploaderDropzone"] button span[data-testid="stIconMaterial"],
    section[data-testid="stFileUploaderDropzone"] button svg {
        color: #1F6F8B !important;
        fill: #1F6F8B !important;
    }

    /* Clear, readable dropzone instruction text (200MB limit, file formats) */
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    section[data-testid="stFileUploaderDropzone"] small {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
    }

    /* 2. Uploaded File Chip & Filename (After upload) */
    [data-testid="stFileChip"] {
        background-color: #FFFFFF !important;
        border: 1px solid #D9E3E8 !important;
        border-radius: 6px !important;
        color: #123B5D !important;
        padding: 4px 10px !important;
        box-shadow: 0 1px 3px rgba(18, 59, 93, 0.05) !important;
    }
    [data-testid="stFileChipName"] {
        color: #123B5D !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stFileChipDeleteBtn"] button {
        color: #5A6E7C !important;
        cursor: pointer !important;
        transition: color 0.15s ease !important;
    }
    [data-testid="stFileChipDeleteBtn"] button:hover {
        color: #C94C4C !important;
    }

    /* 3. Add-Image Button (After upload: transforms '+' into clear '+ Add image' button) */
    [data-testid="stFileUploader"] button[aria-label="Add files"],
    [data-testid="stFileUploaderDropzone"] button[aria-label="Add files"],
    section[data-testid="stFileUploaderDropzone"] button[aria-label="Add files"] {
        background-color: #FFFFFF !important;
        border: 1px solid #B8CBD5 !important;
        border-radius: 6px !important;
        color: #123B5D !important;
        padding: 0.35rem 0.85rem !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 4px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stFileUploader"] button[aria-label="Add files"]:hover,
    [data-testid="stFileUploaderDropzone"] button[aria-label="Add files"]:hover,
    section[data-testid="stFileUploaderDropzone"] button[aria-label="Add files"]:hover {
        background-color: #EBF3F7 !important;
        border-color: #1F6F8B !important;
        color: #123B5D !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18) !important;
    }
    [data-testid="stFileUploader"] button[aria-label="Add files"]::after,
    [data-testid="stFileUploaderDropzone"] button[aria-label="Add files"]::after,
    section[data-testid="stFileUploaderDropzone"] button[aria-label="Add files"]::after {
        content: "Add image";
        margin-left: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #123B5D;
    }
    [data-testid="stFileUploader"] button[aria-label="Add files"] svg,
    [data-testid="stFileUploader"] button[aria-label="Add files"] span {
        color: #1F6F8B !important;
        fill: #1F6F8B !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HEADER COMPONENT (Clean Navigation Bar)
# ==============================================================================
def render_navbar():
    status_class = "status-dot-online" if is_backend_online else "status-dot-offline"
    status_label = "Backend Connected (Port 8000)" if is_backend_online else "Backend Offline"

    c_nav_left, c_nav_mid, c_nav_right = st.columns([4, 4, 3])
    
    with c_nav_left:
        render_html("""
        <div style="display:flex; align-items:center; gap:10px; padding: 6px 0;">
            <div class="med-brand-icon">BS</div>
            <div>
                <span class="med-brand-text">BoneSight</span>
            </div>
        </div>
        """)

    with c_nav_mid:
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("Home", key="nav_home", width="stretch"):
                navigate_to("landing")
                st.rerun()
        with col_btn2:
            if st.button("Analysis Workspace", key="nav_analysis", width="stretch"):
                navigate_to("analysis")
                st.rerun()
        with col_btn3:
            if st.button("About System", key="nav_about", width="stretch"):
                navigate_to("about")
                st.rerun()

    with c_nav_right:
        render_html(f"""
        <div style="display:flex; justify-content:flex-end; align-items:center; padding: 8px 0;">
            <div class="med-status-pill">
                <span class="{status_class}"></span>
                <span>{status_label}</span>
            </div>
        </div>
        """)
        
    render_html("<div style='height: 1px; background-color: #D9E3E8; margin: 8px 0 20px 0;'></div>")


# ==============================================================================
# PAGE 1: LANDING PAGE
# ==============================================================================
def render_landing_page():
    # --- HERO SECTION ---
    c_hero_text, c_hero_img = st.columns([6, 5], gap="large")

    with c_hero_text:
        render_html("""
        <div style="padding: 24px 0 12px 0;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #1F6F8B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;">
                Musculoskeletal Radiology Analysis
            </div>
            <h1 class="hero-title">AI-Assisted Bone Fracture Analysis</h1>
            <p class="hero-subtitle">
                Analyze plain bone radiographs using deep-learning-based anatomical region classification and calibrated binary fracture detection. Designed for rigorous diagnostic research workflows.
            </p>
        </div>
        """)

        c_cta1, c_cta2 = st.columns([5, 4])
        with c_cta1:
            if st.button("Analyze an X-ray →", key="hero_start_btn", width="stretch"):
                navigate_to("analysis")
                st.rerun()
        with c_cta2:
            if st.button("Review Technical Specs", key="hero_about_btn", width="stretch"):
                navigate_to("about")
                st.rerun()

        render_html("""
        <div style="display: flex; gap: 24px; margin-top: 36px; padding-top: 20px; border-top: 1px solid #E2E8F0;">
            <div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #123B5D;">7 Sites</div>
                <div style="font-size: 0.8rem; color: #5A6E7C;">Anatomical Taxonomy</div>
            </div>
            <div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #123B5D;">Dual-Task</div>
                <div style="font-size: 0.8rem; color: #5A6E7C;">Clinical Inference</div>
            </div>
            <div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #123B5D;">ResNet-50</div>
                <div style="font-size: 0.8rem; color: #5A6E7C;">Validated Backbone</div>
            </div>
        </div>
        """)

    with c_hero_img:
        # High-resolution clinical radiograph inside radiology workstation viewer frame
        render_html(f"""
        <div class="radiology-frame">
            <div class="viewport-header">
                <span>DICOM WORKSTATION // VIEWPORT 01</span>
                <span>AP VIEW</span>
            </div>
            <div class="viewport-image-wrapper">
                <img src="data:image/jpeg;base64,{hero_b64}" alt="Clinical Plain Radiograph">
                <div class="viewport-crosshair">✚</div>
            </div>
            <div class="viewport-footer">
                <span>MATRIX: 2048 &times; 2048</span>
                <span>16-BIT GRAYSCALE</span>
                <span>ASPECT PRESERVED</span>
            </div>
        </div>
        """)

    # --- CAPABILITY STRIP ---
    render_html('<div class="section-heading">Clinical Capabilities</div>')
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_html("""
        <div class="capability-card">
            <div class="cap-icon">🦴</div>
            <div class="cap-title">Anatomical Classification</div>
            <div class="cap-desc">Identifies target body sites across 7 musculoskeletal classes (Wrist, Hand, Arm, Lower leg, Thigh, Pelvis, Foot).</div>
        </div>
        """)

    with c2:
        render_html("""
        <div class="capability-card">
            <div class="cap-icon">⚡</div>
            <div class="cap-title">Fracture Assessment</div>
            <div class="cap-desc">Binary presence detection specialized for localized cortical disruptions, hairline fissures, and severe bone displacement.</div>
        </div>
        """)

    with c3:
        render_html("""
        <div class="capability-card">
            <div class="cap-icon">📊</div>
            <div class="cap-title">Calibrated Scoring</div>
            <div class="cap-desc">Outputs calibrated diagnostic confidence probabilities for anatomical site classification and fracture detection.</div>
        </div>
        """)

    with c4:
        render_html("""
        <div class="capability-card">
            <div class="cap-icon">📋</div>
            <div class="cap-title">Factual Summary</div>
            <div class="cap-desc">Generates disciplined, evidence-bounded clinical natural-language summaries without hallucinating unobserved attributes.</div>
        </div>
        """)

    # --- HOW IT WORKS SECTION ---
    render_html('<div class="section-heading">Diagnostic Evaluation Workflow</div>')
    w1, w2, w3 = st.columns(3)

    with w1:
        render_html("""
        <div class="capability-card">
            <div class="step-badge">STEP 01</div>
            <div class="cap-title">Upload Plain Radiograph</div>
            <div class="cap-desc">Select a bone X-ray file (PNG, JPG, or WEBP). Preprocessing automatically performs aspect-ratio preserved padding and grayscale normalization.</div>
        </div>
        """)

    with w2:
        render_html("""
        <div class="capability-card">
            <div class="step-badge">STEP 02</div>
            <div class="cap-title">Deep Learning Inference</div>
            <div class="cap-desc">The radiograph is evaluated through our validated ResNet-50 models, extracting feature embeddings for anatomy and cortical discontinuity.</div>
        </div>
        """)

    with w3:
        render_html("""
        <div class="capability-card">
            <div class="step-badge">STEP 03</div>
            <div class="cap-title">Review Model Analysis</div>
            <div class="cap-desc">Inspect classified anatomical region, fracture presence status, confidence probability score, and structured factual clinical summary.</div>
        </div>
        """)

    # --- FOOTER ---
    render_html("""
    <div style="margin-top: 50px; padding: 24px 0; border-top: 1px solid #D9E3E8; display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: #5A6E7C;">
        <div>
            <b>BoneSight Platform</b> &bull; Deep Learning Bone Fracture Detection & Captioning Research Group
        </div>
        <div>
            FastAPI Inference Backend &bull; ResNet-50 PyTorch Models &bull; Mendeley BoneFract Taxonomy
        </div>
    </div>
    <div class="disclaimer-box" style="margin-top: 10px;">
        <b>Medical Disclaimer:</b> Research / educational prototype only. This software is intended exclusively for computational imaging research and pair-programming evaluation. It is not an FDA-cleared or CE-marked medical device and must not be used as a substitute for professional clinical diagnosis.
    </div>
    """)


# ==============================================================================
# PAGE 2: ANALYSIS / DIAGNOSTIC WORKSPACE
# ==============================================================================
def render_analysis_page():
    # Workspace Sub-Header
    c_top_left, c_top_right = st.columns([8, 3])
    with c_top_left:
        render_html("""
        <div style="margin-bottom: 12px;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #1F6F8B; text-transform: uppercase; letter-spacing: 0.06em;">Diagnostic Workspace</div>
            <h2 style="font-family:'IBM Plex Sans', sans-serif; font-size:1.6rem; font-weight:700; color:#123B5D; margin:2px 0 0 0;">Radiograph Evaluation Workstation</h2>
        </div>
        """)
    with c_top_right:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("Reset View", key="ws_reset_btn", width="stretch"):
                reset_analysis()
                st.rerun()
        with col_r2:
            if st.button("← Home", key="ws_home_btn", width="stretch"):
                navigate_to("landing")
                st.rerun()

    # Preset Radiograph Demonstrator Strip
    render_html("""
    <div style="background-color: #FFFFFF; border: 1px solid #D9E3E8; border-radius: 6px; padding: 10px 16px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between;">
        <span style="font-size: 0.85rem; font-weight: 600; color: #123B5D;">Quick-Load Preset Clinical Radiographs:</span>
    </div>
    """)
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("Wrist X-ray (AP)", key="preset_wrist", width="stretch"):
            load_preset("sample_wrist.jpg", "Preset: Wrist X-ray (AP)")
            st.rerun()
    with col_p2:
        if st.button("Hip / Pelvis X-ray", key="preset_pelvis", width="stretch"):
            load_preset("sample_pelvis.webp", "Preset: Hip / Pelvis X-ray")
            st.rerun()
    with col_p3:
        if st.button("Forearm X-ray", key="preset_forearm", width="stretch"):
            load_preset("sample_forearm.jpg", "Preset: Forearm X-ray")
            st.rerun()
    with col_p4:
        if st.button("Hand X-ray (PA)", key="preset_hand", width="stretch"):
            load_preset("sample_hand.jpg", "Preset: Hand X-ray (PA)")
            st.rerun()

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Plain Bone Radiograph (PNG, JPG, JPEG, WEBP)",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        help="Upload an X-ray image for anatomical site identification and fracture assessment."
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        if st.session_state["uploaded_bytes"] != file_bytes:
            st.session_state["uploaded_bytes"] = file_bytes
            st.session_state["uploaded_filename"] = uploaded_file.name
            st.session_state["prediction_result"] = None

    # Main 2-Column Workstation Layout (Issue 4: Stable Two-Column Structure)
    col_viewer, col_results = st.columns([6, 5], gap="large")

    with col_viewer:
        render_html('<div class="result-label">X-Ray Viewport // Radiology Workstation</div>')
        
        if st.session_state["uploaded_bytes"] is not None:
            # Parse PIL image
            try:
                pil_img = Image.open(io.BytesIO(st.session_state["uploaded_bytes"]))
                orig_w, orig_h = pil_img.size
                filename_disp = st.session_state.get("uploaded_filename") or "radiograph.jpg"
                
                # Apply inversion if toggled
                display_img = pil_img.convert("L")
                if st.session_state["view_inverted"]:
                    display_img = ImageOps.invert(display_img)
                
                # Convert to base64 JPEG for controlled fixed-size viewport rendering
                img_b64 = pil_to_base64_jpeg(display_img)
                aspect_tag = "LANDSCAPE" if orig_w > orig_h else ("PORTRAIT" if orig_h > orig_w else "SQUARE")

                # Fixed-Size Radiology Viewport Frame (Issue 2 & Issue 3)
                render_html(f"""
                <div class="radiology-fixed-viewer">
                    <div class="viewport-header">
                        <span>VIEWPORT: PRIMARY RADIOGRAPH</span>
                        <span>{orig_w} &times; {orig_h} PX</span>
                    </div>
                    <div class="viewport-stage">
                        <img src="data:image/jpeg;base64,{img_b64}" alt="Clinical Plain Radiograph">
                        <div class="viewport-crosshair">&#10010;</div>
                    </div>
                    <div class="viewport-footer">
                        <span>MATRIX: {orig_w} &times; {orig_h}</span>
                        <span>16-BIT GRAYSCALE</span>
                        <span>ASPECT: {aspect_tag}</span>
                    </div>
                </div>
                """)
                
                # Image metadata & contrast control (Issue 5)
                render_html("<div style='height: 8px;'></div>")
                c_tool1, c_tool2 = st.columns([6, 4])
                with c_tool1:
                    render_html(f"""
                    <div style="font-size: 0.8rem; color: #5A6E7C; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-top: 4px;" title="{filename_disp}">
                        Source: <strong style="color: #123B5D;">{filename_disp}</strong>
                    </div>
                    """)
                with c_tool2:
                    invert_check = st.checkbox("Invert Grayscale (Bone Contrast)", value=st.session_state["view_inverted"])
                    if invert_check != st.session_state["view_inverted"]:
                        st.session_state["view_inverted"] = invert_check
                        st.rerun()

                # Action Controls (Issue 5 & Issue 7)
                render_html("<div style='height: 6px;'></div>")
                btn_col1, btn_col2 = st.columns([7, 3])
                with btn_col1:
                    analyze_clicked = st.button("Analyze Radiograph", type="primary", key="ws_analyze_btn", width="stretch")
                    if analyze_clicked:
                        if not is_backend_online:
                            st.error("Analysis service is currently unavailable. Please verify the FastAPI backend is online.")
                        else:
                            with st.spinner("Analyzing radiograph..."):
                                try:
                                    files = {"image": (filename_disp, st.session_state["uploaded_bytes"], "image/jpeg")}
                                    resp = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=60.0)
                                    if resp.status_code == 200:
                                        st.session_state["prediction_result"] = resp.json()
                                        st.success("Analysis complete.")
                                    elif resp.status_code == 400:
                                        err = resp.json().get("error", "Invalid radiograph format.")
                                        st.error(f"Validation Error: {err}")
                                    else:
                                        err = resp.json().get("error", "Server inference failed.")
                                        st.error(f"Analysis error: {err}")
                                except Exception as e:
                                    st.error("Analysis could not be completed. Please try again.")
                with btn_col2:
                    if st.button("Clear Image", key="ws_clear_btn", width="stretch"):
                        reset_analysis()
                        st.rerun()

            except Exception as e:
                st.error("Please upload a valid X-ray image file.")
        else:
            # Standby prompt inside fixed workstation viewport frame (Issue 2 & Issue 4)
            render_html("""
            <div class="radiology-fixed-viewer">
                <div class="viewport-header">
                    <span>VIEWPORT: STANDBY // NO RADIOGRAPH LOADED</span>
                    <span>READY</span>
                </div>
                <div class="radiology-standby-stage">
                    <div style="font-size: 2.2rem; color: #3C8DAD; margin-bottom: 10px;">&#128193;</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #FFFFFF; margin-bottom: 6px;">Radiology Workstation Ready</div>
                    <div style="font-size: 0.85rem; color: #82929E; max-width: 360px; line-height: 1.5; margin: 0 auto;">
                        Select one of the preset clinical radiographs above, or upload a plain musculoskeletal radiograph to begin evaluation.
                    </div>
                </div>
                <div class="viewport-footer">
                    <span>CHANNEL: DICOM / PLAIN X-RAY</span>
                    <span>AWAITING INPUT</span>
                    <span>7 MUSCULOSKELETAL SITES</span>
                </div>
            </div>
            """)

    with col_results:
        # Issue 7: Terminology update from Clinical Assessment Results to Model Analysis Results
        render_html('<div class="result-label">Model Analysis Results</div>')
        
        res = st.session_state.get("prediction_result")
        
        if res is not None:
            # --- 1. Anatomical Region Card ---
            region_name = res.get("anatomical_region", "Unknown").upper()
            reg_conf = res.get("anatomical_confidence", 0.0)
            
            render_html(f"""
            <div class="result-card">
                <div class="result-label">Anatomical Region</div>
                <div class="result-value-lg">{region_name}</div>
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#5A6E7C; margin-top:8px;">
                    <span>Confidence</span>
                    <b>{reg_conf*100:.1f}%</b>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill-primary" style="width: {min(100, max(0, reg_conf*100)):.1f}%;"></div>
                </div>
            </div>
            """)
            
            # --- 2. Fracture Assessment Card ---
            is_fracture = res.get("fracture", False)
            frac_prob = res.get("fracture_confidence", 0.0)
            
            if is_fracture:
                status_badge = '<div class="status-badge-fracture">&#9679; FRACTURE DETECTED</div>'
                bar_class = "conf-bar-fill-red"
            else:
                status_badge = '<div class="status-badge-normal">&#9679; NO FRACTURE DETECTED</div>'
                bar_class = "conf-bar-fill-green"

            render_html(f"""
            <div class="result-card">
                <div class="result-label">Fracture Assessment</div>
                <div style="margin: 8px 0 14px 0;">
                    {status_badge}
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#5A6E7C;">
                    <span>Probability</span>
                    <b>{frac_prob*100:.1f}%</b>
                </div>
                <div class="conf-bar-bg">
                    <div class="{bar_class}" style="width: {min(100, max(0, frac_prob*100)):.1f}%;"></div>
                </div>
            </div>
            """)

            # --- 3. Factual Clinical Summary Card ---
            caption_text = res.get("caption", "Musculoskeletal radiograph analyzed.")
            render_html(f"""
            <div class="result-card">
                <div class="result-label">Factual Summary</div>
                <div class="summary-box">
                    <div class="summary-text">&ldquo;{caption_text}&rdquo;</div>
                    <div class="summary-meta">Generated from the model&rsquo;s available predictions.</div>
                </div>
            </div>
            """)

        else:
            # Standby panel
            render_html("""
            <div class="result-card" style="padding: 40px 24px; text-align: center;">
                <div style="font-size: 1.8rem; color: #1F6F8B; margin-bottom: 12px;">&#129658;</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #123B5D; margin-bottom: 6px;">Diagnostic Standby</div>
                <div style="font-size: 0.88rem; color: #5A6E7C; line-height: 1.6;">
                    Upload a bone X-ray or click one of the preset clinical radiographs, then press <b>Analyze Radiograph</b> to evaluate the anatomical site and fracture status.
                </div>
            </div>
            """)

        # Medical Disclaimer
        render_html("""
        <div class="disclaimer-box">
            <b>Research Disclaimer:</b> Research / educational use only. This system is not a substitute for professional medical diagnosis.
        </div>
        """)


# ==============================================================================
# PAGE 3: ABOUT / TECHNICAL SPECIFICATIONS
# ==============================================================================
def render_about_page():
    render_html("""
    <div style="margin-bottom: 24px;">
        <div style="font-size: 0.8rem; font-weight: 600; color: #1F6F8B; text-transform: uppercase; letter-spacing: 0.06em;">System Architecture</div>
        <h2 style="font-family:'IBM Plex Sans', sans-serif; font-size:1.8rem; font-weight:700; color:#123B5D; margin:4px 0 0 0;">Technical Specifications & Clinical Framework</h2>
    </div>
    """)

    t1, t2 = st.columns(2, gap="large")

    with t1:
        render_html("""
        <div class="result-card">
            <h4 style="color:#123B5D; margin-top:0;">Deep Learning Architecture</h4>
            <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0; padding:6px 0;">
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Backbone Architecture</b></td>
                    <td style="padding:8px 0; text-align:right;">ResNet-50 Convolutional Network</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Feature Dimensionality</b></td>
                    <td style="padding:8px 0; text-align:right;">2048-dimensional dense vector</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Anatomical Region Head</b></td>
                    <td style="padding:8px 0; text-align:right;">7 classes (Linear 2048 &rarr; 512 &rarr; 7)</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Fracture Detection Model</b></td>
                    <td style="padding:8px 0; text-align:right;">Dedicated Classifier (Linear 2048 &rarr; 256 &rarr; 1)</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Operating Decision Threshold</b></td>
                    <td style="padding:8px 0; text-align:right;">&tau; = 0.50 (Calibrated on validation split)</td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Input Preprocessing</b></td>
                    <td style="padding:8px 0; text-align:right;">224 &times; 224 aspect-preserving pad, ImageNet norm</td>
                </tr>
            </table>
        </div>
        """)

    with t2:
        render_html("""
        <div class="result-card">
            <h4 style="color:#123B5D; margin-top:0;">Dataset & Taxonomy</h4>
            <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Primary Training Dataset</b></td>
                    <td style="padding:8px 0; text-align:right;">Mendeley BoneFract Radiograph Archive</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Supported Anatomical Sites</b></td>
                    <td style="padding:8px 0; text-align:right;">Arm, Foot, Hand, Lower leg, Thigh, Pelvis, Wrist</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Hip Joint Classification</b></td>
                    <td style="padding:8px 0; text-align:right;">Categorized under <code>pelvis</code> class</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Serving Backend</b></td>
                    <td style="padding:8px 0; text-align:right;">FastAPI Production REST Service (Port 8000)</td>
                </tr>
                <tr>
                    <td style="padding:8px 0; color:#5A6E7C;"><b>Localization Module</b></td>
                    <td style="padding:8px 0; text-align:right;"><i>Under active development (excluded from UI)</i></td>
                </tr>
            </table>
        </div>
        """)

    render_html("""
    <div class="result-card" style="margin-top: 16px;">
        <h4 style="color:#123B5D; margin-top:0;">Disciplined Factual Natural-Language Captioning</h4>
        <p style="font-size:0.9rem; color:#5A6E7C; line-height:1.6;">
            The factual caption synthesizer adheres strictly to clinical validation boundaries. It synthesizes natural-language observations using solely verified model outputs:
        </p>
        <ul style="font-size:0.9rem; color:#5A6E7C; line-height:1.6;">
            <li><b>Fracture Present:</b> <i>&ldquo;X-ray of the [region] showing a fracture.&rdquo;</i></li>
            <li><b>Fracture Absent:</b> <i>&ldquo;X-ray of the [region] with no fracture detected by the model.&rdquo;</i></li>
        </ul>
        <p style="font-size:0.85rem; color:#82929E;">
            The captioning engine is explicitly barred from generating laterality (left/right), fracture subtypes (transverse/spiral/comminuted), or displaced bone anatomical naming unless directly supported by trained model classifiers.
        </p>
    </div>
    """)

    if st.button("← Return to Analysis Workspace", width="stretch"):
        navigate_to("analysis")
        st.rerun()


# ==============================================================================
# MAIN ROUTER
# ==============================================================================
render_navbar()

if st.session_state["current_page"] == "landing":
    render_landing_page()
elif st.session_state["current_page"] == "analysis":
    render_analysis_page()
elif st.session_state["current_page"] == "about":
    render_about_page()
