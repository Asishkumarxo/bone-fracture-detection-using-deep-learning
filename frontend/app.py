"""
Bone Fracture Detection & Image Captioning
==========================================
Streamlit Frontend Application for Deep-Learning Bone Radiograph Analysis.

Communicates with the production FastAPI backend (POST /predict)
without duplicating any ML inference or model weights.
"""

import os
import requests
import streamlit as st
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Bone Fracture Detection & Captioning",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card-positive {
        background-color: #fef2f2;
        border: 1px solid #f87171;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-card-negative {
        background-color: #f0fdf4;
        border: 1px solid #4ade80;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-card-neutral {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .caption-box {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 1.1rem;
        font-style: italic;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Default backend URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Session state initialization
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "uploaded_bytes" not in st.session_state:
    st.session_state["uploaded_bytes"] = None
if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None

def reset_state():
    st.session_state["prediction_result"] = None
    st.session_state["uploaded_bytes"] = None
    st.session_state["uploaded_filename"] = None

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/x-ray.png", width=70)
    st.title("Project Controls")
    
    st.markdown("### Backend Connection")
    backend_input = st.text_input("FastAPI Endpoint URL", value=BACKEND_URL)
    if backend_input:
        BACKEND_URL = backend_input.rstrip("/")
        
    # Check backend health
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=2.0)
        if health_resp.status_code == 200:
            st.success("Backend Online (Port 8000)")
        else:
            st.warning("Backend reachable but status not OK")
    except Exception:
        st.error("Backend Offline. Start FastAPI server first.")
        
    st.markdown("---")
    st.markdown("### System Architecture")
    st.markdown("""
    - **Backbone:** ResNet-50 Multi-Task
    - **Heads:** Anatomy (7-class) + Binary Fracture
    - **Detector:** Faster R-CNN MobileNetV3 FPN
    - **Dataset:** BoneFract (Mendeley) + FracAtlas
    - **Captioning:** Factual Medical Synthesizer
    """)
    
    st.markdown("#### Architecture Downloads")
    pdf_file = "Bone_Fracture_System_Architecture.pdf"
    docx_file = "Bone_Fracture_System_Architecture.docx"
    png_file = "reports/architecture_block_diagram.png"
    
    if os.path.exists(pdf_file):
        with open(pdf_file, "rb") as f:
            st.download_button("📄 Download Architecture PDF", f.read(), file_name="Bone_Fracture_System_Architecture.pdf", mime="application/pdf", width="stretch")
    if os.path.exists(docx_file):
        with open(docx_file, "rb") as f:
            st.download_button("📝 Download Architecture Word (.docx)", f.read(), file_name="Bone_Fracture_System_Architecture.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", width="stretch")
    if os.path.exists(png_file):
        with open(png_file, "rb") as f:
            st.download_button("🖼️ Download Block Diagram PNG", f.read(), file_name="architecture_block_diagram.png", mime="image/png", width="stretch")
            
    st.markdown("---")
    if st.button("Reset / New Image", width="stretch"):
        reset_state()
        st.rerun()

# --- Main Page Header ---
st.markdown('<div class="main-header">Bone Fracture Detection & Image Captioning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-task deep learning radiograph analysis with spatial fracture localization and disciplined factual captioning.</div>', unsafe_allow_html=True)

# --- File Uploader ---
st.markdown("### Upload X-ray image")
uploaded_file = st.file_uploader(
    "Select a plain bone radiograph (PNG, JPG, JPEG, BMP)",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    help="Upload an X-ray image to identify anatomical site, fracture presence, and localized lesions."
)

if uploaded_file is not None:
    # Save in session state if changed
    file_bytes = uploaded_file.getvalue()
    if st.session_state["uploaded_bytes"] != file_bytes:
        st.session_state["uploaded_bytes"] = file_bytes
        st.session_state["uploaded_filename"] = uploaded_file.name
        st.session_state["prediction_result"] = None # Reset previous prediction for new upload

    col_img, col_btn = st.columns([2, 1])
    
    with col_img:
        st.markdown("**Original X-Ray Image**")
        st.image(uploaded_file, caption=f"File: {uploaded_file.name}", width="stretch")
        
    with col_btn:
        st.markdown("**Run Diagnosis**")
        st.write("Click the button below to send the radiograph to the FastAPI inference backend:")
        
        analyze_clicked = st.button("Analyze X-ray", type="primary", width="stretch")
        
        if analyze_clicked:
            with st.spinner("Analyzing radiograph through deep-learning models..."):
                try:
                    files = {"image": (uploaded_file.name, file_bytes, uploaded_file.type or "image/png")}
                    response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=60.0)
                    
                    if response.status_code == 200:
                        st.session_state["prediction_result"] = response.json()
                        st.success("Analysis complete!")
                    elif response.status_code == 400:
                        err_msg = response.json().get("error", "Invalid or corrupted image format.")
                        st.error(f"Validation Error: {err_msg}")
                    elif response.status_code == 503:
                        st.error("Backend Service Unavailable: The model is currently loading. Please retry in a few moments.")
                    else:
                        err_msg = response.json().get("error", "Analysis failed on server.")
                        st.error(f"Inference Error: {err_msg}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Connection Failed: Cannot reach FastAPI backend at {BACKEND_URL}. Please start the backend service.")
                except Exception as e:
                    st.error("An unexpected error occurred while communicating with the server.")

# --- RESULT SECTION ---
if st.session_state["prediction_result"] is not None:
    res = st.session_state["prediction_result"]
    st.markdown("---")
    st.markdown("## Diagnostic Results")
    
    # 1. Metric Cards
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Anatomical Region**")
        region_name = res.get("anatomical_region", "Unknown").upper()
        reg_conf = res.get("anatomical_confidence", 0.0)
        st.markdown(f"""
        <div class="metric-card-neutral">
            <h2 style="margin:0; color:#1e293b;">{region_name}</h2>
            <p style="margin:4px 0 0 0; color:#64748b; font-size:0.95rem;">Confidence: <b>{reg_conf*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("**Fracture Status**")
        is_fracture = res.get("fracture", False)
        frac_conf = res.get("fracture_confidence", 0.0)
        
        if is_fracture:
            status_text = "FRACTURE DETECTED"
            card_class = "metric-card-positive"
            status_color = "#dc2626"
        else:
            status_text = "NO FRACTURE (NORMAL)"
            card_class = "metric-card-negative"
            status_color = "#16a34a"
            
        st.markdown(f"""
        <div class="{card_class}">
            <h2 style="margin:0; color:{status_color};">{status_text}</h2>
            <p style="margin:4px 0 0 0; color:#64748b; font-size:0.95rem;">Fracture Probability: <b>{frac_conf*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("**Localization Status**")
        loc_avail = res.get("localization_available", False)
        loc_boxes = res.get("localization", None)
        
        if is_fracture and loc_avail and loc_boxes:
            loc_title = f"{len(loc_boxes)} LESION(S) LOCALIZED"
            loc_color = "#b91c1c"
            loc_card = "metric-card-positive"
        elif is_fracture:
            loc_title = "UNLOCALIZED"
            loc_color = "#d97706"
            loc_card = "metric-card-neutral"
        else:
            loc_title = "NOT APPLICABLE"
            loc_color = "#16a34a"
            loc_card = "metric-card-negative"
            
        st.markdown(f"""
        <div class="{loc_card}">
            <h2 style="margin:0; color:{loc_color};">{loc_title}</h2>
            <p style="margin:4px 0 0 0; color:#64748b; font-size:0.95rem;">Supported: <b>{loc_avail}</b></p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Generated Factual Caption
    st.markdown("### Generated Factual Caption")
    caption_text = res.get("caption", "No description available.")
    st.markdown(f'<div class="caption-box">"{caption_text}"</div>', unsafe_allow_html=True)

    # 3. Spatial Localization & Visual Overlay Section
    st.markdown("### Fracture Spatial Localization")
    
    if not is_fracture:
        st.info("Normal bone structure — fracture localization is not applicable for non-fractured radiographs.")
    elif loc_avail and loc_boxes:
        st.success(f"Spatial lesion localized with {len(loc_boxes)} bounding box region(s).")
        
        vis_url = res.get("visualization_url", None)
        if vis_url:
            full_vis_url = f"{BACKEND_URL}{vis_url}"
            try:
                vis_img_resp = requests.get(full_vis_url, timeout=10.0)
                if vis_img_resp.status_code == 200:
                    vis_img = Image.open(io.BytesIO(vis_img_resp.content))
                    st.image(vis_img, caption="Diagnostic Visual Overlay with Bounding Box Localization", width="stretch")
            except Exception:
                st.warning("Could not retrieve annotated visual overlay from backend.")
                
        # Display bounding box table
        box_data = []
        for idx, b_item in enumerate(loc_boxes):
            coords = b_item.get("box", b_item.get("box_2d", []))
            conf = b_item.get("confidence", frac_conf)
            box_data.append({
                "Lesion #": idx + 1,
                "Coordinates [x1, y1, x2, y2]": str(coords),
                "Confidence": f"{conf*100:.1f}%"
            })
        st.table(box_data)
    else:
        # Prompt requirement: If localization is unavailable: "Fracture localization is not available for this prediction."
        st.info("Fracture localization is not available for this prediction.")

    # 4. Raw Structured Response
    with st.expander("View Raw Structured JSON Response"):
        st.json(res)
