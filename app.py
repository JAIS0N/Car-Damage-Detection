import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import plotly.express as px
from fpdf import FPDF

from car_damage_detector import CarDamageDetector

# ------------------------------------------------------
# PAGE CONFIG + CENTER WIDTH
# ------------------------------------------------------
st.set_page_config(page_title="Car Damage Assessment AI", layout="wide")

# Center page content
def centered_container(max_width=1200):
    st.markdown(
        f"""
        <style>
        .appview-container .main .block-container {{
            max-width: {max_width}px;
            margin-left: auto;
            margin-right: auto;
            padding-top: 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
centered_container()

# ------------------------------------------------------
# PREMIUM UI CSS
# ------------------------------------------------------
st.markdown("""
<style>
body { background-color: #0f0f0f !important; }

/* Title */
.main-title {
    font-size: 48px;
    text-align: center;
    font-weight: 900;
    margin-bottom: 15px;
    background: linear-gradient(60deg, #ff4b4b, #6c63ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Section headers */
.section-header {
    font-size: 30px;
    font-weight: 800;
    margin-top: 35px;
    margin-bottom: 12px;
    color: #eaeaea;
}

/* Summary title */
.summary-title {
    font-size: 34px;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.07);
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    backdrop-filter: blur(6px);
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}

.metric-title { color: #bcbcbc; font-size: 15px; }

.metric-value {
    font-size: 32px;
    font-weight: 900;
    background: linear-gradient(45deg, #ff4b4b, #6c63ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Details expander */
.streamlit-expanderHeader {
    font-size: 20px !important;
    color: #eaeaea !important;
    font-weight: 700 !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background-color: #1a1a1a !important; }
[data-testid="stSidebar"] h2, label { color: white !important; }

/* Upload box */
.upload-box {
    border: 2px dashed #6c63ff;
    padding: 26px;
    border-radius: 13px;
    background: rgba(255,255,255,0.05);
    text-align: center;
    color: #cccccc;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------
# PDF REPORT GENERATOR
# ------------------------------------------------------
def generate_pdf_report(detections, image_info, annotated_np):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Car Damage Assessment Report", ln=1, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    total_cost = sum([d['estimated_cost'] for d in detections])
    total_area = sum([d['area_percentage'] for d in detections])
    avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0

    pdf.cell(0, 8, f"Generated: {datetime.now()}", ln=1)
    pdf.cell(0, 8, f"Total Damages: {len(detections)}", ln=1)
    pdf.cell(0, 8, f"Avg Confidence: {avg_conf:.2f}", ln=1)
    pdf.cell(0, 8, f"Affected Area: {total_area:.2f}%", ln=1)
    pdf.cell(0, 8, f"Estimated Cost: ${total_cost}", ln=1)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Damage Breakdown:", ln=1)

    pdf.set_font("Arial", size=12)
    for d in detections:
        pdf.multi_cell(
            0, 8,
            f"{d['type'].title()} - {d['severity']}\n"
            f"Confidence: {d['confidence']}\n"
            f"Area: {d['area_percentage']}%\n"
            f"Cost: ${d['estimated_cost']}\n"
            f"Location: {d['location']}\n"
        )
        pdf.ln(1)

    pdf.add_page()
    pdf.cell(0, 10, "Annotated Image", ln=1)

    temp_path = "annotated_temp.jpg"
    Image.fromarray(annotated_np).save(temp_path, "JPEG")
    pdf.image(temp_path, x=10, y=30, w=180)

    return pdf.output(dest="S").encode("latin-1")


# ------------------------------------------------------
# MAIN UI
# ------------------------------------------------------
st.markdown("<div class='main-title'>🚗  Car Damage Assessment</div>", unsafe_allow_html=True)

# Sidebar — Settings
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

detector = CarDamageDetector(confidence_threshold=confidence)


# Upload Image
st.markdown("<div class='section-header'>Upload Vehicle Image</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Damage", use_container_width=True):
        with st.spinner("Running AI Damage Detection..."):
            result = detector.detect_damage(image)
            st.session_state.detections = result["damages"]
            st.session_state.annotated = detector.annotate_image(image, result["damages"])
            st.session_state.img_info = image.size


# ------------------------------------------------------
# RESULTS SECTION
# ------------------------------------------------------
if "detections" in st.session_state:

    detections = st.session_state.detections
    annotated = st.session_state.annotated

    # ---- SUMMARY ----
    st.markdown("<div class='summary-title'>Assessment Summary</div>", unsafe_allow_html=True)

    total_cost = sum(d["estimated_cost"] for d in detections)
    avg_conf = np.mean([d["confidence"] for d in detections])
    total_area = sum([d["area_percentage"] for d in detections])

    col1, col2, col3 = st.columns(3, gap="large")

    col1.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Total Damages</div>
            <div class='metric-value'>{len(detections)}</div>
        </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Avg Confidence</div>
            <div class='metric-value'>{avg_conf:.2%}</div>
        </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Estimated Repair Cost</div>
            <div class='metric-value'>${total_cost}</div>
        </div>
    """, unsafe_allow_html=True)

    # ---- SIDE-BY-SIDE IMAGE DISPLAY ----
    st.markdown("<div class='section-header'>Image Comparison</div>", unsafe_allow_html=True)

    img_col1, img_col2 = st.columns([1, 1], gap="large")

    with img_col1:
        st.markdown("<h5 style='color:#cccccc;text-align:center;'>Uploaded Image</h5>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)

    with img_col2:
        st.markdown("<h5 style='color:#cccccc;text-align:center;'>AI-Annotated Image</h5>", unsafe_allow_html=True)
        st.image(annotated, use_column_width=True)

    # ---- CHARTS ----
    st.markdown("<div class='section-header'>Damage Analysis Charts</div>", unsafe_allow_html=True)

    pie = px.pie(
        names=[d["type"].title() for d in detections],
        values=[d["area_percentage"] for d in detections],
        hole=0.35,
        title="Damage Type Distribution",
        color_discrete_sequence=[
            "#FF6B6B", "#6C63FF", "#4ECDC4",
            "#FFD93D", "#00A8E8", "#9D4EDD"
        ]
    )
    pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))

    bar = px.bar(
        x=[d["type"].title() for d in detections],
        y=[d["area_percentage"] for d in detections],
        color=[d["severity"] for d in detections],
        title="Damage Severity Distribution",
        color_discrete_map={
            "Light": "#4ECDC4",
            "Moderate": "#FFD93D",
            "Severe": "#FF6B6B"
        }
    )
    bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))

    chart1, chart2 = st.columns(2)
    chart1.plotly_chart(pie, use_column_width=True)
    chart2.plotly_chart(bar, use_column_width=True)

    # ---- DETAILS ----
    st.markdown("<div class='section-header'>Damage Details</div>", unsafe_allow_html=True)

    for d in detections:
        with st.expander(f"🔧 {d['type'].title()} ({d['severity']})"):
            st.write(f"Confidence: {d['confidence']}")
            st.write(f"Area Affected: {d['area_percentage']}%")
            st.write(f"Estimated Cost: ${d['estimated_cost']}")
            st.write(f"Location: {d['location']}")

    # ---- REPORT DOWNLOAD ----
    st.markdown("<div class='section-header'>Download Full Report</div>", unsafe_allow_html=True)

    pdf_bytes = generate_pdf_report(detections, st.session_state.img_info, annotated)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="Car_Damage_Report.pdf",
        mime="application/pdf"
    )
