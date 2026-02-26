"""
app.py
======
Streamlit web interface for the AI Tuberculosis Detection and Severity Assessment System.

This app allows users to:
  1. Upload a chest X-ray image.
  2. Run the YOLO-based TB detection pipeline.
  3. View the predicted severity level and risk score.
  4. View the generated medical report and annotated image.
  5. Provide expert feedback for reinforcement learning (RL) updates.
"""

import os
import time
import json
import tempfile
from PIL import Image
import streamlit as st

# ── Ensure project root is on Python path if running from outside ─────────────
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from inference.detect_tb import TBDetector
from severity.severity_calculator import SeverityCalculator
from reports.report_generator import ReportGenerator
from reinforcement.rl_update import RLUpdateEngine

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TB Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for better aesthetics ──────────────────────────────────────────
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0A0A0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .severity-severe { color: #FF5252 !important; }
    .severity-moderate { color: #FFD740 !important; }
    .severity-mild { color: #4CAF50 !important; }
    .severity-none { color: #2196F3 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load AI models once globally."""
    detector = TBDetector()
    calculator = SeverityCalculator()
    generator = ReportGenerator()
    rl_engine = RLUpdateEngine(calculator)
    return detector, calculator, generator, rl_engine

try:
    detector, calculator, generator, rl_engine = load_models()
except Exception as e:
    st.error(f"Failed to load AI models: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3209/3209114.png", width=60)
    st.title("Settings & Info")
    st.markdown("---")
    
    st.info(
        "**MEDICAL DISCLAIMER**\n\n"
        "This system is for RESEARCH AND EDUCATIONAL USE ONLY. "
        "It must NOT be used for clinical diagnosis or treatment decisions."
    )
    
    st.markdown("---")
    st.subheader("Current Thresholds (RL)")
    
    # Reload config dynamically to show updated RL bounds
    import importlib
    importlib.reload(config)
    calculator = SeverityCalculator() # reload to update internal bounds
    
    st.write(f"- **Mild max area:** {calculator.mild_max:.2f}%")
    st.write(f"- **Mod max area:** {calculator.moderate_max:.2f}%")
    
    st.markdown("---")
    st.write(f"Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
    st.write(f"IOU Threshold: {config.IOU_THRESHOLD}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
st.title("🫁 AI Tuberculosis Detection & Severity Assessment")
st.markdown("Upload a chest X-ray to automatically detect TB lesions, calculate severity, and generate a medical report.")

uploaded_file = st.file_uploader("Choose a Chest X-ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display original image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        # Read the file as bytes
        image_bytes = uploaded_file.read()
        st.image(image_bytes, use_column_width=True)
    
    # Process button
    if st.button("🚀 Analyze X-Ray", use_container_width=True, type="primary"):
        with st.spinner("Running AI Analysis..."):
            
            # Save uploaded file to temp file for YOLO processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            try:
                # ── Pipeline Step 1: Detection ──
                start_time = time.time()
                detection = detector.detect(tmp_path)
                
                # ── Pipeline Step 2: Severity ──
                if detection["tb_detected"]:
                    severity_result = calculator.calculate(
                        infected_area_percent=detection["infected_area_percent"],
                        lesion_count=detection["lesion_count"],
                        confidence_avg=detection["confidence_avg"]
                    )
                    severity_level = severity_result["severity_level"]
                    risk_score = severity_result["risk_score"]
                else:
                    severity_level = "None"
                    risk_score = 0
                
                # ── Pipeline Step 3: Report & Visual ──
                report = generator.generate_report(detection)
                
                # Generate visual output file
                out_filename = f"out_{os.path.basename(tmp_path)}"
                out_path = os.path.join(config.OUTPUT_DIR, out_filename)
                
                # Ensure output dir exists
                os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                generator.generate_visual_output(tmp_path, detection, report, output_path=out_path)
                
                end_time = time.time()
                process_time = end_time - start_time
                
                # Store results in session state so they persist for RL feedback
                st.session_state["analyzed"] = True
                st.session_state["detection"] = detection
                st.session_state["severity_level"] = severity_level
                st.session_state["risk_score"] = risk_score
                st.session_state["report"] = report
                st.session_state["out_path"] = out_path
                st.session_state["tmp_path"] = tmp_path
                st.session_state["process_time"] = process_time
                st.session_state["rl_engine"] = rl_engine
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
            finally:
                # Clean up original temp file
                if os.path.exists(tmp_path) and "tmp_path" not in st.session_state:
                     os.remove(tmp_path)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.get("analyzed", False):
    st.success(f"Analysis complete in {st.session_state['process_time']:.2f} seconds.")
    
    detection = st.session_state["detection"]
    severity_level = st.session_state["severity_level"]
    risk_score = st.session_state["risk_score"]
    report = st.session_state["report"]
    out_path = st.session_state["out_path"]
    
    st.markdown("---")
    
    col_img, col_metrics = st.columns([1.2, 1])
    
    with col_img:
        st.subheader("AI Annotated Output")
        if os.path.exists(out_path):
            st.image(out_path, use_column_width=True)
        else:
            st.warning("Visual output not found.")
            
    with col_metrics:
        st.subheader("Diagnosis Metrics")
        
        # Color code severity
        sev_class = severity_level.lower()
        
        # Dashboard cards
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TB Detected</div>
                <div class="metric-value">{"YES" if detection["tb_detected"] else "NO"}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Severity</div>
                <div class="metric-value severity-{sev_class}">{severity_level}</div>
            </div>
            """, unsafe_allow_html=True)
            
        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{risk_score}/100</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Affected Area</div>
                <div class="metric-value">{detection["infected_area_percent"]:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Download Report Button
        report_json = json.dumps(report, indent=4)
        st.download_button(
            label="📄 Download Medical Report (JSON)",
            data=report_json,
            file_name=f"report_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )
        
        # RL Feedback Section
        st.markdown("---")
        st.write("### 👨‍⚕️ Expert Feedback (RL Update)")
        st.write("If the AI severity prediction is incorrect, provide the correct label to adjust the model's internal thresholds via Reinforcement Learning.")
        
        expert_options = ["None", "Mild", "Moderate", "Severe"]
        current_idx = expert_options.index(severity_level) if severity_level in expert_options else 0
        
        expert_fb = st.selectbox("Correct Severity Level:", expert_options, index=current_idx)
        
        if st.button("Submit Expert Feedback", type="secondary", use_container_width=True):
            if not detection["tb_detected"] and expert_fb != "None":
                 st.warning("Cannot apply severity feedback when no lesions are detected.")
            elif expert_fb == severity_level:
                st.info("Feedback matches AI prediction. RL thresholds reinforced.")
                st.session_state["rl_engine"].process_feedback(detection, severity_level, expert_fb)
                st.balloons()
            else:
                try:
                    rl_record = st.session_state["rl_engine"].process_feedback(
                        detection_result=detection,
                        predicted_severity=severity_level,
                        expert_severity=expert_fb
                    )
                    st.success(f"Feedback adopted! System adjusted thresholds towards {expert_fb}.")
                    st.json(rl_record)
                    st.balloons()
                except Exception as e:
                    st.error(f"Failed to apply RL feedback: {e}")
