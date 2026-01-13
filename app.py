
import streamlit as st
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tempfile
# from src.predict import predict (Removed)
from src.utils import setup_logging

# Page Config
st.set_page_config(
    page_title="Momenta Deepfake Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CSS Styling (The "Sexy" Part)
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Global Imports */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Global Text Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117; /* Standard Dark Mode Background */
        color: #ffffff; /* FORCE WHITE TEXT */
    }

    /* Gradient Background - Very Subtle */
    .stApp {
        background: #0e1117;
    }

    /* Cards - Lighter grey to pop from background */
    .glass-card {
        background-color: #1e2329;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Headers - 100% White */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Highlight Spans (The "Yellow" request) */
    h1 span {
        color: #FFD700; /* Gold */
        background: none;
        -webkit-text-fill-color: #FFD700;
    }

    /* Paragraphs / Markdown text */
    p, li, div {
        color: #e6e6e6;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Buttons - High Visibility Gold/Yellow */
    .stButton > button {
        background-color: #FFD700; /* Gold */
        color: #000000; /* Black text on Gold = Max Contrast */
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #ffe066;
        color: #000000;
        transform: scale(1.02);
    }

    /* Metrics */
    div[data-testid="stMetricLabel"] {
        color: #b0b0b0;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 36px !important;
    }

    /* Inputs/Uploaders */
    .stFileUploader > div > div {
        background-color: #161b22;
        border: 2px dashed #30363d;
    }
    .stFileUploader > div > div:hover {
        border-color: #FFD700;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Functionality
# -------------------------------------------------------------

def plot_waveform(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.numpy()
    
    fig = plt.figure(figsize=(10, 3), facecolor='none')
    ax = fig.add_subplot(111)
    ax.plot(waveform[0], color='#3b82f6', alpha=0.7, linewidth=0.5)
    ax.axis('off')
    return fig

def plot_spectrogram(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    specgram = torchaudio.transforms.Spectrogram()(waveform)
    specgram_db = torchaudio.transforms.AmplitudeToDB()(specgram)
    
    fig = plt.figure(figsize=(10, 3), facecolor='none')
    ax = fig.add_subplot(111)
    ax.imshow(specgram_db[0].numpy(), origin='lower', aspect='auto', cmap='magma')
    ax.axis('off')
    return fig

def create_gauge(score):
    if score > 0.5:
        color = "#22c55e" # Green
        title = "REAL (Bonafide)"
    else:
        color = "#ef4444" # Red
        title = "FAKE (Spoof)"
        
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24, 'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [50, 100], 'color': 'rgba(34, 197, 94, 0.2)'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font = {'color': "white", 'family': "Outfit"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=250
    )
    return fig

# -------------------------------------------------------------
# Main Layout
# -------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/microphone.png", width=64)
    st.markdown("### Momenta Audio")
    st.markdown("Deepfake Detection System")
    st.markdown("---")
    st.markdown("**Version:** 1.0.0")
    st.markdown("**Model:** CNN-GRU")
    st.markdown("**Backend:** PyTorch")
    
    st.markdown("---")
    st.markdown("### Settings")
    threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    st.info("💡 Upload an audio file to analyze its authenticity. The model analyzes spectral patterns to detect artifacts.")

# Main Content
st.markdown("# Momenta <span>Deepfake Detector</span>", unsafe_allow_html=True)
st.markdown("### Secure your conversations against AI-generated voice attacks.")

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("#### 📂 Upload Analysis Target")
uploaded_file = st.file_uploader("", type=['mp3', 'wav', 'flac'])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile_path = tfile.name
    tfile.close()

    # Layout for analysis
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🎧 Playback")
        st.audio(tfile_path)
        
        st.markdown("#### 🌊 Waveform")
        try:
            st.pyplot(plot_waveform(tfile_path))
        except Exception:
            st.warning("Could not generate waveform visualization")
        st.markdown('</div>', unsafe_allow_html=True)

    from src.predict import DeepfakeDetector

    @st.cache_resource
    def load_detector():
        return DeepfakeDetector()

    detector = load_detector()

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Analysis Results")
        
        # DEBUG EXPANDER
        with st.expander("🛠️ System Diagnostics"):
            st.write(f"Current Dir: {os.getcwd()}")
            st.write(f"Model Path Config: {detector.model_path if detector else 'Not Loaded'}")
            if os.path.exists("models/cnn_gru_model.pth"):
                st.success(f"Model File Found ({os.path.getsize('models/cnn_gru_model.pth')} bytes)")
            else:
                st.error("CRITICAL: Model File NOT Found in 'models/'")
            
            try:
                import soundfile
                st.success("SoundFile Backend: Available")
            except ImportError as e:
                st.error(f"SoundFile Backend Missing: {e}")

        if st.button("Analyze Audio", key="analyze_btn"):
            if not detector:
                 st.error("Model failed to load. Check diagnostics.")
                 st.stop()
                 
            with st.spinner("Analyzing spectral artifacts..."):
                try:
                     # 1. Prediction
                    st.toast("Processing Audio...")
                    label, score = detector.predict(tfile_path)
                    
                    if label == "error":
                        st.error("Prediction failed inside the model. Check logs.")
                    else:
                        # 2. Display Gauge
                        st.plotly_chart(create_gauge(score), use_container_width=True)
                        
                        # 3. Details
                        st.markdown("---")
                        d_col1, d_col2 = st.columns(2)
                        d_col1.metric("Prediction", label.upper())
                        d_col2.metric("Confidence Score", f"{score:.4f}")
                        
                        # 4. Spectrogram
                        st.markdown("#### 📊 Spectral Analysis")
                        try:
                            st.pyplot(plot_spectrogram(tfile_path))
                        except Exception as e:
                            st.warning(f"Spectrogram failed (Non-critical): {e}")
                        
                        if label == "spoof":
                            st.error("⚠️ **Warning:** High probability of AI synthesis detected.")
                        else:
                            st.success("✅ **Verified:** Audio appears to be authentic.")
                        
                except Exception as e:
                    st.error(f"CRITICAL ERROR during analysis pipeline: {e}")
                    st.exception(e) # Print full trace to UI
        else:
            st.info("Click 'Analyze Audio' to start the deepfake detection process.")
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Cleanup (Optional: usually OS handles temp cleanup on reboot, but good practice to delete)
    # os.unlink(tfile_path) # Commented out to allow replays in one session logic
