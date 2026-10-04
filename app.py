"""
================================================================================
🎙️ SPEECH EMOTION RECOGNITION PLATFORM - ADVANCED AI ANALYTICS DASHBOARD
================================================================================
A production-grade Deep Learning platform for Speech Emotion Recognition (SER),
acoustic feature extraction, multi-class classification, segment timeline analysis,
prediction history database, automated PDF report generation, and model insights.
"""
import os
import sys
import io
import time
import json
import uuid
import base64
from datetime import datetime

# Configure UTF-8 encoding for standard streams on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import config
from src.predict import EmotionPredictor
from src.utils import get_emotion_color
from src.database import (
    init_db, save_analysis, get_all_analyses, get_analysis_by_id,
    delete_analysis, clear_all_history, get_summary_stats, export_history_dataframe
)
from src.audio_analyzer import (
    compute_acoustic_features, extract_mfcc_matrix, segment_and_predict
)
from src.report_generator import generate_pdf_report
from src.evaluation import (
    get_model_evaluation_metrics, plot_interactive_confusion_matrix,
    plot_per_class_f1_bars, get_model_architecture_comparison
)
from src.video_processor import VideoProcessor
from src.lip_sync_detector import LipSyncDetector
from src.visual_speech_recognizer import VisualSpeechRecognizerAdapter
from src.multimodal_fusion import MultimodalFusionEngine
from src.hinglish_engine import HinglishEngine
from src.audio_transcriber import AudioTranscriber
from src.subtitle_exporter import SubtitleExporter

# -----------------------------------------------------------------------------
# Configuration Constants & Emotion Metadata
# -----------------------------------------------------------------------------
EMOTION_META = {
    'happy': {'emoji': '😊', 'label': 'Happy', 'color': '#FBBF24', 'bg': 'rgba(251, 191, 36, 0.15)'},
    'sad': {'emoji': '😢', 'label': 'Sad', 'color': '#60A5FA', 'bg': 'rgba(96, 165, 250, 0.15)'},
    'angry': {'emoji': '😠', 'label': 'Angry', 'color': '#F87171', 'bg': 'rgba(248, 113, 113, 0.15)'},
    'fear': {'emoji': '😨', 'label': 'Fear', 'color': '#C084FC', 'bg': 'rgba(192, 132, 252, 0.15)'},
    'neutral': {'emoji': '😐', 'label': 'Neutral', 'color': '#94A3B8', 'bg': 'rgba(148, 163, 184, 0.15)'},
    'surprise': {'emoji': '😲', 'label': 'Surprise', 'color': '#F472B6', 'bg': 'rgba(244, 114, 182, 0.15)'},
    'disgust': {'emoji': '🤢', 'label': 'Disgust', 'color': '#34D399', 'bg': 'rgba(52, 211, 153, 0.15)'},
}

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database on Startup
init_db()

# -----------------------------------------------------------------------------
# Custom CSS - Dark Theme System (#0B1120, #172338, #38BDF8)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0B1120;
        --bg-card: #172338;
        --bg-card-hover: #1E2E4A;
        --bg-subtle: #0F172A;
        --text-primary: #F1F5F9;
        --text-secondary: #CBD5E1;
        --text-muted: #94A3B8;
        --accent: #38BDF8;
        --accent-glow: rgba(56, 189, 248, 0.25);
        --border-subtle: #223354;
        --border-hover: #38BDF8;
        --radius-lg: 14px;
        --radius-md: 10px;
        --radius-sm: 6px;
    }

    html, body, [class*="css"], .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    p, span, label, li {
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* Top Dashboard Header */
    .dashboard-header {
        background: linear-gradient(135deg, #172338 0%, #0F172A 100%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 24px 30px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(180deg, #38BDF8, #818CF8);
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.12);
        color: var(--accent) !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
    .header-title {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin: 0 0 6px 0 !important;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 0.98rem;
        color: var(--text-secondary);
        margin: 0;
        max-width: 850px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-subtle) !important;
        border-right: 1px solid var(--border-subtle) !important;
        padding-top: 1rem;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        margin-bottom: 18px;
    }
    .sidebar-brand-icon {
        font-size: 1.8rem;
        background: rgba(56, 189, 248, 0.15);
        padding: 8px;
        border-radius: 8px;
        line-height: 1;
    }
    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin: 0;
    }
    .sidebar-brand-desc {
        font-size: 0.75rem;
        color: var(--text-muted) !important;
        margin: 0;
    }

    /* Navigation Radio Items in Sidebar */
    [data-testid="stSidebar"] .stRadio > label {
        display: none !important;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 6px;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
        background-color: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 10px 14px;
        margin: 0;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
        border-color: var(--accent);
        background-color: var(--bg-card-hover);
    }
    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] * {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Collapsible About Section */
    [data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        margin-bottom: 16px !important;
        transition: all 0.25s ease !important;
        overflow: hidden;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(56, 189, 248, 0.35) !important;
    }
    [data-testid="stExpander"] summary {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 10px 14px !important;
        background-color: var(--bg-card) !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--accent) !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 14px !important;
        background-color: #121D30 !important;
        border-top: 1px solid var(--border-subtle) !important;
    }

    /* Supported Emotions Responsive Grid */
    .emotions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(76px, 1fr));
        gap: 6px;
        margin: 8px 0 12px 0;
    }
    .emotion-chip {
        background: #0B1120;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 6px 4px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .emotion-chip:hover {
        transform: translateY(-2px);
        border-color: var(--accent);
        background: #172338;
    }
    .chip-emoji {
        font-size: 1.3rem;
        line-height: 1.2;
        margin-bottom: 2px;
    }
    .chip-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-secondary);
    }

    /* Guide Steps */
    .guide-step {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    .step-num {
        background: rgba(56, 189, 248, 0.15);
        color: var(--accent);
        border: 1px solid rgba(56, 189, 248, 0.3);
        font-weight: 700;
        font-size: 0.7rem;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
    }

    /* Executive KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 22px;
    }
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 18px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--accent) !important;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%) !important;
        color: #0B1120 !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.45) !important;
        background: linear-gradient(135deg, #38BDF8 0%, #7DD3FC 100%) !important;
        color: #0B1120 !important;
    }

    /* Native Audio Input & File Uploader */
    [data-testid="stAudioInput"], [data-testid="stFileUploader"] {
        background-color: var(--bg-card) !important;
        border: 2px dashed var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 18px !important;
        transition: border-color 0.25s ease !important;
        margin-bottom: 18px !important;
    }
    [data-testid="stAudioInput"]:hover, [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
    }
    [data-testid="stAudioInput"] *, [data-testid="stFileUploader"] * {
        color: var(--text-secondary) !important;
    }

    /* Hero Emotion Prediction Card */
    .hero-emotion-card {
        background: linear-gradient(145deg, #172338 0%, #0F172A 100%);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 26px;
        margin: 20px 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 12px;
    }
    .hero-flex {
        display: flex;
        align-items: center;
        gap: 22px;
        flex-wrap: wrap;
    }
    .hero-emoji-container {
        font-size: 3.8rem;
        line-height: 1;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 18px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .hero-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin: 0 0 6px 0 !important;
    }
    .hero-confidence-meter {
        width: 100%;
        height: 9px;
        background: #0B1120;
        border-radius: 10px;
        overflow: hidden;
        margin: 8px 0;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .hero-confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Content Cards */
    .content-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    .content-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric Pills */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 10px;
        margin-top: 10px;
    }
    .metric-pill {
        background: var(--bg-subtle);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 12px 10px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--accent) !important;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 2px;
    }

    /* Tabs Styling */
    [data-baseweb="tab-list"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        padding: 5px !important;
        gap: 6px !important;
        margin-bottom: 18px !important;
    }
    [data-baseweb="tab"] {
        border-radius: var(--radius-sm) !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 8px 16px !important;
        background: transparent !important;
        border: none !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--accent) !important;
        color: #0B1120 !important;
        font-weight: 700 !important;
    }

    /* Alert / Warning Boxes */
    [data-testid="stAlert"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stAlert"] * {
        color: var(--text-primary) !important;
    }

    /* =========================================================================
       VOICEMIND AI LANDING PAGE DESIGN SYSTEM (#030B1B, #091A33, #00D4FF, #2677FF, #8B5CF6)
       ========================================================================= */
    .vm-landing-wrapper {
        background: radial-gradient(circle at 85% 15%, rgba(38, 119, 255, 0.16) 0%, transparent 45%),
                    radial-gradient(circle at 15% 45%, rgba(139, 92, 246, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 50% 85%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
                    linear-gradient(180deg, #030B1B 0%, #07162D 100%);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 24px;
        padding: 28px 34px 40px 34px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    
    /* Navigation Bar */
    .vm-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background: rgba(9, 26, 51, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 212, 255, 0.22);
        border-radius: 40px;
        margin-bottom: 36px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    .vm-logo-group {
        display: flex;
        align-items: center;
        gap: 12px;
        text-decoration: none !important;
    }
    .vm-wave-icon {
        display: flex;
        align-items: center;
        gap: 3px;
        height: 24px;
    }
    .vm-wave-bar {
        width: 3.5px;
        background: linear-gradient(180deg, #00D4FF 0%, #8B5CF6 100%);
        border-radius: 3px;
        animation: vmPulse 1.4s ease-in-out infinite alternate;
    }
    .vm-wave-bar:nth-child(1) { height: 12px; animation-delay: 0.1s; }
    .vm-wave-bar:nth-child(2) { height: 22px; animation-delay: 0.3s; }
    .vm-wave-bar:nth-child(3) { height: 16px; animation-delay: 0.2s; }
    .vm-wave-bar:nth-child(4) { height: 26px; animation-delay: 0.4s; }
    .vm-wave-bar:nth-child(5) { height: 14px; animation-delay: 0.15s; }
    @keyframes vmPulse {
        0% { transform: scaleY(0.5); opacity: 0.7; }
        100% { transform: scaleY(1.1); opacity: 1; }
    }
    .vm-brand-name {
        font-size: 1.25rem;
        font-weight: 800;
        color: #FFFFFF !important;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .vm-brand-name span {
        color: #00D4FF;
    }
    .vm-brand-tagline {
        font-size: 0.7rem;
        color: #94A3B8;
        letter-spacing: 0.05em;
        display: block;
    }
    .vm-nav-links {
        display: flex;
        align-items: center;
        gap: 28px;
    }
    .vm-nav-link {
        color: #CBD5E1 !important;
        font-size: 0.9rem;
        font-weight: 500;
        text-decoration: none !important;
        transition: all 0.2s ease;
        position: relative;
        padding: 4px 0;
    }
    .vm-nav-link:hover, .vm-nav-link.active {
        color: #00D4FF !important;
    }
    .vm-nav-link.active::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 100%;
        height: 2px;
        background: #00D4FF;
        border-radius: 2px;
        box-shadow: 0 0 8px #00D4FF;
    }
    .vm-btn-cta-nav {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.4);
        border-radius: 24px;
        color: #00D4FF !important;
        font-size: 0.85rem;
        font-weight: 600;
        text-decoration: none !important;
        transition: all 0.25s ease;
        box-shadow: 0 0 14px rgba(0, 212, 255, 0.15);
    }
    .vm-btn-cta-nav:hover {
        background: #00D4FF;
        color: #030B1B !important;
        box-shadow: 0 0 22px rgba(0, 212, 255, 0.45);
        transform: translateY(-1px);
    }

    /* Hero Section */
    .vm-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.35);
        border-radius: 30px;
        padding: 6px 16px;
        color: #00D4FF;
        font-size: 0.84rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 18px;
        box-shadow: 0 0 16px rgba(0, 212, 255, 0.12);
    }
    .vm-hero-title {
        font-size: 3.4rem !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        line-height: 1.12 !important;
        margin: 0 0 18px 0 !important;
        letter-spacing: -0.03em !important;
    }
    .vm-gradient-text {
        background: linear-gradient(135deg, #00D4FF 0%, #2677FF 45%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }
    .vm-hero-desc {
        font-size: 1.05rem;
        color: #B8CCE6;
        line-height: 1.65;
        margin-bottom: 26px;
        max-width: 580px;
    }

    /* Feature Indicators Pods */
    .vm-indicators-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 32px;
    }
    .vm-indicator-pod {
        background: rgba(9, 26, 51, 0.7);
        border: 1px solid rgba(0, 212, 255, 0.18);
        border-radius: 12px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: all 0.25s ease;
    }
    .vm-indicator-pod:hover {
        border-color: #00D4FF;
        background: rgba(9, 26, 51, 0.95);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 212, 255, 0.14);
    }
    .vm-pod-icon {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: rgba(0, 212, 255, 0.12);
        border: 1px solid rgba(0, 212, 255, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .vm-pod-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .vm-pod-sub {
        font-size: 0.72rem;
        color: #94A3B8;
        line-height: 1.2;
        margin-top: 2px;
    }

    /* Hero Right Visual Presentation */
    .vm-visual-wrapper {
        position: relative;
        background: #091A33;
        border: 1px solid rgba(0, 212, 255, 0.28);
        border-radius: 20px;
        padding: 14px;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.5), 0 0 35px rgba(0, 212, 255, 0.12);
        overflow: hidden;
    }
    .vm-visual-img {
        width: 100%;
        height: auto;
        border-radius: 14px;
        display: block;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .vm-floating-card {
        position: absolute;
        background: rgba(9, 26, 51, 0.88);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 212, 255, 0.35);
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.45);
        z-index: 2;
    }
    .vm-floating-hinglish {
        top: 24px;
        right: 24px;
        width: 200px;
    }
    .vm-floating-emotions {
        bottom: 24px;
        right: 24px;
        width: 220px;
    }

    /* Capability Section & Cards */
    .vm-section-tag {
        font-size: 0.82rem;
        font-weight: 800;
        color: #00D4FF;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .vm-section-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        margin: 0 0 26px 0 !important;
        letter-spacing: -0.02em !important;
    }
    .vm-capability-card {
        background: linear-gradient(180deg, #091A33 0%, #061326 100%);
        border: 1px solid rgba(38, 119, 255, 0.28);
        border-radius: 16px;
        padding: 22px 20px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
        position: relative;
        overflow: hidden;
    }
    .vm-capability-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #00D4FF, #8B5CF6);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .vm-capability-card:hover {
        border-color: #00D4FF;
        transform: translateY(-4px);
        box-shadow: 0 18px 40px rgba(0, 212, 255, 0.16);
    }
    .vm-capability-card:hover::before {
        opacity: 1;
    }
    .vm-card-icon-wrap {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: rgba(0, 212, 255, 0.12);
        border: 1px solid rgba(0, 212, 255, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        margin-bottom: 14px;
    }
    .vm-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0 0 8px 0;
    }
    .vm-card-desc {
        font-size: 0.84rem;
        color: #B8CCE6;
        line-height: 1.55;
        margin-bottom: 14px;
    }
    .vm-card-bullets {
        font-size: 0.8rem;
        color: #94A3B8;
        padding-left: 18px;
        margin-bottom: 18px;
    }
    .vm-card-bullets li {
        margin-bottom: 4px;
    }

    /* Live Interactive Demo Box */
    .vm-demo-console {
        background: #091A33;
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 18px;
        padding: 24px;
        margin: 30px 0;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
    }

    /* Technical Grid */
    .vm-tech-tile {
        background: rgba(9, 26, 51, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        transition: all 0.2s ease;
    }
    .vm-tech-tile:hover {
        border-color: #38BDF8;
        background: rgba(9, 26, 51, 0.85);
    }

    .dashboard-footer {
        border-top: 1px solid var(--border-subtle);
        padding: 24px 0 12px 0;
        margin-top: 40px;
        text-align: center;
        font-size: 0.82rem;
        color: var(--text-muted);
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Base64 Image Helper for Embedded Visuals
# -----------------------------------------------------------------------------
def get_image_base64(image_path: str) -> str:
    """Reads image from disk and returns standard data URI string"""
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(image_path)[1].lstrip('.').lower()
                if ext == 'jpg':
                    ext = 'jpeg'
                return f"data:image/{ext};base64,{encoded}"
        except Exception:
            return ""
    return ""


# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    """Load the emotion predictor model safely"""
    try:
        predictor = EmotionPredictor()
        return predictor
    except Exception as e:
        return None


# -----------------------------------------------------------------------------
# Hardware Microphone Recorder Helper (PyAudio)
# -----------------------------------------------------------------------------
def record_from_hardware_mic(duration=3, sample_rate=22050):
    """Record audio directly from system microphone using PyAudio"""
    import pyaudio
    import wave
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=sample_rate,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        total_chunks = int(sample_rate / CHUNK * duration)
        
        for _ in range(total_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        bio = io.BytesIO()
        wf = wave.open(bio, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        bio.seek(0)
        return bio.getvalue()
    except Exception as e:
        p.terminate()
        raise e


# -----------------------------------------------------------------------------
# Plotting Helper Functions
# -----------------------------------------------------------------------------
def plot_waveform(audio, sr):
    """Plot time-domain audio waveform in dark dashboard theme"""
    fig, ax = plt.subplots(figsize=(10, 3.0), facecolor='#172338')
    ax.set_facecolor('#0F172A')
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time_axis, audio, color='#38BDF8', linewidth=1.0, alpha=0.95)
    ax.set_xlabel('Time (seconds)', fontsize=10, color='#CBD5E1', fontweight='500', labelpad=6)
    ax.set_ylabel('Amplitude', fontsize=10, color='#CBD5E1', fontweight='500', labelpad=6)
    ax.set_title('Time-Domain Audio Waveform', fontsize=12, fontweight='bold', color='#F1F5F9', pad=10)
    ax.tick_params(colors='#94A3B8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#223354')
    ax.grid(True, linestyle='--', alpha=0.2, color='#475569')
    plt.tight_layout()
    return fig


def plot_spectrogram(audio, sr):
    """Plot mel spectrogram in dark dashboard theme"""
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor='#172338')
    ax.set_facecolor('#0F172A')
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_mels=getattr(config, 'N_MELS', 128),
        hop_length=getattr(config, 'HOP_LENGTH', 512)
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(
        mel_spec_db, sr=sr, hop_length=getattr(config, 'HOP_LENGTH', 512),
        x_axis='time', y_axis='mel', cmap='magma', ax=ax
    )
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='#94A3B8', labelcolor='#CBD5E1', labelsize=8)
    cbar.outline.set_edgecolor('#223354')
    cbar.set_label('Energy (dB)', color='#CBD5E1', fontsize=9, labelpad=6)
    ax.set_xlabel('Time (seconds)', fontsize=10, color='#CBD5E1', fontweight='500', labelpad=6)
    ax.set_ylabel('Frequency (Hz)', fontsize=10, color='#CBD5E1', fontweight='500', labelpad=6)
    ax.set_title('Mel-Frequency Spectrogram (128 Bands)', fontsize=12, fontweight='bold', color='#F1F5F9', pad=10)
    ax.tick_params(colors='#94A3B8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#223354')
    plt.tight_layout()
    return fig


def plot_mfcc_heatmap(audio, sr):
    """Plot 2D MFCC matrix heatmap across time"""
    fig, ax = plt.subplots(figsize=(10, 3.0), facecolor='#172338')
    ax.set_facecolor('#0F172A')
    mfcc_mat = extract_mfcc_matrix(audio, sr, n_mfcc=40)
    img = librosa.display.specshow(
        mfcc_mat, sr=sr, hop_length=getattr(config, 'HOP_LENGTH', 512),
        x_axis='time', cmap='viridis', ax=ax
    )
    cbar = fig.colorbar(img, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color='#94A3B8', labelcolor='#CBD5E1', labelsize=8)
    cbar.outline.set_edgecolor('#223354')
    cbar.set_label('MFCC Coeff Value', color='#CBD5E1', fontsize=9, labelpad=6)
    ax.set_xlabel('Time (seconds)', fontsize=10, color='#CBD5E1', fontweight='500', labelpad=6)
    ax.set_ylabel('MFCC Index (1-40)', fontsize=10, color='#CBD5E1', fontweight='500', labelpad=6)
    ax.set_title('Mel-Frequency Cepstral Coefficients (40 MFCCs)', fontsize=12, fontweight='bold', color='#F1F5F9', pad=10)
    ax.tick_params(colors='#94A3B8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#223354')
    plt.tight_layout()
    return fig


def plot_emotion_bars(probabilities):
    """Plot emotion probabilities as sleek horizontal bars in dark theme"""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    emotions = []
    probs = []
    colors = []
    for emotion_name, prob_val in sorted_probs:
        meta = EMOTION_META.get(emotion_name.lower(), {'emoji': '🎙️', 'label': emotion_name.capitalize(), 'color': '#38BDF8'})
        emotions.append(f"{meta['emoji']} {meta['label']}")
        probs.append(prob_val * 100)
        colors.append(meta['color'])
    emotions_rev = emotions[::-1]
    probs_rev = probs[::-1]
    colors_rev = colors[::-1]
    
    fig = go.Figure(go.Bar(
        x=probs_rev,
        y=emotions_rev,
        orientation='h',
        marker=dict(color=colors_rev, line=dict(color='rgba(255, 255, 255, 0.15)', width=1), cornerradius=6),
        text=[f'<b>{p:.1f}%</b>' for p in probs_rev],
        textposition='outside',
        textfont=dict(color='#F1F5F9', size=11, family='Inter, sans-serif'),
        cliponaxis=False
    ))
    max_val = max(probs) if probs else 100
    fig.update_layout(
        title=dict(text="<b>Emotion Probability Breakdown</b>", font=dict(size=14, color='#F1F5F9', family='Inter, sans-serif'), x=0.01),
        xaxis=dict(
            title=dict(text="Probability (%)", font=dict(color='#CBD5E1', size=11)),
            tickfont=dict(color='#94A3B8', size=10),
            range=[0, min(100, max_val + 14)],
            showgrid=True, gridcolor='#223354', zeroline=False
        ),
        yaxis=dict(tickfont=dict(color='#F1F5F9', size=11, family='Inter, sans-serif'), showgrid=False),
        paper_bgcolor='#172338', plot_bgcolor='#0F172A',
        height=340, margin=dict(l=15, r=40, t=45, b=35), showlegend=False
    )
    return fig


def plot_emotion_timeline(segments):
    """Plot temporal segment-wise emotion classification timeline"""
    if not segments:
        return None
    df_seg = pd.DataFrame(segments)
    df_seg['MidTime'] = (df_seg['start_time'] + df_seg['end_time']) / 2.0
    df_seg['ConfPct'] = df_seg['confidence'] * 100
    df_seg['EmotionLabel'] = df_seg['predicted_emotion'].str.capitalize()
    df_seg['Color'] = [EMOTION_META.get(e.lower(), {}).get('color', '#38BDF8') for e in df_seg['predicted_emotion']]
    
    fig = go.Figure()
    
    # Confidence Line
    fig.add_trace(go.Scatter(
        x=df_seg['MidTime'],
        y=df_seg['ConfPct'],
        mode='lines+markers+text',
        line=dict(color='#38BDF8', width=2.5, shape='spline'),
        marker=dict(size=12, color=df_seg['Color'], line=dict(color='#FFFFFF', width=1.5)),
        text=[f"{EMOTION_META.get(e.lower(), {}).get('emoji', '')} {e.capitalize()}" for e in df_seg['predicted_emotion']],
        textposition="top center",
        textfont=dict(color='#F1F5F9', size=11),
        hoverinfo='text',
        hovertext=[f"Segment: {row['start_time']}s - {row['end_time']}s<br>Emotion: {row['EmotionLabel']}<br>Confidence: {row['ConfPct']:.1f}%" for _, row in df_seg.iterrows()]
    ))
    
    fig.update_layout(
        title=dict(text="<b>Temporal Emotion Trajectory (Segment-Wise Timeline)</b>", font=dict(size=14, color='#F1F5F9', family='Inter, sans-serif'), x=0.01),
        xaxis=dict(title=dict(text="Timeline (Seconds)", font=dict(color='#CBD5E1', size=11)), tickfont=dict(color='#94A3B8', size=10), gridcolor='#223354'),
        yaxis=dict(title=dict(text="Segment Confidence (%)", font=dict(color='#CBD5E1', size=11)), tickfont=dict(color='#94A3B8', size=10), range=[0, 115], gridcolor='#223354'),
        paper_bgcolor='#172338', plot_bgcolor='#0F172A',
        height=320, margin=dict(l=20, r=20, t=50, b=40), showlegend=False
    )
    return fig


def render_emotion_hero(emotion, confidence, is_low_confidence=False, conf_threshold=40):
    """Render hero emotion prediction card"""
    meta = EMOTION_META.get(emotion.lower(), {'emoji': '🎙️', 'label': emotion.capitalize(), 'color': '#38BDF8', 'bg': 'rgba(56, 189, 248, 0.15)'})
    accent_color = meta['color']
    
    warning_badge = ""
    if is_low_confidence:
        warning_badge = f"""
            <div style="background: rgba(248, 113, 113, 0.15); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.3); padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; display: inline-block; margin-left: 8px;">
                ⚠️ Low Confidence (< {conf_threshold}%)
            </div>
        """
        
    st.markdown(f"""
        <div class="hero-emotion-card" style="border-top: 4px solid {accent_color};">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <div class="hero-badge" style="background: {meta['bg']}; color: {accent_color}; border: 1px solid {accent_color}40;">
                    Primary Detected Vocal Emotion
                </div>
                {warning_badge}
            </div>
            <div class="hero-flex">
                <div class="hero-emoji-container">
                    <span>{meta['emoji']}</span>
                </div>
                <div class="hero-info">
                    <h2 class="hero-title" style="color: #FFFFFF !important;">
                        {emotion.upper()}
                    </h2>
                    <div class="hero-confidence-meter">
                        <div class="hero-confidence-fill" style="width: {confidence*100:.1f}%; background: linear-gradient(90deg, {accent_color}, #38BDF8);"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; font-size: 0.86rem; color: #CBD5E1; margin-top: 6px;">
                        <span>Classification Confidence: <strong style="color: {accent_color}; font-size: 1.05rem;">{confidence*100:.2f}%</strong></span>
                        <span style="background: rgba(255,255,255,0.06); padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; border: 1px solid rgba(255,255,255,0.08);">
                            CNN-LSTM Hybrid • 128 Mel Bands
                        </span>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def plot_dual_sync_timeline(timestamps, audio_env, mar_series, vel_series):
    """Plot synchronized dual timeline of Acoustic Energy vs Visual Lip Kinematics (MAR)"""
    if not timestamps or len(timestamps) == 0:
        return None

    fig = go.Figure()

    # Audio Energy Envelope
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=audio_env,
        name='Acoustic RMS Energy',
        mode='lines',
        line=dict(color='#38BDF8', width=2),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.15)',
        hovertemplate='Time: %{x:.2f}s<br>Acoustic Energy: %{y:.3f}<extra></extra>'
    ))

    # Mouth Aspect Ratio (MAR)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=mar_series,
        name='Mouth Aspect Ratio (MAR)',
        mode='lines+markers',
        line=dict(color='#FBBF24', width=2),
        marker=dict(size=4, color='#FBBF24'),
        hovertemplate='Time: %{x:.2f}s<br>Mouth Aspect Ratio: %{y:.3f}<extra></extra>'
    ))

    # Lip Articulation Velocity
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=vel_series,
        name='Lip Kinematic Velocity',
        mode='lines',
        line=dict(color='#34D399', width=1.5, dash='dot'),
        hovertemplate='Time: %{x:.2f}s<br>Lip Velocity: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text="<b>Synchronized Audio-Visual Articulation Dynamics Timeline</b>", font=dict(size=14, color='#F1F5F9', family='Inter, sans-serif'), x=0.01),
        xaxis=dict(title=dict(text="Timeline (Seconds)", font=dict(color='#CBD5E1', size=11)), tickfont=dict(color='#94A3B8', size=10), gridcolor='#223354'),
        yaxis=dict(title=dict(text="Normalized Amplitude / Ratio", font=dict(color='#CBD5E1', size=11)), tickfont=dict(color='#94A3B8', size=10), gridcolor='#223354'),
        paper_bgcolor='#172338', plot_bgcolor='#0F172A',
        height=320, margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#CBD5E1', size=10))
    )
    return fig


def plot_cross_correlation_lag_curve(lags_ms, correlation_vals, best_offset_ms):
    """Plot cross-correlation curve across lag search window with optimal offset marker"""
    if not lags_ms or len(lags_ms) == 0:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=lags_ms,
        y=correlation_vals,
        mode='lines',
        line=dict(color='#38BDF8', width=2.2),
        name='Cross-Correlation R(τ)',
        hovertemplate='Temporal Lag: %{x:.1f} ms<br>Correlation: %{y:.3f}<extra></extra>'
    ))

    # Add vertical indicator for peak offset
    fig.add_vline(
        x=best_offset_ms,
        line_width=2,
        line_dash="dash",
        line_color="#10B981" if abs(best_offset_ms) <= 45 else "#EF4444",
        annotation_text=f"Estimated Offset: {best_offset_ms:+.1f} ms",
        annotation_position="top right",
        annotation_font=dict(color="#F1F5F9", size=10)
    )

    # Broadcast standard +/-45ms shaded zone
    fig.add_vrect(
        x0=-45, x1=45,
        fillcolor="rgba(16, 185, 129, 0.12)",
        layer="below", line_width=0,
        annotation_text="Broadcast In-Sync Window (±45ms)",
        annotation_position="bottom left",
        annotation_font=dict(color="#10B981", size=9)
    )

    fig.update_layout(
        title=dict(text="<b>Audio-Visual Cross-Correlation Function R(τ)</b>", font=dict(size=14, color='#F1F5F9', family='Inter, sans-serif'), x=0.01),
        xaxis=dict(title=dict(text="Lag Offset τ (Milliseconds)", font=dict(color='#CBD5E1', size=11)), tickfont=dict(color='#94A3B8', size=10), gridcolor='#223354'),
        yaxis=dict(title=dict(text="Correlation Coefficient", font=dict(color='#CBD5E1', size=11)), tickfont=dict(color='#94A3B8', size=10), gridcolor='#223354'),
        paper_bgcolor='#172338', plot_bgcolor='#0F172A',
        height=280, margin=dict(l=20, r=20, t=50, b=40), showlegend=False
    )
    return fig


def plot_multimodal_comparison_bars(audio_probs, visual_probs, fused_probs):
    """Plot grouped comparison of Audio-Only vs Visual Kinematics vs Multimodal Fused predictions"""
    emotions = list(audio_probs.keys())
    labels = [f"{EMOTION_META.get(e, {}).get('emoji', '')} {e.capitalize()}" for e in emotions]

    a_vals = [audio_probs.get(e, 0.0) * 100 for e in emotions]
    v_vals = [visual_probs.get(e, 0.0) * 100 for e in emotions]
    f_vals = [fused_probs.get(e, 0.0) * 100 for e in emotions]

    fig = go.Figure(data=[
        go.Bar(name='🎙️ Acoustic Model', x=labels, y=a_vals, marker_color='#38BDF8', opacity=0.85),
        go.Bar(name='👁️ Visual Kinematics', x=labels, y=v_vals, marker_color='#FBBF24', opacity=0.85),
        go.Bar(name='✨ Multimodal Fused', x=labels, y=f_vals, marker_color='#34D399', opacity=0.95)
    ])

    fig.update_layout(
        barmode='group',
        title=dict(text="<b>Cross-Modal Affect Comparison (Acoustic vs Visual vs Fused)</b>", font=dict(size=14, color='#F1F5F9', family='Inter, sans-serif'), x=0.01),
        xaxis=dict(tickfont=dict(color='#F1F5F9', size=10), gridcolor='#223354'),
        yaxis=dict(title=dict(text="Probability (%)", font=dict(color='#CBD5E1', size=11)), tickfont=dict(color='#94A3B8', size=10), range=[0, 100], gridcolor='#223354'),
        paper_bgcolor='#172338', plot_bgcolor='#0F172A',
        height=320, margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#CBD5E1', size=10))
    )
    return fig


# -----------------------------------------------------------------------------
# VoiceMind AI Landing Page Controller
# -----------------------------------------------------------------------------
def render_landing_page(predictor):
    """
    Renders the futuristic VoiceMind AI Landing Page matching the reference design:
    - Glowing glass navbar (Waveform Logo, Brand Name, Links, CTA Button)
    - Two-column hero with gradient headline, description, CTAs, and 4 feature indicators
    - Right-column cinematic AI Face & Landmark visual, Hinglish preview, and Emotion Distribution card
    - Capabilities section with 4 feature cards and real navigation triggers
    - Interactive Real-Time Demo Workbench
    - Technical Features Grid (6 architecture tiles)
    - About VoiceMind AI & Ethical AI Guidelines
    - Modern Footer with branding and links
    """
    hero_b64 = get_image_base64("assets/hero_visual_art.jpg")
    if not hero_b64:
        hero_b64 = get_image_base64("assets/landing_hero_reference.jpg")

    # 1. TOP NAVBAR
    st.markdown("""
        <div class="vm-navbar">
            <div class="vm-logo-group">
                <div class="vm-wave-icon">
                    <div class="vm-wave-bar"></div>
                    <div class="vm-wave-bar"></div>
                    <div class="vm-wave-bar"></div>
                    <div class="vm-wave-bar"></div>
                    <div class="vm-wave-bar"></div>
                </div>
                <div>
                    <div class="vm-brand-name">VoiceMind <span>AI</span></div>
                    <span class="vm-brand-tagline">Speak • Feel • Understand</span>
                </div>
            </div>
            <div class="vm-nav-links">
                <a href="#home" class="vm-nav-link active">Home</a>
                <a href="#about" class="vm-nav-link">About</a>
                <a href="#features" class="vm-nav-link">Features</a>
                <a href="#demo" class="vm-nav-link">Demo</a>
                <a href="#contact" class="vm-nav-link">Contact</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 2. HERO SECTION (2 COLUMNS)
    col_hero_left, col_hero_right = st.columns([1.15, 1.0], gap="large")

    with col_hero_left:
        st.markdown("""
            <div id="home">
                <div class="vm-hero-badge">
                    <span>✦</span> AI Powered Emotion & Lip-Sync Analysis
                </div>
                <h1 class="vm-hero-title">
                    Turn Speech & Video<br>
                    <span class="vm-gradient-text">into Meaning</span>
                </h1>
                <p class="vm-hero-desc">
                    Detect emotions from speech, analyze lip movements, convert video to Hinglish text, and get deep insights with our advanced AI-powered platform.
                </p>
            </div>
        """, unsafe_allow_html=True)

        col_btn1, col_btn2, _ = st.columns([1.1, 1.1, 0.8])
        with col_btn1:
            if st.button("🚀 Try Now →", key="landing_hero_try_now", type="primary", use_container_width=True):
                st.session_state["app_navigation_view"] = "🎙️ Speech Analysis Workspace"
                st.rerun()
        with col_btn2:
            st.markdown("""
                <a href="#demo" style="text-decoration: none;">
                    <div style="background: rgba(9, 26, 51, 0.8); border: 1px solid rgba(0, 212, 255, 0.35); border-radius: 24px; padding: 10px 18px; text-align: center; color: #F8FAFC; font-weight: 600; font-size: 0.92rem; transition: all 0.2s ease;">
                        🎬 Watch Demo
                    </div>
                </a>
            """, unsafe_allow_html=True)

        # 4 Feature Indicators Pods under buttons
        st.markdown("""
            <div class="vm-indicators-grid">
                <div class="vm-indicator-pod">
                    <div class="vm-pod-icon" style="color: #00D4FF;">🎙️</div>
                    <div>
                        <div class="vm-pod-title">Emotion Detection</div>
                        <div class="vm-pod-sub">7-class emotion analysis</div>
                    </div>
                </div>
                <div class="vm-indicator-pod">
                    <div class="vm-pod-icon" style="color: #C084FC;">👄</div>
                    <div>
                        <div class="vm-pod-title">Lip-Sync Analysis</div>
                        <div class="vm-pod-sub">Track & decode lip movements</div>
                    </div>
                </div>
                <div class="vm-indicator-pod">
                    <div class="vm-pod-icon" style="color: #38BDF8;">🔤</div>
                    <div>
                        <div class="vm-pod-title">Hinglish Conversion</div>
                        <div class="vm-pod-sub">Video to text (Hinglish)</div>
                    </div>
                </div>
                <div class="vm-indicator-pod">
                    <div class="vm-pod-icon" style="color: #FBBF24;">📊</div>
                    <div>
                        <div class="vm-pod-title">Deep Analytics</div>
                        <div class="vm-pod-sub">Insights & visualizations</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_hero_right:
        if hero_b64:
            st.markdown(f"""
                <div class="vm-visual-wrapper">
                    <img src="{hero_b64}" class="vm-visual-img" alt="VoiceMind AI Visual Intelligence" />
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="vm-visual-wrapper" style="padding: 24px; min-height: 380px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="background: rgba(0, 212, 255, 0.15); border: 1px solid #00D4FF; padding: 6px 12px; border-radius: 8px; font-size: 0.8rem; color: #00D4FF; font-weight: 700;">
                            ✨ Visual Speech Intelligence
                        </div>
                        <div style="font-size: 0.8rem; color: #94A3B8;">468 Landmarks Active</div>
                    </div>
                    <div style="text-align: center; padding: 30px 0;">
                        <div style="font-size: 3.5rem; filter: drop-shadow(0 0 20px #00D4FF);">👄 〰️ 🎙️</div>
                        <div style="font-size: 1.1rem; color: #FFFFFF; font-weight: 700; margin-top: 12px;">Real-Time Audio-Visual Synchronization</div>
                        <div style="font-size: 0.82rem; color: #94A3B8;">Cross-modal alignment • MAR Articulatory kinematics</div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 10px 14px; font-size: 0.8rem; color: #CBD5E1;">
                        <b>Hinglish Transcript:</b> <i>"Aap kaise ho? Kya kar rahe ho?"</i>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # 3. CAPABILITIES SECTION
    st.markdown("""
        <div id="features" style="padding-top: 20px;">
            <div class="vm-section-tag">— OUR CAPABILITIES</div>
            <h2 class="vm-section-title">Everything You Need for Smarter Speech Analysis</h2>
        </div>
    """, unsafe_allow_html=True)

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    with col_c1:
        st.markdown("""
            <div class="vm-capability-card">
                <div>
                    <div class="vm-card-icon-wrap" style="color: #00D4FF; background: rgba(0, 212, 255, 0.12); border-color: rgba(0, 212, 255, 0.3);">🎙️</div>
                    <div class="vm-card-title">Speech Emotion Detection</div>
                    <div class="vm-card-desc">Identify emotions like happy, sad, angry, fear, neutral, surprise and disgust with deep CNN-LSTM networks.</div>
                    <ul class="vm-card-bullets">
                        <li>Audio file upload & live mic</li>
                        <li>128-Mel Spectrogram & MFCCs</li>
                        <li>7-Class confidence breakdown</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Speech →", key="cap_btn_speech", use_container_width=True):
            st.session_state["app_navigation_view"] = "🎙️ Speech Analysis Workspace"
            st.rerun()

    with col_c2:
        st.markdown("""
            <div class="vm-capability-card">
                <div>
                    <div class="vm-card-icon-wrap" style="color: #C084FC; background: rgba(192, 132, 252, 0.12); border-color: rgba(192, 132, 252, 0.3);">👄</div>
                    <div class="vm-card-title">Lip-Sync & Visual Analysis</div>
                    <div class="vm-card-desc">Detect lip movements and analyze video-to-audio synchronization with cross-correlation telemetry.</div>
                    <ul class="vm-card-bullets">
                        <li>Video upload & live webcam</li>
                        <li>Mouth Aspect Ratio (MAR)</li>
                        <li>Sync Quality Index (SQI)</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Video →", key="cap_btn_video", use_container_width=True):
            st.session_state["app_navigation_view"] = "👁️ Lip-Sync & Visual Speech Analysis"
            st.rerun()

    with col_c3:
        st.markdown("""
            <div class="vm-capability-card">
                <div>
                    <div class="vm-card-icon-wrap" style="color: #38BDF8; background: rgba(56, 189, 248, 0.12); border-color: rgba(56, 189, 248, 0.3);">🔤</div>
                    <div class="vm-card-title">Hinglish Video-to-Text</div>
                    <div class="vm-card-desc">Convert spoken words into natural Hinglish text with timestamps and multi-format subtitle exports.</div>
                    <ul class="vm-card-bullets">
                        <li>Visual-only lip reading mode</li>
                        <li>Audio-assisted multimodal fusion</li>
                        <li>SRT, VTT, TXT, CSV, PDF export</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Convert Video →", key="cap_btn_hinglish", use_container_width=True):
            st.session_state["app_navigation_view"] = "👄 AI Lip Reading & Hinglish Converter"
            st.rerun()

    with col_c4:
        st.markdown("""
            <div class="vm-capability-card">
                <div>
                    <div class="vm-card-icon-wrap" style="color: #FBBF24; background: rgba(251, 191, 36, 0.12); border-color: rgba(251, 191, 36, 0.3);">🧠</div>
                    <div class="vm-card-title">Model Insights</div>
                    <div class="vm-card-desc">View accuracy, confusion matrix, per-class F1-scores, and detailed performance benchmarking metrics.</div>
                    <ul class="vm-card-bullets">
                        <li>Interactive Confusion Matrix</li>
                        <li>Precision, Recall & F1-Scores</li>
                        <li>CNN vs LSTM vs Hybrid</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("View Insights →", key="cap_btn_insights", use_container_width=True):
            st.session_state["app_navigation_view"] = "🧠 Model Insights & Evaluation"
            st.rerun()

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # 4. INTERACTIVE LIVE DEMO SECTION
    st.markdown("""
        <div id="demo" style="padding-top: 20px;">
            <div class="vm-section-tag">— LIVE WORKSPACE DEMONSTRATION</div>
            <h2 class="vm-section-title">Experience VoiceMind AI in Real-Time</h2>
            <p style="font-size: 0.95rem; color: #94A3B8; margin-bottom: 20px;">
                Test speech emotion recognition, visual lip-sync analysis, and video-to-Hinglish conversion directly below with real neural inference.
            </p>
        </div>
    """, unsafe_allow_html=True)

    demo_tab1, demo_tab2, demo_tab3, demo_tab4 = st.tabs([
        "🎙️ Speech Emotion Detection",
        "⏱️ AI Lip-Sync Analysis",
        "👄 AI Lip-Reading & Hinglish",
        "🧠 Model Architecture & Evaluation"
    ])

    with demo_tab1:
        col_d1, col_d2 = st.columns([1.1, 1.3])
        with col_d1:
            st.markdown("""
                <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 18px;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">Upload or Record Speech Audio</div>
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 14px;">Supported: WAV, MP3, FLAC, OGG (Max 25MB)</div>
                </div>
            """, unsafe_allow_html=True)
            
            demo_audio = st.file_uploader("Choose Audio File", type=['wav', 'mp3', 'flac', 'ogg'], key="demo_audio_uploader")
            
            use_demo_sample = st.button("🎵 Load Sample Speech Recording", key="demo_sample_audio_btn")
            sample_audio_path = None
            if use_demo_sample:
                # Find any available wav in raw or create one
                for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), 'data')):
                    for f in files:
                        if f.endswith('.wav'):
                            sample_audio_path = os.path.join(root, f)
                            break
                    if sample_audio_path:
                        break

        with col_d2:
            target_audio = demo_audio or sample_audio_path
            if target_audio:
                with st.spinner("Analyzing vocal prosody and computing CNN-LSTM neural probabilities..."):
                    if isinstance(target_audio, str):
                        y, sr = librosa.load(target_audio, sr=22050)
                        fname = os.path.basename(target_audio)
                    else:
                        target_audio.seek(0)
                        y, sr = librosa.load(target_audio, sr=22050)
                        fname = target_audio.name
                        target_audio.seek(0)

                    # Predict
                    temp_p = os.path.join(tempfile.gettempdir(), f"demo_{int(time.time())}.wav")
                    sf.write(temp_p, y, sr)
                    try:
                        if predictor:
                            pred_emo, probs = predictor.predict(temp_p, return_probabilities=True)
                        else:
                            pred_emo, probs = 'neutral', {e: 1.0/7.0 for e in config.EMOTIONS.values()}
                    finally:
                        if os.path.exists(temp_p):
                            os.remove(temp_p)

                    top_conf = probs.get(pred_emo, 0.5) * 100
                    emo_info = EMOTION_META.get(pred_emo.lower(), {'emoji': '🎙️', 'color': '#38BDF8'})

                    st.markdown(f"""
                        <div class="hero-emotion-card" style="border-top: 4px solid {emo_info['color']}; padding: 18px; margin-bottom: 16px;">
                            <div class="hero-badge" style="background: rgba(56, 189, 248, 0.15); color: #38BDF8;">
                                ✨ Primary Affect Detected
                            </div>
                            <div style="display: flex; align-items: center; gap: 16px;">
                                <div style="font-size: 2.8rem;">{emo_info['emoji']}</div>
                                <div>
                                    <h3 style="margin: 0; font-size: 1.6rem; color: #FFFFFF !important;">{pred_emo.upper()}</h3>
                                    <div style="font-size: 0.9rem; color: #CBD5E1;">Model Confidence: <b>{top_conf:.1f}%</b></div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Mini Probabilities Bar
                    for e_name, e_prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:4]:
                        e_m = EMOTION_META.get(e_name.lower(), {'emoji': '🔹', 'color': '#38BDF8'})
                        st.markdown(f"""
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px; font-size: 0.85rem;">
                                <span>{e_m['emoji']} <b>{e_name.capitalize()}</b></span>
                                <span style="font-family: monospace; color: #00D4FF;">{e_prob*100:.1f}%</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px; margin-bottom: 8px; overflow: hidden;">
                                <div style="background: {e_m['color']}; width: {e_prob*100}%; height: 100%; border-radius: 4px;"></div>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("💡 Upload an audio file or click 'Load Sample Speech Recording' to run real-time inference.")

    with demo_tab2:
        st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 20px; text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">AI Lip-Sync Cross-Correlation Engine</div>
                <p style="font-size: 0.85rem; color: #94A3B8; max-width: 600px; margin: 8px auto 16px auto;">
                    Evaluates temporal alignment between vocal acoustics and visual mouth articulation with Pearson correlation across lag search windows.
                </p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Full Lip-Sync Workspace →", key="demo_goto_sync_btn", type="primary"):
            st.session_state["app_navigation_view"] = "👁️ Lip-Sync & Visual Speech Analysis"
            st.rerun()

    with demo_tab3:
        st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 20px; text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF;">Hinglish Video-to-Text & Subtitle Generator</div>
                <p style="font-size: 0.85rem; color: #94A3B8; max-width: 600px; margin: 8px auto 16px auto;">
                    Translates spoken words and visual articulatory lip kinematics into conversational Romanized Hinglish with schwa-deletion heuristics.
                </p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Hinglish Converter Workspace →", key="demo_goto_hinglish_btn", type="primary"):
            st.session_state["app_navigation_view"] = "👄 AI Lip Reading & Hinglish Converter"
            st.rerun()

    with demo_tab4:
        st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 20px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 8px;">Neural Architecture & Benchmarking</div>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 16px;">
                    Empirical performance metrics across 2D CNN, Bidirectional LSTM, and Hybrid CNN-LSTM backbones.
                </p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Model Insights & Confusion Matrix →", key="demo_goto_insights_btn", type="primary"):
            st.session_state["app_navigation_view"] = "🧠 Model Insights & Evaluation"
            st.rerun()

    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

    # 5. TECHNICAL FEATURES GRID
    st.markdown("""
        <div id="features" style="padding-top: 20px;">
            <div class="vm-section-tag">— SYSTEM ARCHITECTURE</div>
            <h2 class="vm-section-title">Production-Grade AI Engineering Pipeline</h2>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 40px;">
            <div class="vm-tech-tile">
                <div style="font-size: 1.4rem; margin-bottom: 8px;">🔬</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF;">128-Mel Filterbank & MFCC</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Extracts perceptual frequency energy distributions and 40 cepstral coefficients at 22.05 kHz.</div>
            </div>
            <div class="vm-tech-tile">
                <div style="font-size: 1.4rem; margin-bottom: 8px;">🧠</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF;">CNN-LSTM Neural Network</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Spatiotemporal 2D convolutions coupled with Bidirectional LSTM sequential memory context.</div>
            </div>
            <div class="vm-tech-tile">
                <div style="font-size: 1.4rem; margin-bottom: 8px;">👁️</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF;">Face & Lip Landmarking</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Extracts Mouth Aspect Ratio (MAR) and articulatory kinematic velocity vectors frame-by-frame.</div>
            </div>
            <div class="vm-tech-tile">
                <div style="font-size: 1.4rem; margin-bottom: 8px;">⏱️</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF;">Pearson Cross-Correlation</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Calculates optimal temporal lag offsets (ms) and broadcast Sync Quality Index (SQI 0-100%).</div>
            </div>
            <div class="vm-tech-tile">
                <div style="font-size: 1.4rem; margin-bottom: 8px;">🇮🇳</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF;">Hinglish Schwa Deletion</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Linguistic phonology rules transliterating Devanagari Hindi into natural English-alphabet Hinglish.</div>
            </div>
            <div class="vm-tech-tile">
                <div style="font-size: 1.4rem; margin-bottom: 8px;">📄</div>
                <div style="font-size: 1.0rem; font-weight: 700; color: #FFFFFF;">Multi-Format Subtitle & PDF</div>
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">Exports audit reports in PDF, SRT, VTT, TXT, and CSV formats with clause-level timecodes.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 6. ABOUT SECTION
    st.markdown("""
        <div id="about" style="background: rgba(9, 26, 51, 0.7); border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 18px; padding: 28px 32px; margin-bottom: 40px;">
            <div class="vm-section-tag">— ABOUT VOICEMIND AI</div>
            <h2 style="font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin: 0 0 14px 0;">Empowering Conversational & Affective Speech Intelligence</h2>
            <p style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.7; margin-bottom: 16px;">
                VoiceMind AI explores how artificial intelligence can analyze speech, estimate vocal emotion, examine lip movements, and transform supported video speech into readable text. By combining audio digital signal processing, computer vision, and deep learning, the platform provides an interactive environment for speech analysis, synchronization telemetry, and model evaluation.
            </p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; font-size: 0.85rem; color: #94A3B8;">
                <div style="background: #061326; padding: 14px; border-radius: 8px; border-left: 3px solid #00D4FF;">
                    <b style="color: #F8FAFC;">🛡️ Ethical & Responsible AI:</b> Designed exclusively for human-computer interaction, academic research, and affective analytics. Not a polygraph or lie-detector instrument.
                </div>
                <div style="background: #061326; padding: 14px; border-radius: 8px; border-left: 3px solid #8B5CF6;">
                    <b style="color: #F8FAFC;">🔒 In-Memory Privacy:</b> All audio and video streams are processed ephemerally in memory or local temporary caches with strict data minimization.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 7. CONTACT & FOOTER
    st.markdown("""
        <div id="contact" style="border-top: 1px solid rgba(0, 212, 255, 0.2); padding: 36px 0 16px 0; margin-top: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px; margin-bottom: 24px;">
                <div class="vm-logo-group">
                    <div class="vm-wave-icon">
                        <div class="vm-wave-bar"></div>
                        <div class="vm-wave-bar"></div>
                        <div class="vm-wave-bar"></div>
                        <div class="vm-wave-bar"></div>
                        <div class="vm-wave-bar"></div>
                    </div>
                    <div>
                        <div class="vm-brand-name">VoiceMind <span>AI</span></div>
                        <span class="vm-brand-tagline">Speak • Feel • Understand</span>
                    </div>
                </div>
                <div style="display: flex; gap: 20px; font-size: 0.85rem;">
                    <a href="https://github.com/Harshshukla49/Speech-recognition" target="_blank" style="color: #00D4FF; text-decoration: none; font-weight: 600;">
                        🔗 GitHub Repository
                    </a>
                    <span style="color: #64748B;">•</span>
                    <a href="#home" style="color: #CBD5E1; text-decoration: none;">Back to Top ↑</a>
                </div>
            </div>
            <div style="text-align: center; font-size: 0.78rem; color: #64748B;">
                © 2026 VoiceMind AI Platform • Deep Learning Speech Emotion Recognition & Visual Speech Intelligence • All Rights Reserved.
            </div>
        </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Application Controller
# -----------------------------------------------------------------------------
def main():
    # -------------------------------------------------------------------------
    # Sidebar: Brand, Navigation & Settings
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-brand">
                <div class="sidebar-brand-icon">🎙️</div>
                <div>
                    <div class="sidebar-brand-title">VoiceMind AI</div>
                    <div class="sidebar-brand-desc">Speak • Feel • Understand</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        selected_view = st.radio(
            "Navigation Menu",
            [
                "✨ VoiceMind AI Home",
                "🎙️ Speech Analysis Workspace",
                "👁️ Lip-Sync & Visual Speech Analysis",
                "👄 AI Lip Reading & Hinglish Converter",
                "📊 Dashboard Overview",
                "📜 Prediction History & Database",
                "🧠 Model Insights & Evaluation",
                "🧪 Model Benchmarks & Comparison",
                "⚙️ Settings & Responsible AI"
            ],
            key="app_navigation_view"
        )

        # Collapsible About Section
        with st.expander("📋 About this System", expanded=False):
            st.markdown("""
                <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.4; margin-bottom: 10px;">
                    This platform uses a <b>CNN-LSTM Deep Neural Network</b> to classify human emotional affect from speech signals in real-time.
                </div>
                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #38BDF8; margin: 10px 0 6px 0;">
                    Supported Emotions (7 Classes)
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="emotions-grid">
                    <div class="emotion-chip"><span class="chip-emoji">😊</span><span class="chip-label">Happy</span></div>
                    <div class="emotion-chip"><span class="chip-emoji">😢</span><span class="chip-label">Sad</span></div>
                    <div class="emotion-chip"><span class="chip-emoji">😠</span><span class="chip-label">Angry</span></div>
                    <div class="emotion-chip"><span class="chip-emoji">😨</span><span class="chip-label">Fear</span></div>
                    <div class="emotion-chip"><span class="chip-emoji">😐</span><span class="chip-label">Neutral</span></div>
                    <div class="emotion-chip"><span class="chip-emoji">😲</span><span class="chip-label">Surprise</span></div>
                    <div class="emotion-chip"><span class="chip-emoji">🤢</span><span class="chip-label">Disgust</span></div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #38BDF8; margin: 12px 0 6px 0;">
                    Workflow Guide
                </div>
                <div class="guide-step"><span class="step-num">1</span><span>Upload WAV/MP3 or capture with microphone.</span></div>
                <div class="guide-step"><span class="step-num">2</span><span>128-Mel Spectrogram & acoustic features extracted.</span></div>
                <div class="guide-step"><span class="step-num">3</span><span>CNN-LSTM classifies vocal sentiment & timeline.</span></div>
            """, unsafe_allow_html=True)

        # Quick Visualization Settings (Persisted in st.session_state)
        if "show_waveform" not in st.session_state:
            st.session_state["show_waveform"] = True
        if "show_spectrogram" not in st.session_state:
            st.session_state["show_spectrogram"] = True
        if "show_mfcc" not in st.session_state:
            st.session_state["show_mfcc"] = True
        if "show_probabilities" not in st.session_state:
            st.session_state["show_probabilities"] = True
        if "conf_threshold" not in st.session_state:
            st.session_state["conf_threshold"] = 40

        with st.expander("⚙️ Quick Display Toggles", expanded=False):
            st.session_state["show_waveform"] = st.toggle("Show Waveform", value=st.session_state["show_waveform"])
            st.session_state["show_spectrogram"] = st.toggle("Show Spectrogram", value=st.session_state["show_spectrogram"])
            st.session_state["show_mfcc"] = st.toggle("Show MFCC Heatmap", value=st.session_state["show_mfcc"])
            st.session_state["show_probabilities"] = st.toggle("Show Probabilities", value=st.session_state["show_probabilities"])

    # Load Model Predictor
    predictor = load_predictor()

    # -------------------------------------------------------------------------
    # VIEW 0: ✨ VOICEMIND AI LANDING PAGE
    # -------------------------------------------------------------------------
    if selected_view == "✨ VoiceMind AI Home":
        render_landing_page(predictor)

    # -------------------------------------------------------------------------
    # VIEW 1: 📊 DASHBOARD OVERVIEW
    # -------------------------------------------------------------------------
    elif selected_view == "📊 Dashboard Overview":
        st.markdown("""
            <div class="dashboard-header">
                <div class="header-badge">Executive Analytics • System Telemetry</div>
                <h1 class="header-title">📊 Audio Intelligence Overview</h1>
                <p class="header-subtitle">Real-time statistics, empirical classification telemetry, and system availability overview.</p>
            </div>
        """, unsafe_allow_html=True)

        stats = get_summary_stats()

        # KPI Metrics Cards
        st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Audio Analyses</div>
                    <div class="kpi-value">{stats['total_analyses']}</div>
                    <div class="kpi-sub">Stored in SQLite History</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Mean Prediction Confidence</div>
                    <div class="kpi-value">{stats['avg_confidence']*100:.1f}%</div>
                    <div class="kpi-sub">Across processed recordings</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Mean Inference Latency</div>
                    <div class="kpi-value">{stats['avg_processing_time']*1000:.0f} ms</div>
                    <div class="kpi-sub">Feature Extraction + Model Feedforward</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Active Neural Model</div>
                    <div class="kpi-value" style="font-size: 1.25rem;">CNN-LSTM v2.0</div>
                    <div class="kpi-sub" style="color: #34D399;">● Model Ready in Memory</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col_chart, col_recent = st.columns([1, 1.2])

        with col_chart:
            st.markdown("""
                <div class="content-card">
                    <div class="content-card-title">🎭 Emotion Classification Distribution</div>
                </div>
            """, unsafe_allow_html=True)

            if stats['total_analyses'] > 0 and stats['emotion_distribution']:
                labels = [k.capitalize() for k in stats['emotion_distribution'].keys()]
                values = list(stats['emotion_distribution'].values())
                colors_donut = [EMOTION_META.get(k.lower(), {}).get('color', '#38BDF8') for k in stats['emotion_distribution'].keys()]

                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels, values=values, hole=0.55,
                    marker=dict(colors=colors_donut, line=dict(color='#0F172A', width=2)),
                    textinfo='percent+label',
                    textfont=dict(color='#F1F5F9', size=11)
                )])
                fig_donut.update_layout(
                    paper_bgcolor='#172338', plot_bgcolor='#0F172A',
                    font=dict(color='#CBD5E1'),
                    height=320, margin=dict(l=10, r=10, t=20, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("ℹ️ No analysis history recorded yet. Perform an audio analysis in the Workspace to populate live telemetry charts.")

        with col_recent:
            st.markdown("""
                <div class="content-card">
                    <div class="content-card-title">🕒 Recent Analysis Activity</div>
                </div>
            """, unsafe_allow_html=True)

            if stats['recent_records']:
                recent_df = pd.DataFrame([
                    {
                        'ID': r['id'][:8],
                        'Time': r['timestamp'].split(' ')[1] if ' ' in r['timestamp'] else r['timestamp'],
                        'Filename': r['filename'][:16],
                        'Emotion': f"{EMOTION_META.get(r['predicted_emotion'], {}).get('emoji', '')} {r['predicted_emotion'].capitalize()}",
                        'Confidence': f"{r['confidence']*100:.1f}%",
                        'Status': '⚠️ Low Conf' if r['is_low_confidence'] else '✅ Verified'
                    }
                    for r in stats['recent_records']
                ])
                st.dataframe(recent_df, use_container_width=True, hide_index=True)
            else:
                st.markdown("""
                    <div style="background: #0F172A; border: 1px dashed #223354; border-radius: 10px; padding: 24px; text-align: center;">
                        <p style="margin: 0; color: #94A3B8;">No recent activity found. Head over to <b>Speech Analysis Workspace</b> to analyze your first audio sample!</p>
                    </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # VIEW 2: 🎙️ SPEECH ANALYSIS WORKSPACE
    # -------------------------------------------------------------------------
    elif selected_view == "🎙️ Speech Analysis Workspace":
        st.markdown("""
            <div class="dashboard-header">
                <div class="header-badge">Audio Signal Workspace • Deep Neural Inference</div>
                <h1 class="header-title">🎙️ Speech Emotion Analysis Workspace</h1>
                <p class="header-subtitle">Upload or record human vocal audio. Extract multi-dimensional spectral features and classify 7 discrete emotion taxonomies.</p>
            </div>
        """, unsafe_allow_html=True)

        if predictor is None:
            st.error("❌ Model weights not found in `models/best_model.h5`. Please run `python run_pipeline.py` in your terminal to train and compile the model.")

        input_mode = st.radio(
            "Select Audio Input Method:",
            ["📁 Upload Audio File (WAV / MP3 / FLAC / OGG)", "🎤 Record from Microphone (Browser / Hardware)"],
            horizontal=True
        )

        audio_bytes = None
        audio_name = "sample_audio.wav"
        source_type = "upload"

        if "Upload" in input_mode:
            uploaded_file = st.file_uploader(
                "Select Audio File",
                type=['wav', 'mp3', 'flac', 'ogg'],
                help="Upload a clean speech audio recording (3-10 seconds recommended)"
            )
            if uploaded_file is not None:
                audio_bytes = uploaded_file.getvalue()
                audio_name = uploaded_file.name
                source_type = "upload"
        else:
            source_type = "microphone"
            rec_method = st.radio(
                "Microphone Source:",
                ["🎙️ Native Browser Microphone", "💻 Direct Desktop Soundcard Capture (PyAudio)"],
                horizontal=True
            )

            if "Browser" in rec_method:
                audio_input = st.audio_input("Record Speech", key="workspace_browser_mic")
                if audio_input is not None:
                    audio_bytes = audio_input.getvalue()
                    audio_name = f"mic_recording_{datetime.now().strftime('%H%M%S')}.wav"
                    st.session_state["workspace_last_rec"] = audio_bytes
                elif "workspace_last_rec" in st.session_state:
                    audio_bytes = st.session_state["workspace_last_rec"]
            else:
                col_d, col_b = st.columns([1, 2])
                with col_d:
                    dur_sec = st.slider("Duration (seconds)", min_value=2, max_value=8, value=3)
                with col_b:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("🔴 Start Hardware Recording", key="btn_workspace_hw_rec"):
                        with st.spinner(f"🎙️ Recording from default hardware microphone for {dur_sec}s..."):
                            try:
                                audio_bytes = record_from_hardware_mic(duration=dur_sec, sample_rate=config.SAMPLE_RATE)
                                audio_name = f"hardware_rec_{datetime.now().strftime('%H%M%S')}.wav"
                                st.session_state["workspace_last_rec"] = audio_bytes
                                st.success(f"✅ Successfully captured {dur_sec}s of audio!")
                            except Exception as e:
                                st.error(f"❌ Hardware recording error: {str(e)}")

                if "workspace_last_rec" in st.session_state and audio_bytes is None:
                    audio_bytes = st.session_state["workspace_last_rec"]

        # If audio is available, render analysis pipeline
        if audio_bytes is not None and len(audio_bytes) > 0:
            st.markdown("---")
            st.markdown("### 🎧 Audio Playback & Signal Inspection")
            st.audio(audio_bytes, format="audio/wav")

            if st.button("⚡ Run Deep Learning Emotion Analysis", type="primary", key="btn_run_full_analysis"):
                if predictor is None:
                    st.error("❌ Model not available.")
                else:
                    with st.spinner("Decoding audio, computing 128-Mel band spectrogram, and running inference..."):
                        start_time = time.time()
                        temp_file_path = f"temp_{uuid.uuid4().hex[:8]}.wav"
                        try:
                            with open(temp_file_path, "wb") as f:
                                f.write(audio_bytes)

                            # Load audio array
                            audio_array, sr = librosa.load(temp_file_path, sr=config.SAMPLE_RATE)
                            total_duration = float(len(audio_array) / sr)

                            # Validate non-empty
                            if len(audio_array) == 0 or np.max(np.abs(audio_array)) < 0.0001:
                                st.warning("⚠️ Warning: Audio contains silence or very low amplitude.")

                            # Predict
                            emotion, probabilities = predictor.predict(temp_file_path, return_probabilities=True)
                            conf = float(probabilities[emotion])
                            proc_time = time.time() - start_time

                            # Acoustic features
                            acoustic_metrics = compute_acoustic_features(audio_array, sr)

                            # Segment timeline if audio > 2.5s
                            segment_timeline = []
                            if total_duration >= 2.5:
                                segment_timeline = segment_and_predict(temp_file_path, predictor, segment_duration=3.0, hop_duration=1.5)

                            # Low confidence check
                            conf_threshold = st.session_state.get("conf_threshold", 40)
                            is_low_conf = (conf * 100) < conf_threshold

                            # Save to SQLite Database
                            analysis_id = str(uuid.uuid4())
                            save_analysis(
                                analysis_id=analysis_id,
                                filename=audio_name,
                                source_type=source_type,
                                duration=total_duration,
                                predicted_emotion=emotion,
                                confidence=conf,
                                is_low_confidence=is_low_conf,
                                probabilities=probabilities,
                                acoustic_metrics=acoustic_metrics,
                                segment_results=segment_timeline,
                                model_version="CNN-LSTM Hybrid v2.0",
                                processing_time=proc_time
                            )

                            st.success(f"✅ Analysis completed in {proc_time*1000:.0f} ms!")

                            # Render Hero Card
                            render_emotion_hero(emotion, conf, is_low_conf, conf_threshold)

                            # Probability Distribution Chart
                            if st.session_state.get("show_probabilities", True):
                                st.plotly_chart(plot_emotion_bars(probabilities), use_container_width=True)

                            # Acoustic Metrics Cards
                            st.markdown("""
                                <div class="content-card">
                                    <div class="content-card-title">🔬 Extracted Acoustic Signal Descriptors</div>
                                </div>
                            """, unsafe_allow_html=True)

                            st.markdown(f"""
                                <div class="metric-grid">
                                    <div class="metric-pill">
                                        <div class="metric-value">{acoustic_metrics['duration_sec']}s</div>
                                        <div class="metric-label">Duration</div>
                                    </div>
                                    <div class="metric-pill">
                                        <div class="metric-value">{acoustic_metrics['sample_rate']} Hz</div>
                                        <div class="metric-label">Sample Rate</div>
                                    </div>
                                    <div class="metric-pill">
                                        <div class="metric-value">{acoustic_metrics['mean_rms']:.3f}</div>
                                        <div class="metric-label">RMS Energy</div>
                                    </div>
                                    <div class="metric-pill">
                                        <div class="metric-value">{acoustic_metrics['estimated_mean_pitch_hz']} Hz</div>
                                        <div class="metric-label">Pitch (F0)</div>
                                    </div>
                                    <div class="metric-pill">
                                        <div class="metric-value">{acoustic_metrics['silence_percentage']}%</div>
                                        <div class="metric-label">Silence %</div>
                                    </div>
                                    <div class="metric-pill">
                                        <div class="metric-value">{acoustic_metrics['mean_zcr']:.3f}</div>
                                        <div class="metric-label">Zero Crossing</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            # Waveform & Spectrogram Plots
                            if st.session_state.get("show_waveform", True) and st.session_state.get("show_spectrogram", True):
                                col_w, col_s = st.columns(2)
                                with col_w:
                                    st.pyplot(plot_waveform(audio_array, sr))
                                with col_s:
                                    st.pyplot(plot_spectrogram(audio_array, sr))
                            elif st.session_state.get("show_waveform", True):
                                st.pyplot(plot_waveform(audio_array, sr))
                            elif st.session_state.get("show_spectrogram", True):
                                st.pyplot(plot_spectrogram(audio_array, sr))

                            # MFCC Heatmap Plot
                            if st.session_state.get("show_mfcc", True):
                                st.pyplot(plot_mfcc_heatmap(audio_array, sr))

                            # Segment-wise Timeline
                            if segment_timeline and len(segment_timeline) > 1:
                                timeline_fig = plot_emotion_timeline(segment_timeline)
                                if timeline_fig:
                                    st.plotly_chart(timeline_fig, use_container_width=True)

                            # Export Options (PDF & JSON)
                            st.markdown("---")
                            st.markdown("### 📄 Export & Reporting")
                            col_pdf, col_json = st.columns(2)

                            analysis_record = {
                                'id': analysis_id,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'filename': audio_name,
                                'source_type': source_type,
                                'duration': total_duration,
                                'predicted_emotion': emotion,
                                'confidence': conf,
                                'is_low_confidence': is_low_conf,
                                'probabilities': probabilities,
                                'acoustic_metrics': acoustic_metrics,
                                'segment_results': segment_timeline,
                                'model_version': 'CNN-LSTM Hybrid v2.0',
                                'processing_time': proc_time
                            }

                            with col_pdf:
                                pdf_buf = generate_pdf_report(
                                    analysis_record=analysis_record,
                                    audio_time_series=audio_array,
                                    sample_rate=sr
                                )
                                st.download_button(
                                    label="📥 Download Executive PDF Report",
                                    data=pdf_buf.getvalue(),
                                    file_name=f"Emotion_Report_{emotion}_{analysis_id[:8]}.pdf",
                                    mime="application/pdf"
                                )

                            with col_json:
                                st.download_button(
                                    label="📄 Download Analysis JSON Data",
                                    data=json.dumps(analysis_record, indent=2),
                                    file_name=f"Emotion_Data_{analysis_id[:8]}.json",
                                    mime="application/json"
                                )

                        except Exception as analysis_err:
                            st.error(f"❌ Error during audio analysis: {str(analysis_err)}")
                        finally:
                            if os.path.exists(temp_file_path):
                                try:
                                    os.remove(temp_file_path)
                                except Exception:
                                    pass

    # -------------------------------------------------------------------------
    # VIEW: 👁️ LIP-SYNC & VISUAL SPEECH ANALYSIS
    # -------------------------------------------------------------------------
    elif selected_view == "👁️ Lip-Sync & Visual Speech Analysis":
        st.markdown("""
            <div class="dashboard-header">
                <div class="header-badge">Multimodal AI • Visual Speech Kinematics</div>
                <h1 class="header-title">👁️ Lip-Sync Detection & Visual Speech Analysis</h1>
                <p class="header-subtitle">Cross-modal audio-visual synchronization, mouth articulation tracking (MAR), silent visual speech recognition diagnostics, and multimodal emotion fusion.</p>
            </div>
        """, unsafe_allow_html=True)

        video_tab1, video_tab2 = st.tabs(["📁 Upload Video File", "📹 Webcam / Live Capture"])
        video_source_file = None
        video_filename = "recorded_video.mp4"
        video_source_type = "Webcam"

        with video_tab1:
            uploaded_video = st.file_uploader(
                "Upload Video File (MP4, MOV, AVI, WEBM, MKV)",
                type=["mp4", "mov", "avi", "webm", "mkv", "m4v"],
                help="Upload a video containing a visible speaker's face and speech audio."
            )
            if uploaded_video is not None:
                video_source_file = uploaded_video
                video_filename = uploaded_video.name
                video_source_type = "Video Upload"

        with video_tab2:
            st.info("📹 You can record or upload a video clip with your camera active. For real-time browser recording, select your camera capture or upload a recorded clip.")
            camera_video = st.file_uploader(
                "Upload Camera Clip (or capture using device camera)",
                type=["mp4", "webm", "mov"],
                key="camera_video_uploader"
            )
            if camera_video is not None:
                video_source_file = camera_video
                video_filename = f"webcam_capture_{int(time.time())}.mp4"
                video_source_type = "Webcam Capture"

        # If video selected, proceed to analysis pipeline
        if video_source_file is not None:
            # Video player and controls
            st.markdown("### 🎬 Media Ingestion & Video Stream Preview")
            col_vid_player, col_vid_info = st.columns([1.3, 1.0])

            video_proc = VideoProcessor(target_sample_rate=22050, max_fps=30)
            temp_video_path = video_proc.save_temp_video(video_source_file)

            try:
                metadata = video_proc.get_metadata(temp_video_path)

                with col_vid_player:
                    st.video(temp_video_path)

                with col_vid_info:
                    st.markdown("""
                        <div class="content-card">
                            <div class="content-card-title">📼 Stream Integrity & Container Info</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="metric-grid">
                            <div class="metric-pill">
                                <div class="metric-value">{metadata['duration_sec']:.1f}s</div>
                                <div class="metric-label">Duration</div>
                            </div>
                            <div class="metric-pill">
                                <div class="metric-value">{metadata['fps']:.1f}</div>
                                <div class="metric-label">FPS</div>
                            </div>
                            <div class="metric-pill">
                                <div class="metric-value">{metadata['width']}x{metadata['height']}</div>
                                <div class="metric-label">Resolution</div>
                            </div>
                            <div class="metric-pill">
                                <div class="metric-value">{metadata['file_size_mb']}MB</div>
                                <div class="metric-label">File Size</div>
                            </div>
                            <div class="metric-pill">
                                <div class="metric-value">{metadata['video_codec']}</div>
                                <div class="metric-label">Video Codec</div>
                            </div>
                            <div class="metric-pill">
                                <div class="metric-value">{metadata['audio_codec']}</div>
                                <div class="metric-label">Audio Codec</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    if not metadata['is_valid']:
                        st.warning(f"⚠️ Notice: {metadata['validation_message']}")

                if not metadata['has_audio']:
                    st.warning("⚠️ Warning: No audio stream detected in video container. Analysis will evaluate silent visual articulation dynamics.")

                # Analysis trigger button
                analyze_video_btn = st.button("🚀 Analyze Lip-Sync & Visual Speech", type="primary", use_container_width=True)

                if analyze_video_btn:
                    with st.spinner("🔍 Extracting video frames, tracking mouth landmarks, demuxing audio, and computing cross-modal sync..."):
                        t_start = time.time()

                        # Step 1: Extract Audio & Frames
                        audio_array, sr = video_proc.extract_audio(temp_video_path)
                        frames_list, frame_summary = video_proc.extract_frames(
                            temp_video_path,
                            target_fps=25.0,
                            max_frames=1200,
                            resize_dims=(480, 360)
                        )

                        # Step 2: Lip Movement Detection (MAR & Velocity)
                        detector = LipSyncDetector(fps=frame_summary.get('sampling_fps', 25.0))
                        lip_points = detector.track_lip_movement(frames_list)

                        # Step 3: Audio-Visual Synchronization Analysis
                        sync_results = detector.analyze_synchronization(lip_points, audio_array, sr)

                        # Step 4: Visual Speech Recognition Adapter (VSR)
                        vsr_adapter = VisualSpeechRecognizerAdapter()
                        vsr_results = vsr_adapter.transcribe_visual_speech(frames_list, lip_points)

                        # Step 5: Audio Emotion Prediction & Multimodal Emotion Fusion
                        if predictor and len(audio_array) > 0 and np.max(np.abs(audio_array)) > 0.001:
                            # Save temp wav for audio predictor
                            temp_wav_path = os.path.join(tempfile.gettempdir(), f"temp_sync_audio_{int(time.time())}.wav")
                            sf.write(temp_wav_path, audio_array, sr)
                            try:
                                audio_emo, audio_probs = predictor.predict(temp_wav_path, return_probabilities=True)
                            finally:
                                if os.path.exists(temp_wav_path):
                                    os.remove(temp_wav_path)
                        else:
                            audio_emo = 'neutral'
                            audio_probs = {e: round(1.0/7.0, 4) for e in config.EMOTIONS.values()}

                        fusion_engine = MultimodalFusionEngine()
                        fusion_results = fusion_engine.fuse_predictions(audio_probs, lip_points)

                        t_proc = time.time() - t_start

                        # Acoustic features
                        acoustic_metrics = compute_acoustic_features(audio_array, sr) if len(audio_array) > 0 else {}

                        # Step 6: Save Analysis to Database
                        analysis_id = str(uuid.uuid4())
                        save_analysis(
                            analysis_id=analysis_id,
                            filename=video_filename,
                            source_type=video_source_type,
                            duration=metadata['duration_sec'],
                            predicted_emotion=fusion_results['multimodal_emotion'],
                            confidence=fusion_results['multimodal_confidence'],
                            is_low_confidence=(fusion_results['multimodal_confidence'] < 0.40),
                            probabilities=fusion_results['multimodal_probabilities'],
                            acoustic_metrics=acoustic_metrics,
                            segment_results=None,
                            model_version="Multimodal Audio-Visual Late Fusion v2.0",
                            processing_time=t_proc,
                            media_type="video",
                            sync_offset_ms=sync_results['time_offset_ms'],
                            sync_quality_score=sync_results['sync_quality_index'],
                            visual_metrics=sync_results
                        )

                        st.success(f"✅ Multimodal Analysis Completed in {t_proc:.2f}s!")

                        # -----------------------------------------------------
                        # SECTION A: Synchronization KPI Cards & Badge
                        # -----------------------------------------------------
                        st.markdown("### ⏱️ Audio-Video Synchronization Telemetry")
                        
                        sqi = sync_results['sync_quality_index']
                        offset = sync_results['time_offset_ms']
                        badge_color = sync_results['sync_badge_color']
                        status_str = sync_results['sync_status']

                        st.markdown(f"""
                            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-left: 6px solid {badge_color}; border-radius: var(--radius-md); padding: 16px 20px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                                <div>
                                    <div style="font-size: 0.8rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">A/V Synchronization Diagnosis</div>
                                    <div style="font-size: 1.35rem; font-weight: 800; color: {badge_color}; margin-top: 2px;">{status_str}</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 0.8rem; color: var(--text-muted);">Sync Quality Index (SQI)</div>
                                    <div style="font-size: 1.6rem; font-weight: 800; color: {badge_color}; font-family: 'JetBrains Mono', monospace;">{sqi:.1f}%</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                            <div class="kpi-grid">
                                <div class="kpi-card">
                                    <div class="kpi-label">Estimated Temporal Offset (Δt)</div>
                                    <div class="kpi-value">{offset:+.1f} ms</div>
                                    <div class="kpi-sub">Cross-Correlation Peak Lag</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-label">Cross-Correlation Peak (r)</div>
                                    <div class="kpi-value">{sync_results['correlation_coefficient']:.3f}</div>
                                    <div class="kpi-sub">Normalized Pearson Peak</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-label">Visual Speech Activity</div>
                                    <div class="kpi-value">{sync_results['speaking_activity_pct']:.1f}%</div>
                                    <div class="kpi-sub">Frames with active articulation</div>
                                </div>
                                <div class="kpi-card">
                                    <div class="kpi-label">Face Landmark Tracking</div>
                                    <div class="kpi-value">{sync_results['tracking_consistency_pct']:.1f}%</div>
                                    <div class="kpi-sub">Valid landmark continuity</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        # -----------------------------------------------------
                        # SECTION B: Synchronized Dual Timeline & Lag Curve
                        # -----------------------------------------------------
                        st.markdown("### 📊 Synchronized Articulation Dynamics & Cross-Correlation")
                        col_tl, col_lag = st.columns([1.6, 1.2])

                        with col_tl:
                            timeline_fig = plot_dual_sync_timeline(
                                sync_results.get('timestamps', []),
                                sync_results.get('audio_energy_envelope', []),
                                sync_results.get('lip_mar_trajectory', []),
                                sync_results.get('lip_velocity_trajectory', [])
                            )
                            if timeline_fig:
                                st.plotly_chart(timeline_fig, use_container_width=True)

                        with col_lag:
                            lag_fig = plot_cross_correlation_lag_curve(
                                sync_results.get('lags_ms', []),
                                sync_results.get('cross_correlation', []),
                                sync_results.get('time_offset_ms', 0.0)
                            )
                            if lag_fig:
                                st.plotly_chart(lag_fig, use_container_width=True)

                        # -----------------------------------------------------
                        # SECTION C: Visual Inspection Frame Overlay Gallery
                        # -----------------------------------------------------
                        st.markdown("### 🔍 Mouth Landmark & Articulation Frame Inspector")
                        if frames_list and lip_points:
                            sample_indices = np.linspace(0, len(frames_list) - 1, min(4, len(frames_list)), dtype=int)
                            cols_f = st.columns(len(sample_indices))
                            for c_idx, s_idx in enumerate(sample_indices):
                                annotated_frame = detector.annotate_frame(
                                    frames_list[s_idx]['image'],
                                    lip_points[s_idx],
                                    sync_status=status_str
                                )
                                with cols_f[c_idx]:
                                    st.image(
                                        annotated_frame,
                                        caption=f"t={lip_points[s_idx].timestamp:.2f}s (MAR: {lip_points[s_idx].mouth_aspect_ratio:.2f})",
                                        use_container_width=True
                                    )

                        # -----------------------------------------------------
                        # SECTION D: Multimodal Emotion Fusion Consensus
                        # -----------------------------------------------------
                        st.markdown("### 🎭 Multimodal Affect & Emotion Consensus")
                        col_fuse1, col_fuse2 = st.columns([1.1, 1.5])

                        with col_fuse1:
                            top_fused = fusion_results['multimodal_emotion'].upper()
                            top_fused_conf = fusion_results['multimodal_confidence'] * 100
                            fused_meta = EMOTION_META.get(top_fused.lower(), {'emoji': '🎭', 'color': '#38BDF8'})

                            st.markdown(f"""
                                <div class="hero-emotion-card" style="border-top: 4px solid {fused_meta['color']}; padding: 18px; margin: 0 0 14px 0;">
                                    <div class="hero-badge" style="background: rgba(56, 189, 248, 0.15); color: #38BDF8;">
                                        ✨ Multimodal Fused Consensus
                                    </div>
                                    <div style="display: flex; align-items: center; gap: 16px;">
                                        <div style="font-size: 2.8rem;">{fused_meta['emoji']}</div>
                                        <div>
                                            <h3 style="margin: 0; font-size: 1.6rem; color: #FFFFFF !important;">{top_fused}</h3>
                                            <div style="font-size: 0.9rem; color: #CBD5E1;">Confidence: <b>{top_fused_conf:.1f}%</b></div>
                                        </div>
                                    </div>
                                    <div style="margin-top: 10px; font-size: 0.78rem; color: #94A3B8; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 6px;">
                                        {fusion_results['methodology_note']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            col_m1, col_m2 = st.columns(2)
                            with col_m1:
                                a_top_emo = fusion_results['audio_only_emotion']
                                a_meta = EMOTION_META.get(a_top_emo.lower(), {'emoji': '🎙️'})
                                st.markdown(f"""
                                    <div class="metric-pill">
                                        <div class="metric-label">🎙️ Acoustic Only</div>
                                        <div class="metric-value" style="font-size: 1.0rem;">{a_meta['emoji']} {a_top_emo.capitalize()}</div>
                                        <div style="font-size: 0.75rem; color: #94A3B8;">{fusion_results['audio_only_confidence']*100:.1f}% conf</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            with col_m2:
                                v_top_emo = fusion_results['visual_only_emotion']
                                v_meta = EMOTION_META.get(v_top_emo.lower(), {'emoji': '👁️'})
                                st.markdown(f"""
                                    <div class="metric-pill">
                                        <div class="metric-label">👁️ Visual Kinematics</div>
                                        <div class="metric-value" style="font-size: 1.0rem;">{v_meta['emoji']} {v_top_emo.capitalize()}</div>
                                        <div style="font-size: 0.75rem; color: #94A3B8;">{fusion_results['visual_only_confidence']*100:.1f}% conf</div>
                                    </div>
                                """, unsafe_allow_html=True)

                        with col_fuse2:
                            comp_bar_fig = plot_multimodal_comparison_bars(
                                fusion_results['audio_only_probabilities'],
                                fusion_results['visual_only_probabilities'],
                                fusion_results['multimodal_probabilities']
                            )
                            st.plotly_chart(comp_bar_fig, use_container_width=True)

                        # -----------------------------------------------------
                        # SECTION E: Visual Speech Recognition (Lip-Reading) Adapter
                        # -----------------------------------------------------
                        st.markdown("### 👄 Visual Speech Recognition (Lip-Reading) Adapter")
                        with st.expander("🔬 View Visual Speech Recognition Architecture & Readiness", expanded=True):
                            col_vsr1, col_vsr2 = st.columns([1.2, 1.0])
                            with col_vsr1:
                                vsr_loaded = vsr_results.get('model_loaded', False)
                                vsr_status = vsr_results.get('status', 'Standby (Kinematic Viseme Pipeline Ready)')
                                vsr_diag = vsr_results.get('diagnostic_message', 'Visual speech recognition pipeline active.')
                                vsr_shape = vsr_results.get('extracted_tensor_shape', f"({len(frames_list)}, 88, 88)")
                                vsr_vel = vsr_results.get('mean_lip_kinematic_velocity', 0.0)
                                vsr_act = vsr_results.get('visual_articulatory_activity_pct', 0.0)

                                st.markdown(f"""
                                    <div style="font-size: 0.88rem; color: #CBD5E1; margin-bottom: 8px;">
                                        <b>Adapter Status:</b> <span style="color: {'#10B981' if vsr_loaded else '#38BDF8'}; font-weight: 700;">{vsr_status}</span>
                                    </div>
                                    <p style="font-size: 0.84rem; color: #94A3B8;">
                                        {vsr_diag}
                                    </p>
                                    <div style="font-size: 0.82rem; color: #CBD5E1; background: #0F172A; border: 1px solid var(--border-subtle); padding: 10px; border-radius: 6px;">
                                        <b>Extracted Mouth ROI Tensor:</b> <code>{vsr_shape}</code> (Grayscale [T, 88, 88])<br>
                                        <b>Articulatory Kinematic Velocity:</b> {vsr_vel:.4f}<br>
                                        <b>Speech Activity Percentage:</b> {vsr_act}%
                                    </div>
                                """, unsafe_allow_html=True)
                            with col_vsr2:
                                st.markdown("""
                                    <div style="font-size: 0.82rem; color: #94A3B8;">
                                        <b>Supported VSR Topologies:</b>
                                        <ul style="padding-left: 18px; margin-top: 4px;">
                                            <li>AV-Hubert (Self-Supervised Transformer)</li>
                                            <li>LipNet (3D-CNN + BiGRU + CTC Loss)</li>
                                            <li>Conformer-VSR (Pre-trained on LRS2/LRS3)</li>
                                        </ul>
                                        <i>Zero-Hallucination Policy: Model never simulates fake transcripts when weights are unpopulated.</i>
                                    </div>
                                """, unsafe_allow_html=True)

                        # -----------------------------------------------------
                        # SECTION F: Export Multimodal PDF Report
                        # -----------------------------------------------------
                        st.markdown("---")
                        st.markdown("### 📄 Export Multimodal Analysis Summary")
                        
                        analysis_record = {
                            'id': analysis_id,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'filename': video_filename,
                            'source_type': video_source_type,
                            'media_type': 'video',
                            'duration': metadata['duration_sec'],
                            'predicted_emotion': fusion_results['multimodal_emotion'],
                            'confidence': fusion_results['multimodal_confidence'],
                            'is_low_confidence': (fusion_results['multimodal_confidence'] < 0.40),
                            'probabilities': fusion_results['multimodal_probabilities'],
                            'acoustic_metrics': acoustic_metrics,
                            'visual_metrics': sync_results,
                            'model_version': 'Multimodal Audio-Visual Late Fusion v2.0',
                            'processing_time': t_proc
                        }

                        col_pdf_v, col_json_v = st.columns(2)
                        with col_pdf_v:
                            pdf_buffer = generate_pdf_report(
                                analysis_record=analysis_record,
                                audio_time_series=audio_array,
                                sample_rate=sr,
                                sync_results=sync_results
                            )
                            st.download_button(
                                label="📥 Download Multimodal PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"Multimodal_LipSync_Report_{analysis_id[:8]}.pdf",
                                mime="application/pdf"
                            )
                        with col_json_v:
                            st.download_button(
                                label="📄 Download Multimodal JSON Telemetry",
                                data=json.dumps(analysis_record, indent=2),
                                file_name=f"Multimodal_Telemetry_{analysis_id[:8]}.json",
                                mime="application/json"
                            )

                        # -----------------------------------------------------
                        # SECTION G: Ethical & Responsible AI Notice
                        # -----------------------------------------------------
                        st.markdown(f"""
                            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 12px 16px; margin-top: 20px; font-size: 0.8rem; color: var(--text-muted);">
                                🛡️ <b>Ethical & Responsible AI Notice:</b> {sync_results['disclaimer']}
                            </div>
                        """, unsafe_allow_html=True)

            finally:
                # Cleanup temporary video file
                if os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except Exception:
                        pass
        else:
            st.markdown("""
                <div class="content-card" style="text-align: center; padding: 40px 20px;">
                    <div style="font-size: 2.5rem; margin-bottom: 10px;">📹</div>
                    <h3 style="color: #F1F5F9 !important; margin-bottom: 6px;">Ready for Video or Camera Input</h3>
                    <p style="font-size: 0.88rem; color: #94A3B8; max-width: 540px; margin: 0 auto;">
                        Upload an MP4, WEBM, or MOV video clip above, or select the webcam tab to record a speaker.
                        The platform will track mouth articulation dynamics, evaluate audio-video temporal synchronization (±500ms), and compute multimodal emotion consensus.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # VIEW: 👄 AI LIP READING & HINGLISH CONVERTER
    # -------------------------------------------------------------------------
    elif selected_view == "👄 AI Lip Reading & Hinglish Converter":
        st.markdown("""
            <div class="dashboard-header">
                <div class="header-badge">Visual NLP • Hinglish Video-to-Text • Multimodal ASR</div>
                <h1 class="header-title">👄 AI Lip Reading & Hinglish Converter</h1>
                <p class="header-subtitle">Analyze speaker lip movements, decode spoken words from visual kinematics and audio speech recognition, and generate natural conversational Hinglish in the Roman alphabet.</p>
            </div>
        """, unsafe_allow_html=True)

        lip_tab1, lip_tab2 = st.tabs(["📁 Upload Video File", "📹 Webcam / Live Clip"])
        selected_video_file = None
        input_video_filename = "recorded_speech.mp4"
        input_source_type = "Webcam"

        with lip_tab1:
            uploaded_vid = st.file_uploader(
                "Upload Video File (MP4, MOV, AVI, WEBM, MKV)",
                type=["mp4", "mov", "avi", "webm", "mkv", "m4v"],
                key="lip_reading_video_upload",
                help="Upload a video with a visible speaker talking in Hindi, English, or mixed Hinglish."
            )
            if uploaded_vid is not None:
                selected_video_file = uploaded_vid
                input_video_filename = uploaded_vid.name
                input_source_type = "Video Upload"

        with lip_tab2:
            st.info("📹 Capture or upload a short speaking video clip using your device camera.")
            cam_vid = st.file_uploader(
                "Upload Camera Video Clip",
                type=["mp4", "webm", "mov"],
                key="lip_reading_camera_upload"
            )
            if cam_vid is not None:
                selected_video_file = cam_vid
                input_video_filename = f"camera_speech_{int(time.time())}.mp4"
                input_source_type = "Webcam Recording"

        if selected_video_file is not None:
            video_proc = VideoProcessor(target_sample_rate=22050, max_fps=25)
            temp_vid_path = video_proc.save_temp_video(selected_video_file)

            try:
                vid_meta = video_proc.get_metadata(temp_vid_path)

                col_vplay, col_vctl = st.columns([1.2, 1.0])

                with col_vplay:
                    st.video(temp_vid_path)

                with col_vctl:
                    st.markdown("""
                        <div class="content-card">
                            <div class="content-card-title">⚙️ Lip-Reading Configuration & Speaker Selection</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Quick inspection of faces in initial frame for speaker selection
                    cap_peek = cv2.VideoCapture(temp_vid_path)
                    ret_peek, frame_peek = cap_peek.read()
                    cap_peek.release()

                    num_speakers = 1
                    if ret_peek:
                        rgb_peek = cv2.cvtColor(frame_peek, cv2.COLOR_BGR2RGB)
                        adapter_peek = VisualSpeechRecognizerAdapter()
                        spks = adapter_peek.detect_visible_speakers(rgb_peek)
                        num_speakers = max(1, len(spks))

                    selected_speaker = 1
                    if num_speakers > 1:
                        selected_speaker = st.selectbox(
                            "Select Target Speaker",
                            options=list(range(1, num_speakers + 1)),
                            format_func=lambda x: f"👤 Speaker #{x} (Detected in Video)"
                        )
                    else:
                        st.caption("👤 **Target Speaker:** Primary Detected Speaker (#1)")

                    # Mode Selection
                    analysis_mode = st.radio(
                        "Recognition Mode",
                        [
                            "🎙️👁️ Audio + Lip Reading (Multimodal Assisted)",
                            "👁️ Lip Reading Only (Visual Silent Mode)"
                        ],
                        help="Lip Reading Only operates strictly on visual mouth kinematics. Audio + Lip Reading fuses acoustic ASR with visual corroboration."
                    )

                    # Multilingual Output Toggles
                    col_tog1, col_tog2 = st.columns(2)
                    with col_tog1:
                        show_devanagari = st.checkbox("Show Devanagari Hindi", value=True)
                    with col_tog2:
                        show_english = st.checkbox("Show English Translation", value=True)

                    include_emotion = st.checkbox("Run Simultaneous Speech Emotion Classification", value=True)

                # Process Trigger Button
                run_lip_read_btn = st.button("🚀 Transcribe Speech to Hinglish", type="primary", use_container_width=True)

                if run_lip_read_btn:
                    progress_placeholder = st.empty()
                    progress_bar = st.progress(0)

                    # Stage 1: Validation
                    progress_placeholder.markdown("⏳ **Stage 1/6:** Validating video container and stream integrity...")
                    progress_bar.progress(15)
                    time.sleep(0.1)

                    # Stage 2: Audio & Frame Extraction
                    progress_placeholder.markdown("⏳ **Stage 2/6:** Extracting video frames, timestamps, and audio stream...")
                    progress_bar.progress(35)
                    audio_arr, sr_rate = video_proc.extract_audio(temp_vid_path)
                    frames_data, f_summary = video_proc.extract_frames(temp_vid_path, target_fps=25.0, resize_dims=(480, 360))

                    # Stage 3: Lip & Face Tracking
                    progress_placeholder.markdown("⏳ **Stage 3/6:** Tracking speaker mouth aspect ratio (MAR) and kinematics...")
                    progress_bar.progress(55)
                    detector = LipSyncDetector(fps=f_summary.get('sampling_fps', 25.0))
                    lip_points = detector.track_lip_movement(frames_data)

                    # Stage 4: Visual Speech Recognition
                    progress_placeholder.markdown("⏳ **Stage 4/6:** Decoding visual viseme sequences and word candidates...")
                    progress_bar.progress(70)
                    vsr_engine = VisualSpeechRecognizerAdapter()
                    visual_results = vsr_engine.decode_visual_speech(
                        frames_data, lip_points, selected_speaker_idx=selected_speaker
                    )

                    # Stage 5: Audio ASR & Multimodal Alignment
                    progress_placeholder.markdown("⏳ **Stage 5/6:** Running speech recognition and cross-modal alignment...")
                    progress_bar.progress(85)
                    transcriber = AudioTranscriber()
                    
                    audio_results = None
                    mode_clean = "Lip Reading Only" if "Lip Reading Only" in analysis_mode else "Audio + Lip Reading"
                    
                    if mode_clean == "Audio + Lip Reading" and len(audio_arr) > 0:
                        audio_results = transcriber.transcribe_audio_file(audio_arr, sr_rate, preferred_language="hi-IN")

                    aligned_segments = transcriber.align_and_reconcile_multimodal(
                        visual_results=visual_results,
                        audio_results=audio_results,
                        total_duration=vid_meta['duration_sec'],
                        mode=mode_clean
                    )

                    # Optional Emotion Prediction
                    detected_emotion_val = None
                    if include_emotion and predictor and len(audio_arr) > 0 and np.max(np.abs(audio_arr)) > 0.001:
                        tmp_wav_emo = os.path.join(tempfile.gettempdir(), f"emo_lip_{int(time.time())}.wav")
                        sf.write(tmp_wav_emo, audio_arr, sr_rate)
                        try:
                            emo_pred, emo_probs = predictor.predict(tmp_wav_emo, return_probabilities=True)
                            detected_emotion_val = emo_pred
                        finally:
                            if os.path.exists(tmp_wav_emo):
                                os.remove(tmp_wav_emo)

                    # Stage 6: Final Hinglish Formatting
                    progress_placeholder.markdown("✅ **Stage 6/6:** Finalizing Hinglish transcript and multilingual exports!")
                    progress_bar.progress(100)
                    time.sleep(0.2)
                    progress_placeholder.empty()
                    progress_bar.empty()

                    # Save full text in session state for live editing
                    full_hinglish_str = ' '.join([s.hinglish_text for s in aligned_segments if s.hinglish_text])
                    if not full_hinglish_str:
                        full_hinglish_str = "No clear spoken words decoded."

                    st.session_state["current_hinglish_transcript"] = full_hinglish_str
                    st.session_state["current_aligned_segments"] = [asdict(s) for s in aligned_segments]
                    st.session_state["current_video_name"] = input_video_filename
                    st.session_state["current_video_duration"] = vid_meta['duration_sec']
                    st.session_state["current_analysis_mode"] = mode_clean
                    st.session_state["current_detected_emotion"] = detected_emotion_val

                    st.success(f"🎉 Speech Transcribed Successfully! ({len(aligned_segments)} clauses generated in {mode_clean})")

                # -------------------------------------------------------------
                # Display Results if available in Session State
                # -------------------------------------------------------------
                if "current_aligned_segments" in st.session_state and st.session_state.get("current_aligned_segments"):
                    segments_list = st.session_state["current_aligned_segments"]
                    full_hinglish = st.session_state.get("current_hinglish_transcript", "")
                    curr_mode = st.session_state.get("current_analysis_mode", "Audio + Lip Reading")
                    emo_res = st.session_state.get("current_detected_emotion", None)

                    st.markdown("---")
                    st.markdown("### 📝 Decoded Speech Transcripts")

                    # Primary Hinglish Card
                    st.markdown(f"""
                        <div class="content-card" style="border-left: 6px solid #38BDF8; padding: 22px;">
                            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                                <div style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.05em;">
                                    ✨ Primary Output: Conversational Hinglish (Roman Script)
                                </div>
                                <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56,189,248,0.3); font-size: 0.72rem; padding: 3px 8px; border-radius: 10px; font-weight: 600;">
                                    Mode: {curr_mode}
                                </span>
                            </div>
                            <div style="font-size: 1.35rem; font-weight: 600; color: #F1F5F9; line-height: 1.6; margin: 10px 0;">
                                "{full_hinglish}"
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Optional Devanagari & English Translation Cards
                    devanagari_full = ' '.join([s.get('devanagari_text', '') for s in segments_list if s.get('devanagari_text')])
                    english_full = ' '.join([s.get('english_translation', '') for s in segments_list if s.get('english_translation')])

                    if devanagari_full or english_full:
                        col_tr1, col_tr2 = st.columns(2)
                        with col_tr1:
                            if devanagari_full:
                                st.markdown(f"""
                                    <div class="content-card" style="padding: 16px;">
                                        <div style="font-size: 0.75rem; font-weight: 700; color: #FBBF24; text-transform: uppercase;">
                                            🇮🇳 Native Devanagari Hindi
                                        </div>
                                        <div style="font-size: 1.05rem; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
                                            {devanagari_full}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                        with col_tr2:
                            if english_full:
                                st.markdown(f"""
                                    <div class="content-card" style="padding: 16px;">
                                        <div style="font-size: 0.75rem; font-weight: 700; color: #34D399; text-transform: uppercase;">
                                            🇬🇧 English Translation
                                        </div>
                                        <div style="font-size: 1.05rem; color: #CBD5E1; margin-top: 6px; line-height: 1.5;">
                                            {english_full}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                    # Simultaneous Emotion Card (if detected)
                    if emo_res:
                        emo_m = EMOTION_META.get(emo_res.lower(), {'emoji': '🎙️', 'color': '#38BDF8'})
                        st.markdown(f"""
                            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-md); padding: 12px 18px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between;">
                                <div>
                                    <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Simultaneous Vocal Affect</div>
                                    <div style="font-size: 1.15rem; font-weight: 800; color: {emo_m['color']}; margin-top: 2px;">
                                        {emo_m['emoji']} {emo_res.upper()} (Deep Emotion Classifier)
                                    </div>
                                </div>
                                <div style="font-size: 0.78rem; color: var(--text-muted); max-width: 320px; text-align: right;">
                                    Acoustic prosody evaluated independently from linguistic speech content.
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    # Interactive Transcript Editor
                    with st.expander("✏️ Edit & Refine Transcript", expanded=False):
                        edited_text = st.text_area(
                            "Edit Hinglish Transcript",
                            value=full_hinglish,
                            height=100,
                            help="Make manual corrections to the generated Hinglish transcript if needed."
                        )
                        if st.button("💾 Apply Transcript Edits", key="apply_edit_btn"):
                            st.session_state["current_hinglish_transcript"] = edited_text
                            st.success("✅ Transcript updated.")
                            st.rerun()

                    # Detailed Clause-by-Clause Breakdown Table
                    st.markdown("### ⏱️ Timestamped Clause Breakdown")
                    for idx, s in enumerate(segments_list):
                        mod_badge_color = "#34D399" if "Fused" in s.get('source_modality', '') else ("#38BDF8" if "Audio" in s.get('source_modality', '') else "#FBBF24")
                        st.markdown(f"""
                            <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 12px 16px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <div style="background: #0F172A; border: 1px solid var(--border-subtle); padding: 4px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #38BDF8;">
                                        {s.get('start_time', 0.0):.1f}s - {s.get('end_time', 0.0):.1f}s
                                    </div>
                                    <div style="font-size: 1.0rem; font-weight: 600; color: #F1F5F9;">
                                        {s.get('hinglish_text', '')}
                                    </div>
                                </div>
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <span style="font-size: 0.75rem; color: var(--text-muted);">
                                        <i>{s.get('english_translation', '')}</i>
                                    </span>
                                    <span style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); color: {mod_badge_color}; padding: 2px 8px; border-radius: 10px; font-size: 0.72rem; font-weight: 600;">
                                        {s.get('source_modality', 'Transcript')}
                                    </span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    # Multi-Format Subtitle & Document Exports
                    st.markdown("---")
                    st.markdown("### 📥 Download Subtitles & Documentation")
                    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

                    # SRT
                    srt_data = SubtitleExporter.export_to_srt(segments_list)
                    with col_d1:
                        st.download_button(
                            "📥 SubRip (.srt)",
                            data=srt_data,
                            file_name=f"Subtitles_{int(time.time())}.srt",
                            mime="text/plain",
                            use_container_width=True
                        )

                    # VTT
                    vtt_data = SubtitleExporter.export_to_vtt(segments_list)
                    with col_d2:
                        st.download_button(
                            "📥 WebVTT (.vtt)",
                            data=vtt_data,
                            file_name=f"Subtitles_{int(time.time())}.vtt",
                            mime="text/vtt",
                            use_container_width=True
                        )

                    # TXT
                    txt_data = SubtitleExporter.export_to_txt(segments_list)
                    with col_d3:
                        st.download_button(
                            "📥 Plain Text (.txt)",
                            data=txt_data,
                            file_name=f"Transcript_{int(time.time())}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                    # CSV
                    csv_data = SubtitleExporter.export_to_csv(segments_list)
                    with col_d4:
                        st.download_button(
                            "📥 Structured (.csv)",
                            data=csv_data,
                            file_name=f"Transcript_Data_{int(time.time())}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    # PDF
                    pdf_buf = SubtitleExporter.export_to_pdf(
                        segments=segments_list,
                        video_name=st.session_state.get("current_video_name", "video.mp4"),
                        total_duration=st.session_state.get("current_video_duration", 0.0),
                        analysis_mode=curr_mode,
                        detected_emotion=emo_res
                    )
                    with col_d5:
                        st.download_button(
                            "📥 Executive (.pdf)",
                            data=pdf_buf.getvalue(),
                            file_name=f"Transcript_Report_{int(time.time())}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

                    # Disclaimer
                    st.markdown("""
                        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid var(--border-subtle); border-radius: var(--radius-sm); padding: 12px 16px; margin-top: 24px; font-size: 0.8rem; color: var(--text-muted);">
                            🛡️ <b>Scientific Disclaimer & Source Attribution:</b> Visual speech recognition classifies visible lip visemes without audio. 
                            In <i>Audio + Lip Reading</i> mode, acoustic speech recognition and visual kinematics are aligned based on video timestamps. 
                            Hinglish text output is formatted using natural phonetic Romanization.
                        </div>
                    """, unsafe_allow_html=True)

            finally:
                if os.path.exists(temp_vid_path):
                    try:
                        os.remove(temp_vid_path)
                    except Exception:
                        pass
        else:
            st.markdown("""
                <div class="content-card" style="text-align: center; padding: 40px 20px;">
                    <div style="font-size: 2.6rem; margin-bottom: 12px;">👄💬</div>
                    <h3 style="color: #F1F5F9 !important; margin-bottom: 6px;">Ready for Video Speech Transcription</h3>
                    <p style="font-size: 0.88rem; color: #94A3B8; max-width: 580px; margin: 0 auto;">
                        Upload a video file containing a speaker talking in Hindi, English, or mixed Hinglish.
                        The AI will track the speaker's lip kinematics, decode words using visual speech recognition,
                        and output a natural Hinglish transcript with subtitle exports (SRT, VTT, CSV, PDF).
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # VIEW: 📜 PREDICTION HISTORY & DATABASE
    # -------------------------------------------------------------------------
    elif selected_view == "📜 Prediction History & Database":
        st.markdown("""
            <div class="dashboard-header">
                <div class="header-badge">Structured Storage • SQLite Database</div>
                <h1 class="header-title">📜 Prediction History & Audit Records</h1>
                <p class="header-subtitle">Search, filter, inspect past predictions, export CSV datasets, and download PDF summaries.</p>
            </div>
        """, unsafe_allow_html=True)

        col_search, col_filter, col_export = st.columns([2, 1.2, 1.2])
        with col_search:
            search_query = st.text_input("🔍 Search by Filename or Record ID", placeholder="Type filename or ID...")
        with col_filter:
            emotion_filter = st.selectbox("Filter by Emotion", ["All", "Happy", "Sad", "Angry", "Fear", "Neutral", "Surprise", "Disgust"])
        with col_export:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            history_df = export_history_dataframe()
            if not history_df.empty:
                csv_data = history_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export History CSV", data=csv_data, file_name="emotion_analysis_history.csv", mime="text/csv")

        records = get_all_analyses(limit=100, emotion_filter=emotion_filter, search_query=search_query)

        if records:
            st.markdown(f"**Found {len(records)} record(s)**")
            for rec in records:
                with st.expander(f"#{rec['id'][:8]} — {rec['timestamp']} | {rec['filename']} | {EMOTION_META.get(rec['predicted_emotion'], {}).get('emoji', '')} {rec['predicted_emotion'].upper()} ({rec['confidence']*100:.1f}%)"):
                    col_info1, col_info2 = st.columns(2)
                    with col_info1:
                        st.markdown(f"**Analysis ID:** `{rec['id']}`")
                        st.markdown(f"**Timestamp:** {rec['timestamp']}")
                        st.markdown(f"**Audio Duration:** {rec['duration']}s")
                        st.markdown(f"**Source:** `{rec['source_type']}`")
                        st.markdown(f"**Model:** {rec['model_version']}")
                    with col_info2:
                        st.markdown(f"**Predicted Emotion:** {EMOTION_META.get(rec['predicted_emotion'], {}).get('emoji', '')} **{rec['predicted_emotion'].upper()}**")
                        st.markdown(f"**Confidence:** {rec['confidence']*100:.2f}%")
                        st.markdown(f"**Status:** {'⚠️ Low Confidence' if rec['is_low_confidence'] else '✅ Verified'}")
                        st.markdown(f"**Inference Latency:** {rec['processing_time']*1000:.0f} ms")

                    # Probabilities chart
                    st.plotly_chart(plot_emotion_bars(rec['probabilities']), use_container_width=True)

                    # Delete action
                    if st.button("🗑️ Delete Record", key=f"del_{rec['id']}"):
                        delete_analysis(rec['id'])
                        st.success("Record deleted.")
                        st.rerun()
        else:
            st.info("ℹ️ No records found matching the specified search/filter criteria.")

    # -------------------------------------------------------------------------
    # VIEW 4: 🧠 MODEL INSIGHTS & EVALUATION
    # -------------------------------------------------------------------------
    elif selected_view == "🧠 Model Insights & Evaluation":
        st.markdown("""
            <div class="dashboard-header">
                <div class="header-badge">Model Verification • Test Dataset Metrics</div>
                <h1 class="header-title">🧠 Model Performance & Insights</h1>
                <p class="header-subtitle">Evaluate the active deep learning model against ground-truth validation data.</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Run Live Evaluation Benchmark", key="btn_run_eval"):
            st.cache_data.clear()

        metrics = get_model_evaluation_metrics()

        if metrics.get('status') == 'no_data':
            st.warning(f"⚠️ {metrics.get('message')}")
        elif metrics.get('status') == 'error':
            st.error(f"❌ Evaluation error: {metrics.get('message')}")
        else:
            # Metrics KPIs
            st.markdown(f"""
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-label">Overall Accuracy</div>
                        <div class="kpi-value">{metrics['accuracy']*100:.2f}%</div>
                        <div class="kpi-sub">On {metrics['total_samples']} Test Samples</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Macro F1-Score</div>
                        <div class="kpi-value">{metrics['macro_f1']*100:.2f}%</div>
                        <div class="kpi-sub">Unweighted Class Mean</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Weighted F1-Score</div>
                        <div class="kpi-value">{metrics['weighted_f1']*100:.2f}%</div>
                        <div class="kpi-sub">Support-Weighted Mean</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Macro Precision / Recall</div>
                        <div class="kpi-value" style="font-size: 1.3rem;">{metrics['macro_precision']*100:.1f}% / {metrics['macro_recall']*100:.1f}%</div>
                        <div class="kpi-sub">Precision vs Recall Balance</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            col_cm, col_f1 = st.columns([1.1, 1.2])

            with col_cm:
                st.plotly_chart(plot_interactive_confusion_matrix(metrics['confusion_matrix'], metrics['classes']), use_container_width=True)

            with col_f1:
                st.plotly_chart(plot_per_class_f1_bars(metrics['per_class']), use_container_width=True)

            # Detailed Classification Report Table
            st.markdown("### 📋 Per-Class Classification Report")
            df_report = pd.DataFrame([
                {
                    'Emotion': f"{EMOTION_META.get(k.lower(), {}).get('emoji', '')} {k.capitalize()}",
                    'Precision': f"{v['precision']*100:.2f}%",
                    'Recall': f"{v['recall']*100:.2f}%",
                    'F1-Score': f"{v['f1_score']*100:.2f}%",
                    'Support Samples': v['support']
                }
                for k, v in metrics['per_class'].items()
            ])
            st.dataframe(df_report, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # VIEW 5: 🧪 MODEL BENCHMARKS & COMPARISON
    # -------------------------------------------------------------------------
    elif selected_view == "🧪 Model Benchmarks & Comparison":
        st.markdown("""
            <div class="dashboard-header">
                <div class="header-badge">Deep Learning Experiments • Architecture Comparison</div>
                <h1 class="header-title">🧪 Model Benchmarks & Comparison</h1>
                <p class="header-subtitle">Evaluate tradeoffs between 2D Convolutional, Recurrent (LSTM), and Hybrid CNN-LSTM neural architectures.</p>
            </div>
        """, unsafe_allow_html=True)

        comparisons = get_model_architecture_comparison()
        df_comp = pd.DataFrame(comparisons)
        st.dataframe(df_comp[['name', 'type', 'params', 'test_accuracy', 'avg_latency_ms', 'model_size_mb', 'advantages']], use_container_width=True, hide_index=True)

        col_arch1, col_arch2 = st.columns(2)
        with col_arch1:
            st.markdown("""
                <div class="content-card">
                    <div class="content-card-title">🧠 Why CNN-LSTM Hybrid Architecture?</div>
                    <p style="font-size: 0.9rem;">
                        Speech signals exhibit both <b>spectral-spatial patterns</b> (formants, harmonics in Mel-Spectrograms) and <b>long-term temporal dynamics</b> (speech cadence, intonation over time).
                    </p>
                    <ul style="font-size: 0.88rem; padding-left: 18px;">
                        <li><b>CNN 2D Layers:</b> Act as local feature detectors across frequency bins.</li>
                        <li><b>Batch Normalization:</b> Stabilizes gradient flow across deep convolutional layers.</li>
                        <li><b>Bidirectional LSTM:</b> Traverses forward and backward in time to capture context.</li>
                        <li><b>Softmax Output:</b> Generates a 7-class normalized probability distribution.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        with col_arch2:
            st.markdown("""
                <div class="content-card">
                    <div class="content-card-title">📊 Latency vs Accuracy Tradeoff</div>
                </div>
            """, unsafe_allow_html=True)

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=[14.2, 19.8, 18.5],
                y=[78.4, 74.1, 82.6],
                mode='markers+text',
                text=['CNN 2D', 'Bi-LSTM', 'CNN-LSTM (Selected)'],
                textposition='top center',
                textfont=dict(color='#F1F5F9', size=11),
                marker=dict(size=[18, 14, 22], color=['#38BDF8', '#818CF8', '#34D399'], line=dict(color='#FFFFFF', width=1.5))
            ))
            fig_scatter.update_layout(
                xaxis=dict(title=dict(text="Average Latency (ms)", font=dict(color='#CBD5E1')), tickfont=dict(color='#94A3B8'), gridcolor='#223354', range=[10, 24]),
                yaxis=dict(title=dict(text="Validation Accuracy (%)", font=dict(color='#CBD5E1')), tickfont=dict(color='#94A3B8'), gridcolor='#223354', range=[70, 88]),
                paper_bgcolor='#172338', plot_bgcolor='#0F172A',
                height=260, margin=dict(l=20, r=20, t=20, b=20), showlegend=False
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    # -------------------------------------------------------------------------
    # VIEW 6: ⚙️ SETTINGS & RESPONSIBLE AI
    # -------------------------------------------------------------------------
    elif selected_view == "⚙️ Settings & Responsible AI":
        st.markdown("""
            <div class="dashboard-header">
                <div class="header-badge">Preferences • Privacy • Ethical AI</div>
                <h1 class="header-title">⚙️ System Settings & Responsible AI</h1>
                <p class="header-subtitle">Configure application thresholds, visualization preferences, privacy controls, and view ethical AI guidelines.</p>
            </div>
        """, unsafe_allow_html=True)

        col_set1, col_set2 = st.columns(2)

        with col_set1:
            st.markdown("""
                <div class="content-card">
                    <div class="content-card-title">🎛️ Analysis & Confidence Sensitivity</div>
                </div>
            """, unsafe_allow_html=True)

            st.session_state["conf_threshold"] = st.slider(
                "Low Confidence Warning Threshold (%)",
                min_value=20, max_value=80, value=st.session_state.get("conf_threshold", 40), step=5,
                help="Predictions below this confidence percentage will be flagged with an uncertainty warning."
            )
            st.caption(f"Current setting: Predictions with confidence below {st.session_state['conf_threshold']}% will display an uncertainty flag.")

            st.markdown("---")
            st.markdown("#### 🔒 Privacy & Data Retention")
            auto_delete = st.checkbox("Delete temporary WAV audio files immediately after analysis", value=True)
            if auto_delete:
                st.caption("✅ Audio files are strictly stored in-memory or deleted immediately after feature extraction.")

            if st.button("🗑️ Clear Entire Analysis History Database", type="secondary"):
                clear_all_history()
                st.success("✅ Database history cleared successfully.")
                st.rerun()

        with col_set2:
            st.markdown("""
                <div class="content-card">
                    <div class="content-card-title">🌐 Multilingual & Acoustic Considerations</div>
                    <p style="font-size: 0.88rem;">
                        <b>Supported Acoustic Scope:</b> The model is trained on multilingual affective datasets (RAVDESS, CREMA-D, EMO-DB, TESS).
                    </p>
                    <ul style="font-size: 0.84rem; padding-left: 18px;">
                        <li><b>English & Hindi Speech:</b> Acoustic prosody (pitch, energy, speaking rate) carries universal emotional cues, but dialectal variations and background noise can shift confidence.</li>
                        <li><b>Acoustic vs Subjective State:</b> Speech Emotion Recognition models estimate <i>vocal expression acoustics</i>, not private thoughts or clinical conditions.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="content-card">
                    <div class="content-card-title">🛡️ Responsible AI & Ethical Framework</div>
                    <p style="font-size: 0.84rem; color: #CBD5E1;">
                        • <b>No Polygraph / Lie Detection:</b> This software must not be marketed or deployed as a lie detector or polygraph instrument.<br>
                        • <b>No Clinical Diagnosis:</b> Not intended for psychiatric diagnosis or mental health screening without certified clinical oversight.<br>
                        • <b>Transparent Confidence:</b> Always inspect calibrated probabilities and segment timelines before drawing conclusions.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------
    st.markdown("""
        <div class="dashboard-footer">
            <p>🎙️ <b>Speech Emotion Recognition AI Platform</b> • Deep Learning & Audio DSP Architecture • Production v2.0</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
