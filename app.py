"""
Streamlit Web Application for Speech Emotion Recognition
"""
import os
import sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import librosa
import librosa.display
import matplotlib.pyplot as plt
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    webrtc_streamer = None
    WebRtcMode = None
    RTCConfiguration = None
    WEBRTC_AVAILABLE = False

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import config
from src.predict import EmotionPredictor
from src.utils import get_emotion_color


# Page config
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global Styles */
    body {
        background: #ffffff;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-attachment: fixed;
    }
    .stApp {
        background: transparent;
    }
    .css-1d391kg {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        border-radius: 25px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15), 0 0 0 1px rgba(255,255,255,0.2);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* Header Styles */
    .main-header {
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #ff6b6b, #ffa500, #ffff00, #32cd32, #1e90ff, #9370db, #ff1493);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 2.5rem 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        animation: gradientShift 3s ease-in-out infinite, fadeInUp 1.5s ease-out;
        letter-spacing: -1px;
    }

    /* Emotion Card */
    .emotion-card {
        padding: 3rem;
        border-radius: 30px;
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 50%, #fff5f5 100%);
        margin: 3rem 0;
        box-shadow: 0 25px 50px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.8), inset 0 1px 0 rgba(255,255,255,0.6);
        border: 3px solid transparent;
        background-clip: padding-box;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
    }
    .emotion-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 35px 70px rgba(0,0,0,0.15), 0 0 0 1px rgba(255,255,255,0.9);
    }
    .emotion-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 6px;
        background: linear-gradient(90deg, #ff6b6b, #ffa500, #ffff00, #32cd32, #1e90ff, #9370db, #ff1493);
        background-size: 400% 400%;
        animation: gradientShift 3s ease-in-out infinite;
    }
    .emotion-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #ff6b6b, #ffa500, #ffff00, #32cd32, #1e90ff, #9370db, #ff1493);
        opacity: 0.3;
    }

    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 300% 300%;
        color: white;
        border: none;
        border-radius: 30px;
        padding: 15px 35px;
        font-weight: 700;
        font-size: 16px;
        transition: all 0.4s ease;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3), inset 0 1px 0 rgba(255,255,255,0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    .stButton > button:hover::before {
        left: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255,255,255,0.3);
        animation: buttonGlow 2s ease-in-out infinite;
    }

    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
        border-radius: 20px;
        padding: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 15px;
        background: transparent;
        color: #4a5568;
        font-weight: 600;
        font-size: 15px;
        padding: 12px 20px;
        transition: all 0.3s ease;
        position: relative;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 300% 300%;
        color: white;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        animation: tabGlow 2s ease-in-out infinite;
    }

    /* Sidebar Styles */
    .css-1lcbmhc {
        background: linear-gradient(180deg, rgba(248, 250, 252, 0.95) 0%, rgba(241, 245, 249, 0.95) 50%, rgba(235, 242, 250, 0.95) 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 15px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* File Uploader */
    .stFileUploader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
        border-radius: 20px;
        padding: 25px;
        border: 3px dashed #667eea;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .stFileUploader:hover {
        border-color: #f093fb;
        box-shadow: 0 0 20px rgba(240, 147, 251, 0.3), inset 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Plot Styles */
    .js-plotly-plot {
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.5);
        transition: all 0.3s ease;
    }
    .js-plotly-plot:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15), 0 0 0 1px rgba(255,255,255,0.7);
    }

    /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 15px;
        padding: 15px 20px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* Audio Player */
    .stAudio {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes buttonGlow {
        0%, 100% { box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3), inset 0 1px 0 rgba(255,255,255,0.2); }
        50% { box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5), inset 0 1px 0 rgba(255,255,255,0.3); }
    }

    @keyframes tabGlow {
        0%, 100% { box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3); }
        50% { box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5); }
    }

    /* Typography Enhancements */
    h1, h2, h3, h4, h5, h6 {
        color: #2d3748;
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    p, span, div {
        color: #4a5568;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #f093fb, #f5576c);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
            padding: 1.5rem 0;
        }
        .emotion-card {
            padding: 2rem;
            margin: 2rem 0;
        }
        .css-1d391kg {
            padding: 20px;
            margin: 10px;
        }
    }

    @media (max-width: 480px) {
        .main-header {
            font-size: 2rem;
        }
        .emotion-card {
            padding: 1.5rem;
        }
        .stButton > button {
            padding: 12px 25px;
            font-size: 14px;
        }
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the emotion predictor model"""
    try:
        predictor = EmotionPredictor()
        return predictor
    except Exception as e:
        st.warning(f"⚠️ Model not available: {str(e)}")
        st.info("To use predictions, train a model first using `python run_pipeline.py`")
        return None


def plot_waveform(audio, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time, audio, color='#667eea', linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_spectrogram(audio, sr):
    """Plot mel spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel',
                                    cmap='viridis', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_emotion_bars(probabilities):
    """Plot emotion probabilities as horizontal bars"""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    emotions = [x[0] for x in sorted_probs]
    probs = [x[1] * 100 for x in sorted_probs]
    colors_list = [get_emotion_color(e) for e in emotions]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=emotions,
        orientation='h',
        marker=dict(color=colors_list),
        text=[f'{p:.1f}%' for p in probs],
        textposition='outside',
    ))
    
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Probability (%)",
        yaxis_title="Emotion",
        height=400,
        showlegend=False,
        font=dict(size=14),
        xaxis=dict(range=[0, 100])
    )
    
    return fig


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🎙️ Speech Emotion Recognition</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info(
            """
            This application uses **Deep Learning** to detect emotions from speech.
            
            **Supported Emotions:**
            - 😊 Happy
            - 😢 Sad
            - 😠 Angry
            - 😨 Fear
            - 😐 Neutral
            - 😲 Surprise
            - 🤢 Disgust
            
            **How to use:**
            1. Upload an audio file (WAV format recommended)
            2. Or record your voice
            3. Get instant emotion prediction!
            """
        )
        
        st.header("⚙️ Settings")
        show_waveform = st.checkbox("Show Waveform", value=True)
        show_spectrogram = st.checkbox("Show Spectrogram", value=True)
        show_probabilities = st.checkbox("Show All Probabilities", value=True)
    
    # Load model
    predictor = load_predictor()
    
    if predictor is None:
        st.warning("⚠️ No trained model available. You can still explore the interface!")
        st.info("""
        **To enable predictions:**
        1. The sample dataset has been created: `data/raw/SAMPLE/`
        2. Run the complete pipeline: `python run_pipeline.py`
        3. This will train a model and enable predictions
        
        **Or explore the demo features below without a model!**
        """)
        # Continue to show the interface even without a model
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Upload Audio", "🎤 Record Audio", "ℹ️ Information"])
    
    # Tab 1: Upload Audio
    with tab1:
        st.header("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload an audio file to analyze emotions"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Display audio player
            st.audio(uploaded_file)
            
            # Process button
            if st.button("🔍 Analyze Emotion", type="primary", key="upload_analyze"):
                if predictor is None:
                    st.error("❌ No trained model available. Please run `python run_pipeline.py` first to train a model.")
                else:
                    with st.spinner("Analyzing audio..."):
                        try:
                            # Load audio for visualization
                            audio, sr = librosa.load(temp_path, duration=config.DURATION)
                            
                            # Make prediction
                            emotion, probabilities = predictor.predict(temp_path, return_probabilities=True)
                            
                            # Display result
                            st.success("✅ Analysis Complete!")
                            
                            # Main result
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.markdown(f"""
                                    <div class="emotion-card">
                                        <h2 style="text-align: center; margin: 0;">Detected Emotion</h2>
                                        <h1 style="text-align: center; font-size: 4rem; margin: 1rem 0;">
                                            {emotion.upper()}
                                        </h1>
                                        <p style="text-align: center; font-size: 1.5rem; margin: 0;">
                                            Confidence: {probabilities[emotion]*100:.2f}%
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Visualizations
                            if show_probabilities:
                                st.plotly_chart(plot_emotion_bars(probabilities), use_container_width=True)
                            
                            if show_waveform:
                                st.pyplot(plot_waveform(audio, sr))
                            
                            if show_spectrogram:
                                st.pyplot(plot_spectrogram(audio, sr))
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing audio: {str(e)}")
                        finally:
                            # Clean up
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
    
    # Tab 2: Record Audio
    with tab2:
        st.header("Record Your Voice")

        if not WEBRTC_AVAILABLE:
            st.warning("🎤 Live recording is unavailable in this deployment environment.")
            st.info("Use the 'Upload Audio' tab to analyze a local audio file.")
            webrtc_ctx = None
        else:
            st.markdown("🎤 Click 'START' to begin recording. Speak clearly, then click 'STOP' to end recording.")

            # WebRTC Configuration
            RTC_CONFIGURATION = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })

            # WebRTC Streamer for audio recording
            webrtc_ctx = webrtc_streamer(
                key="audio-recorder",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=1024,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"audio": True, "video": False},
            )

            # Record button
            if st.button("🎙️ Record Audio", key="start_record"):
                st.info("🎤 Recording started! Click 'STOP' when finished.")

        # Check if audio data is available
        if webrtc_ctx and webrtc_ctx.audio_receiver:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)

            if len(audio_frames) > 0:
                st.success("✅ Audio recorded successfully!")

                # Convert frames to audio data
                import av
                audio_data = b"".join(frame.to_ndarray().tobytes() for frame in audio_frames)

                # Save as WAV file
                import soundfile as sf
                import io

                # Convert raw audio to numpy array (assuming 16-bit PCM)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Save to temporary WAV file
                temp_path = "recorded_audio.wav"
                sf.write(temp_path, audio_array, 44100)  # Assuming 44.1kHz sample rate

                # Play back the recorded audio
                st.audio(temp_path)

                # Analyze button
                if st.button("🔍 Analyze Recorded Emotion", type="primary", key="record_analyze"):
                    if predictor is None:
                        st.error("❌ No trained model available. Please run `python run_pipeline.py` first to train a model.")
                    else:
                        with st.spinner("Analyzing recorded audio..."):
                            try:
                                # Load audio for visualization
                                audio, sr = librosa.load(temp_path, duration=config.DURATION)

                                # Make prediction
                                emotion, probabilities = predictor.predict(temp_path, return_probabilities=True)

                                # Display result
                                st.success("✅ Analysis Complete!")

                                # Main result
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.markdown(f"""
                                        <div class="emotion-card">
                                            <h2 style="text-align: center; margin: 0;">Detected Emotion</h2>
                                            <h1 style="text-align: center; font-size: 4rem; margin: 1rem 0;">
                                                {emotion.upper()}
                                            </h1>
                                            <p style="text-align: center; font-size: 1.5rem; margin: 0;">
                                                Confidence: {probabilities[emotion]*100:.2f}%
                                            </p>
                                        </div>
                                    """, unsafe_allow_html=True)

                                # Visualizations
                                if show_probabilities:
                                    st.plotly_chart(plot_emotion_bars(probabilities), use_container_width=True)

                                if show_waveform:
                                    st.pyplot(plot_waveform(audio, sr))

                                if show_spectrogram:
                                    st.pyplot(plot_spectrogram(audio, sr))

                            except Exception as e:
                                st.error(f"❌ Error analyzing recorded audio: {str(e)}")
                            finally:
                                # Clean up
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
            else:
                st.info("🎙️ Click 'START' to begin recording your voice.")
    
    # Tab 3: Information
    with tab3:
        st.header("ℹ️ About This Project")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Features")
            st.markdown("""
                - Audio preprocessing (noise removal, normalization)
                - Feature extraction using Librosa
                - Deep learning models (CNN, LSTM, CNN-LSTM)
                - Real-time emotion detection
                - Interactive visualizations
            """)
            
            st.subheader("📊 Datasets")
            st.markdown("""
                - **RAVDESS**: Ryerson Audio-Visual Database
                - **CREMA-D**: Crowd-sourced Emotional Multimodal Actors
                - **EMO-DB**: Berlin Database of Emotional Speech
            """)
        
        with col2:
            st.subheader("🛠️ Technology Stack")
            st.markdown("""
                - **Python 3.9+**
                - **TensorFlow/Keras** for deep learning
                - **Librosa** for audio processing
                - **Streamlit** for web interface
                - **Plotly** for interactive charts
            """)
            
            st.subheader("📈 Model Performance")
            st.markdown("""
                - **Accuracy**: ~82.6%
                - **F1-Score**: ~0.81
                - **Model**: CNN-LSTM Hybrid
            """)
        
        st.markdown("---")
        st.markdown("""
            <div style="text-align: center;">
                <p><strong>Built with ❤️ using Python and Deep Learning</strong></p>
                <p>For more information, visit the <a href="https://github.com/your-username/speech-emotion-recognition">GitHub Repository</a></p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
