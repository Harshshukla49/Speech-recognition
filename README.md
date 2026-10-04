# 🎙️ Speech Emotion Recognition (SER) — Production AI Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Deep Learning & Audio Digital Signal Processing (DSP) platform for real-time **Speech Emotion Recognition (SER)**. Classifies 7 discrete human emotional affect taxonomies using 128-band Mel-Spectrograms, Mel-Frequency Cepstral Coefficients (MFCCs), and a hybrid **CNN-LSTM Neural Network Architecture**.

---

## 📌 Architectural Overview

```mermaid
flowchart LR
    A[Raw Speech Audio] --> B[Acoustic Preprocessing]
    B --> C[128-Mel Filterbank & MFCC Extraction]
    C --> D[2D Convolutional Feature Extractor]
    D --> E[Bidirectional LSTM Sequential Context]
    E --> F[Softmax Probability Classification Head]
    F --> G[7-Class Emotion Probabilities]
    G --> H[Interactive Dark Dashboard & PDF Reports]
```

### Supported Affective Categories
* 😊 **Happy**
* 😢 **Sad**
* 😠 **Angry**
* 😨 **Fear**
* 😐 **Neutral**
* 😲 **Surprise**
* 🤢 **Disgust**

---

## 🚀 Key Platform Features

### 1. 🎙️ Advanced Speech Analysis Workspace
* **Dual Ingestion**: File Upload (WAV, MP3, FLAC, OGG) and Browser Microphone Recording (`st.audio_input`) with hardware fallback (`PyAudio`).
* **Signal Validation**: Real-time silence detection, amplitude normalization ($L_2$), clipping checks, and sample rate harmonization ($22.05\text{ kHz}$).
* **Segment-Wise Timeline**: Overlapping windowed classification for long audio files, displaying temporal emotion trajectories over time.

### 2. 🔬 Multi-Dimensional Acoustic Signal Descriptors
* **Time-Domain Waveform**: Amplitude envelope variations over duration.
* **128-Band Mel-Spectrogram**: Perceptually-scaled frequency energy density in decibels ($\text{dB}$).
* **40-Coefficient MFCC Heatmap**: Acoustic cepstral coefficients highlighting formant structures.
* **Acoustic Metrics**: Pitch estimation ($F_0$ via YIN), RMS Energy, Zero Crossing Rate (ZCR), Spectral Centroid, Spectral Rolloff, and Silence %.

### 3. 📜 SQLite Database & Prediction History
* Full persistence of past analyses (ID, Timestamp, Audio Name, Emotion, Confidence, Probabilities, Acoustic Descriptors, Model Version, Latency).
* Search, filter by emotion category, single-record deletion, and batch CSV export.

### 4. 📄 Executive PDF Report Generation
* Downloadable audit summary generated on-the-fly via ReportLab.
* Contains analysis IDs, primary emotion badge, 7-class probability breakdown table, acoustic descriptors table, embedded high-resolution waveforms and spectrogram plots, and scientific disclaimers.

### 5. 👁️ AI Lip-Sync Detection & Visual Speech Analysis
* **Video Ingestion & Demuxing**: High-fidelity video upload (MP4, MOV, AVI, WEBM, MKV) and camera capture with PyAV/OpenCV frame and audio stream extraction.
* **Mouth Aspect Ratio (MAR) & Velocity Tracking**: Multi-tier tracking estimating mouth opening ratio $\text{MAR}(t)$, oral cavity contour geometry, and lip kinematic velocity $v_{\text{lip}}(t)$.
* **Audio-Visual Synchronization (Lip-Sync)**: Temporal cross-correlation $R_{EL}(\tau)$ comparing acoustic RMS energy envelopes with visual articulation dynamics, estimating temporal offset $\Delta t$ (ms), Sync Quality Index (SQI 0-100%), and broadcast standard alignment.
* **Visual Speech Recognition (VSR / Lip-Reading) Adapter**: Extensible adapter for 3D-CNN + Conformer silent lip-reading architectures (e.g. AV-Hubert, LipNet), with zero-hallucination diagnostics.
* **Multimodal Emotion Fusion**: Adaptive decision-level late fusion combining acoustic predictions with visual facial dynamics.

### 6. 🧠 Model Insights & Empirical Evaluation
* Multi-class Interactive Confusion Matrix (counts & normalized percentages).
* Per-Class Precision, Recall, and F1-Score grouped bar visualization.
* Model Architecture Comparison Suite (2D CNN vs Bi-LSTM vs CNN-LSTM Hybrid).

### 7. 🛡️ Responsible AI & Privacy
* Configurable low-confidence uncertainty warnings (default $< 40\%$).
* In-memory/ephemeral audio/video retention policy.
* Explicit non-polygraph and non-clinical psychiatric disclaimer.
* Transparent acknowledgment that A/V timing offsets can be caused by hardware, Bluetooth, or container codecs.

---

## 📁 Project Structure

```
speech-emotion-recognition/
│
├── data/
│   ├── raw/                      # Audio datasets (RAVDESS, CREMA-D, EMO-DB, TESS)
│   ├── processed/                # Processed feature arrays (features.npy, labels.npy)
│   └── history.db                # SQLite analysis history database
│
├── models/
│   ├── best_model.h5             # Trained CNN-LSTM model weights
│   ├── confusion_matrix.png      # Training benchmark confusion matrix
│   ├── training_history.png      # Loss and accuracy learning curves
│   └── visual_speech/            # Directory for VSR lip-reading model checkpoints
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Central configuration constants
│   ├── audio_analyzer.py         # Acoustic feature extraction & segment timelines
│   ├── database.py               # SQLite database CRUD operations & analytics
│   ├── report_generator.py       # Automated PDF report builder (ReportLab)
│   ├── evaluation.py             # Empirical metrics, confusion matrix & comparisons
│   ├── evaluate.py               # Standalone CLI evaluation script
│   ├── preprocessing.py          # Audio loading, resampling & silence removal
│   ├── feature_extraction.py     # Mel-spectrogram & MFCC extraction
│   ├── model.py                  # CNN, LSTM, and CNN-LSTM model architectures
│   ├── train.py                  # Model training and checkpoint pipeline
│   ├── predict.py                # Inference and batch prediction utilities
│   ├── utils.py                  # Visualizations, palettes, and I/O helpers
│   ├── video_processor.py        # Video ingestion, metadata inspection & audio demuxing
│   ├── lip_sync_detector.py      # Face landmarking, MAR & A/V cross-correlation
│   ├── visual_speech_recognizer.py # Silent visual speech recognition adapter
│   └── multimodal_fusion.py      # Audio-visual late fusion decision engine
│
├── tests/
│   ├── test_platform.py          # Speech emotion platform test suite (6/6 passing)
│   └── test_lip_sync.py          # Lip-sync & multimodal test suite (5/5 passing)
│
├── app.py                        # Multi-view Streamlit AI Dashboard
├── requirements.txt              # Production Python dependencies
└── README.md                     # Technical documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Create and activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 Running the Application

### Start the AI Dashboard
```bash
# Launch on localhost & LAN
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
* **Localhost URL:** [http://localhost:8501](http://localhost:8501)
* **Local Network URL:** `http://<your-lan-ip>:8501`

---

## 🧪 Testing & Evaluation

### Run the Automated Unit Test Suite
```bash
python -m unittest tests/test_platform.py
```

### Run the Standalone Model Evaluation Benchmark
```bash
python src/evaluate.py
```

---

## 🧠 Model Architecture Details

| Layer | Type | Output Shape | Parameters | Activation |
|---|---|---|---|---|
| 1 | `Conv2D (64 filters, 3x3)` | `(128, 128, 64)` | 640 | ReLU + BatchNorm |
| 2 | `MaxPooling2D (2x2)` | `(64, 64, 64)` | 0 | Dropout (0.25) |
| 3 | `Conv2D (128 filters, 3x3)` | `(64, 64, 128)` | 73,856 | ReLU + BatchNorm |
| 4 | `MaxPooling2D (2x2)` | `(32, 32, 128)` | 0 | Dropout (0.25) |
| 5 | `Conv2D (256 filters, 3x3)` | `(32, 32, 256)` | 295,168 | ReLU + BatchNorm |
| 6 | `MaxPooling2D (2x2)` | `(16, 16, 256)` | 0 | Dropout (0.25) |
| 7 | `TimeDistributed(Flatten)` | `(16, 4096)` | 0 | - |
| 8 | `Bidirectional(LSTM 128)` | `(16, 256)` | 4,326,400 | Tanh / Sigmoid |
| 9 | `Dense (256)` | `(256)` | 65,792 | ReLU + Dropout (0.4) |
| 10 | `Dense (7)` | `(7)` | 1,799 | Softmax |

**Total Trainable Parameters:** ~`8,321,223`  
**Input Shape:** `(128, 128, 1)` representing $128$ Mel-bands $\times$ $128$ time frames.

---

## 🛡️ Ethical AI & Limitations

* **Acoustic vs Subjective Emotional State**: Speech Emotion Recognition assesses acoustic cues in vocal prosody (pitch modulation, spectral energy, articulation rate). These acoustic cues correlate with affective arousal and valence but cannot definitively prove a person's private thoughts or intent.
* **Non-Polygraph Usage**: This application is strictly prohibited from being utilized as a lie detector, polygraph tool, or employment vetting instrument.
* **Non-Clinical**: Not intended for psychiatric diagnosis or mental health screening without certified clinical supervision.
* **Linguistic Variability**: Accents, dialects, background acoustics, and speaking styles can influence classification probabilities. Always consult segment timelines and confidence scores.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
