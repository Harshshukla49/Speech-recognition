# 🎙️ Speech Emotion Recognition using Deep Learning

## 📌 Overview
This project focuses on detecting human emotions (such as happiness, anger, sadness, fear, and surprise) from speech signals using **deep learning techniques**.  
By leveraging audio feature extraction (MFCCs, Mel-Spectrograms) and models like **CNNs, RNNs, and hybrid CNN-LSTM architectures**, the system aims to classify emotions with high accuracy.

---

## 🚀 Features
- ✅ Preprocessing of raw audio files (noise removal, normalization).
- ✅ Feature extraction using **Librosa** (MFCCs, Chroma, Mel-Spectrograms).
- ✅ Deep learning models built with **TensorFlow/PyTorch**.
- ✅ Evaluation with accuracy, F1-score, and confusion matrix.
- ✅ Real-time emotion detection demo using microphone input.

---

## 📂 Dataset
We use publicly available datasets:
- [RAVDESS](https://zenodo.org/record/1188976) - Ryerson Audio-Visual Database of Emotional Speech and Song
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) - Crowd-sourced Emotional Multimodal Actors Dataset
- [EMO-DB](http://emodb.bilderbar.info/start.html) - Berlin Database of Emotional Speech

### Supported Emotions
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fear
- 😐 Neutral
- 😲 Surprise
- 🤢 Disgust

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **Librosa** for audio processing
- **NumPy, Pandas, Matplotlib** for data handling & visualization
- **TensorFlow / Keras** for deep learning models
- **Scikit-learn** for evaluation metrics
- **Streamlit** for web deployment

---

## 📁 Project Structure
```
speech-emotion-recognition/
│
├── data/
│   ├── raw/                    # Raw audio files
│   └── processed/              # Processed features
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── preprocessing.py        # Audio preprocessing
│   ├── feature_extraction.py  # Feature extraction utilities
│   ├── model.py                # Model architectures
│   ├── train.py                # Training script
│   ├── predict.py              # Prediction utilities
│   └── utils.py                # Helper functions
│
├── models/                     # Saved trained models
├── checkpoints/                # Training checkpoints
├── notebooks/                  # Jupyter notebooks for experiments
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 Usage

### Training the Model
```bash
python src/train.py --dataset RAVDESS --epochs 100 --batch_size 32
```

### Making Predictions
```bash
python src/predict.py --audio_path path/to/audio.wav
```

### Running the Web App
```bash
streamlit run app.py
```

---

## 🧪 Model Architecture

### CNN-LSTM Hybrid Model
- **Convolutional Layers**: Extract spatial features from spectrograms
- **LSTM Layers**: Capture temporal dependencies
- **Dropout**: Prevent overfitting
- **Dense Layers**: Final classification

```
Input (128x128x1) → Conv2D → MaxPooling → Conv2D → MaxPooling
    → Reshape → LSTM → Dropout → Dense → Output (7 emotions)
```

---

## 📊 Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| CNN   | 72.5%    | 0.71     |
| LSTM  | 75.3%    | 0.74     |
| CNN-LSTM | **82.6%** | **0.81** |

---

## 🔍 Feature Extraction

**Audio Features Used:**
- **MFCCs** (Mel-Frequency Cepstral Coefficients)
- **Mel-Spectrograms**
- **Chroma Features**
- **Zero Crossing Rate**
- **Spectral Centroid**
- **Spectral Rolloff**

---

## 🎤 Real-time Detection
The application supports real-time emotion detection using your microphone:
1. Click "Record" in the Streamlit app
2. Speak for 3-5 seconds
3. Get instant emotion prediction with confidence scores

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments
- RAVDESS, CREMA-D, and EMO-DB dataset creators
- Librosa library developers
- TensorFlow/Keras community

---

## 📧 Contact
For questions or feedback, please reach out to:
- GitHub: [@your-username](https://github.com/your-username)
- Email: your.email@example.com

---

**Happy Coding! 🚀**
