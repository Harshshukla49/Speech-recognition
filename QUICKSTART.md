# 🚀 Quick Start Guide

## Getting Started with Speech Emotion Recognition

### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download Datasets

Download one or more of these datasets and place them in the `data/raw/` folder:

1. **RAVDESS** - [Download here](https://zenodo.org/record/1188976)
   - Extract to: `data/raw/RAVDESS/`

2. **CREMA-D** - [Download here](https://github.com/CheyneyComputerScience/CREMA-D)
   - Extract to: `data/raw/CREMA-D/`

3. **EMO-DB** - [Download here](http://emodb.bilderbar.info/start.html)
   - Extract to: `data/raw/EMO-DB/`

### Step 3: Extract Features

```bash
python -c "from src.dataset_loader import load_all_datasets; from src.feature_extraction import batch_extract_features; from src.utils import save_features; df = load_all_datasets(); features, labels = batch_extract_features(df['file_path'].tolist(), df['label'].tolist()); save_features(features, labels)"
```

Or use the Jupyter notebook:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Step 4: Train Model

```bash
# Train with default settings (CNN-LSTM model)
python src/train.py

# Or specify model type
python src/train.py --model cnn_lstm --epochs 100 --batch_size 32

# Available models: cnn, lstm, cnn_lstm, attention_cnn_lstm
```

### Step 5: Make Predictions

```bash
# Predict emotion from an audio file
python src/predict.py --audio_path path/to/your/audio.wav

# With visualization
python src/predict.py --audio_path path/to/your/audio.wav --visualize
```

### Step 6: Launch Web App

```bash
# Run the Streamlit app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

---

## 📚 Usage Examples

### Using Python API

```python
from src.predict import EmotionPredictor

# Load model
predictor = EmotionPredictor()

# Predict emotion
emotion, probabilities = predictor.predict(
    'path/to/audio.wav',
    return_probabilities=True
)

print(f"Detected emotion: {emotion}")
print(f"Probabilities: {probabilities}")
```

### Custom Feature Extraction

```python
from src.preprocessing import preprocess_audio
from src.feature_extraction import extract_all_features

# Preprocess audio
audio = preprocess_audio('path/to/audio.wav', augment=True)

# Extract features
features = extract_all_features(audio, sr=22050)
```

### Training with Custom Parameters

```python
from src.train import train_model

# Train model
model, history = train_model(
    model_type='cnn_lstm',
    epochs=50,
    batch_size=16
)
```

---

## 🔧 Troubleshooting

### Issue: "Model not found"
**Solution**: Train a model first using `python src/train.py`

### Issue: "No datasets found"
**Solution**: Download and extract datasets to `data/raw/` folder

### Issue: "ImportError: No module named 'librosa'"
**Solution**: Install dependencies using `pip install -r requirements.txt`

### Issue: "Tensorflow not found"
**Solution**: Install TensorFlow: `pip install tensorflow==2.13.0`

---

## 📊 Expected Results

After training with the CNN-LSTM model on RAVDESS dataset, you should expect:
- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~75-85%
- **Test Accuracy**: ~75-82%

Results may vary based on:
- Dataset size and quality
- Model architecture
- Training hyperparameters
- Data augmentation techniques

---

## 🎯 Next Steps

1. **Experiment with different models**: Try CNN, LSTM, or attention-based models
2. **Data augmentation**: Increase training data using augmentation techniques
3. **Hyperparameter tuning**: Optimize learning rate, batch size, etc.
4. **Real-time detection**: Implement microphone input for live emotion detection
5. **Deploy to cloud**: Host the Streamlit app on Streamlit Cloud or Heroku

---

## 📧 Need Help?

If you encounter any issues:
1. Check the [GitHub Issues](https://github.com/your-username/speech-emotion-recognition/issues)
2. Read the full documentation in README.md
3. Contact: your.email@example.com

---

**Happy Coding! 🚀**
