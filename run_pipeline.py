"""
Complete Pipeline: Load Data → Extract Features → Train Model
"""
import sys
import os
sys.path.append('src')

print("\n" + "="*70)
print("🎙️ SPEECH EMOTION RECOGNITION - COMPLETE PIPELINE")
print("="*70)

# Step 1: Load Dataset
print("\n📂 STEP 1: Loading Dataset...")
from src.dataset_loader import load_all_datasets
from src.utils import save_features

df = load_all_datasets()

if df.empty:
    print("❌ No datasets found!")
    print("   Please run: python create_sample_data.py")
    sys.exit(1)

# Step 2: Extract Features
print("\n🔧 STEP 2: Extracting Features...")
from src.feature_extraction import batch_extract_features

features, labels = batch_extract_features(
    df['file_path'].tolist(),
    df['label'].tolist(),
    feature_type='mel_spectrogram',
    verbose=True
)

# Save features
save_features(features, labels)

# Step 3: Train Model
print("\n🧠 STEP 3: Training Model...")
from src.train import train_model

model, history = train_model(
    model_type='cnn_lstm',
    epochs=50,  # Reduced for faster training on sample data
    batch_size=16
)

print("\n" + "="*70)
print("✅ PIPELINE COMPLETE!")
print("="*70)
print("\nYou can now:")
print("1. Make predictions: python src/predict.py --audio_path path/to/audio.wav")
print("2. Run web app: streamlit run app.py")
print("="*70 + "\n")
