"""
Demo script for Speech Emotion Recognition
This script demonstrates the preprocessing and feature extraction pipeline.
"""
import sys
import os
import numpy as np

print("="*70)
print("🎙️  SPEECH EMOTION RECOGNITION - DEMO")
print("="*70)

# Check imports
print("\n1. Checking required packages...")
try:
    import numpy as np
    print("   ✅ NumPy:", np.__version__)
except ImportError as e:
    print("   ❌ NumPy not installed:", e)

try:
    import pandas as pd
    print("   ✅ Pandas:", pd.__version__)
except ImportError as e:
    print("   ❌ Pandas not installed:", e)

try:
    import matplotlib
    print("   ✅ Matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("   ❌ Matplotlib not installed:", e)

try:
    import librosa
    print("   ✅ Librosa:", librosa.__version__)
except ImportError as e:
    print("   ❌ Librosa not installed:", e)
    print("      Install with: pip install librosa")

try:
    import tensorflow as tf
    print("   ✅ TensorFlow:", tf.__version__)
except ImportError as e:
    print("   ❌ TensorFlow not installed:", e)
    print("      Install with: pip install tensorflow")

try:
    import streamlit
    print("   ✅ Streamlit:", streamlit.__version__)
except ImportError as e:
    print("   ❌ Streamlit not installed:", e)
    print("      Install with: pip install streamlit")

try:
    import sklearn
    print("   ✅ Scikit-learn:", sklearn.__version__)
except ImportError as e:
    print("   ❌ Scikit-learn not installed:", e)

# Check project structure
print("\n2. Checking project structure...")
required_dirs = [
    'data/raw',
    'data/processed',
    'models',
    'checkpoints',
    'src',
    'notebooks'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ❌ {dir_path} (missing)")

# Check source files
print("\n3. Checking source files...")
source_files = [
    'src/__init__.py',
    'src/config.py',
    'src/preprocessing.py',
    'src/feature_extraction.py',
    'src/model.py',
    'src/train.py',
    'src/predict.py',
    'src/utils.py',
    'src/dataset_loader.py'
]

for file_path in source_files:
    if os.path.exists(file_path):
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} (missing)")

# Check for datasets
print("\n4. Checking for datasets...")
dataset_paths = [
    'data/raw/RAVDESS',
    'data/raw/CREMA-D',
    'data/raw/EMO-DB'
]

datasets_found = False
for dataset_path in dataset_paths:
    if os.path.exists(dataset_path):
        print(f"   ✅ {dataset_path}")
        datasets_found = True
    else:
        print(f"   ⚠️  {dataset_path} (not found)")

if not datasets_found:
    print("\n   ℹ️  No datasets found. Download from:")
    print("      - RAVDESS: https://zenodo.org/record/1188976")
    print("      - CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D")
    print("      - EMO-DB: http://emodb.bilderbar.info/start.html")

# Check for trained model
print("\n5. Checking for trained model...")
if os.path.exists('models/best_model.h5'):
    print("   ✅ Trained model found: models/best_model.h5")
else:
    print("   ⚠️  No trained model found")
    print("      Train a model using: python src/train.py")

# Next steps
print("\n" + "="*70)
print("📋 NEXT STEPS:")
print("="*70)

if not datasets_found:
    print("1. Download at least one dataset and place in data/raw/")
    print("2. Extract features: See QUICKSTART.md")
    print("3. Train model: python src/train.py")
    print("4. Run web app: streamlit run app.py")
else:
    if not os.path.exists('data/processed/features.npy'):
        print("1. Extract features from datasets")
        print("2. Train model: python src/train.py")
        print("3. Run web app: streamlit run app.py")
    elif not os.path.exists('models/best_model.h5'):
        print("1. Train model: python src/train.py")
        print("2. Run web app: streamlit run app.py")
    else:
        print("✅ Everything is ready!")
        print("   Run: streamlit run app.py")

print("\n" + "="*70)
print("For more information, see README.md and QUICKSTART.md")
print("="*70)
