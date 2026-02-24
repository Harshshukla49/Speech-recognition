"""
Test imports for the Speech Emotion Recognition project
"""
import sys

print("Testing imports...")
print("-" * 50)

# Test core packages
try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow: {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow: {e}")

try:
    import librosa
    print(f"✅ Librosa: {librosa.__version__}")
except ImportError as e:
    print(f"❌ Librosa: {e}")

try:
    import streamlit as st
    print(f"✅ Streamlit: {st.__version__}")
except ImportError as e:
    print(f"❌ Streamlit: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn: {e}")

print("-" * 50)

# Test project imports
print("\nTesting project modules...")
print("-" * 50)

try:
    sys.path.append('src')
    import config
    print("✅ config module loaded")
except Exception as e:
    print(f"❌ config: {e}")

print("-" * 50)
print("\n✅ Import test complete!")
print("\nRun 'streamlit run app.py' to start the web application")
