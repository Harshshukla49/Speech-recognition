"""
Download Speech Emotion Recognition Dataset
This script downloads the TESS (Toronto Emotional Speech Set) dataset
"""
import os
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)

def download_tess_dataset():
    """Download TESS dataset from Kaggle mirror"""
    print("\n" + "="*70)
    print("📥 DOWNLOADING TESS DATASET")
    print("="*70)
    
    # Create directories
    os.makedirs('data/raw/TESS', exist_ok=True)
    
    # TESS dataset URL (from Kaggle mirror/public source)
    # Note: You may need to provide your own URL or use Kaggle API
    print("\nℹ️  Due to licensing, datasets need to be downloaded manually.")
    print("\nPlease download one of these datasets:")
    print("\n1. RAVDESS (Recommended)")
    print("   URL: https://zenodo.org/record/1188976")
    print("   Size: ~24 GB")
    print("   Extract to: data/raw/RAVDESS/")
    
    print("\n2. CREMA-D")
    print("   URL: https://github.com/CheyneyComputerScience/CREMA-D")
    print("   Size: ~4 GB")
    print("   Extract to: data/raw/CREMA-D/")
    
    print("\n3. EMO-DB (Small, Good for Testing)")
    print("   URL: http://emodb.bilderbar.info/download/")
    print("   Size: ~30 MB")
    print("   Extract to: data/raw/EMO-DB/")
    
    print("\n4. Sample Dataset (Generated)")
    print("   Create synthetic samples for testing")
    
    choice = input("\nEnter choice (1-4) or press Enter to generate sample dataset: ").strip()
    
    if choice == "" or choice == "4":
        create_sample_dataset()
    else:
        print("\n✅ Please download the dataset manually and extract to the appropriate folder.")
        print("   Then run: python demo.py to check status")

def create_sample_dataset():
    """Create a small sample dataset for testing"""
    print("\n" + "="*70)
    print("🎨 CREATING SAMPLE DATASET")
    print("="*70)
    
    try:
        import numpy as np
        import soundfile as sf
        from scipy import signal
    except ImportError:
        print("❌ Required packages not installed. Installing...")
        os.system("pip install numpy scipy soundfile")
        import numpy as np
        import soundfile as sf
        from scipy import signal
    
    # Create sample directory
    sample_dir = 'data/raw/SAMPLE'
    os.makedirs(sample_dir, exist_ok=True)
    
    # Emotion configurations
    emotions = {
        'neutral': {'freq_range': (200, 300), 'amplitude': 0.3},
        'happy': {'freq_range': (400, 600), 'amplitude': 0.5},
        'sad': {'freq_range': (150, 250), 'amplitude': 0.2},
        'angry': {'freq_range': (300, 500), 'amplitude': 0.7},
        'fear': {'freq_range': (350, 550), 'amplitude': 0.6},
        'disgust': {'freq_range': (250, 400), 'amplitude': 0.4},
        'surprise': {'freq_range': (450, 650), 'amplitude': 0.6}
    }
    
    sr = 22050  # Sample rate
    duration = 3  # seconds
    samples_per_emotion = 10
    
    print(f"\nGenerating {samples_per_emotion} samples for each of {len(emotions)} emotions...")
    
    total_files = 0
    for emotion, config in emotions.items():
        emotion_dir = os.path.join(sample_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        for i in range(samples_per_emotion):
            # Generate time array
            t = np.linspace(0, duration, int(sr * duration))
            
            # Generate audio with varying frequencies
            freq = np.random.uniform(config['freq_range'][0], config['freq_range'][1])
            audio = config['amplitude'] * np.sin(2 * np.pi * freq * t)
            
            # Add harmonics
            audio += 0.3 * config['amplitude'] * np.sin(2 * np.pi * freq * 2 * t)
            audio += 0.2 * config['amplitude'] * np.sin(2 * np.pi * freq * 3 * t)
            
            # Add some noise for realism
            noise = np.random.normal(0, 0.05, len(audio))
            audio += noise
            
            # Apply envelope (fade in/out)
            envelope = np.ones_like(audio)
            fade_samples = int(0.1 * sr)  # 0.1 second fade
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            audio *= envelope
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.9
            
            # Save to file
            filename = f"{emotion}_{i+1:02d}.wav"
            filepath = os.path.join(emotion_dir, filename)
            sf.write(filepath, audio, sr)
            total_files += 1
        
        print(f"  ✅ Created {samples_per_emotion} samples for '{emotion}'")
    
    print(f"\n✅ Sample dataset created successfully!")
    print(f"   Total files: {total_files}")
    print(f"   Location: {sample_dir}")
    print(f"\nℹ️  Note: This is a SYNTHETIC dataset for testing only.")
    print("   For real emotion recognition, please download actual speech datasets.")
    
    # Create a mapping file
    create_sample_dataset_loader()

def create_sample_dataset_loader():
    """Create a loader for the sample dataset"""
    loader_code = '''"""
Dataset loader for sample dataset
"""
import os
import glob
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import EMOTIONS, EMOTION_TO_IDX

def load_sample_dataset(data_path='data/raw/SAMPLE'):
    """Load sample dataset"""
    if not os.path.exists(data_path):
        print(f"⚠️  Sample dataset path not found: {data_path}")
        return pd.DataFrame()
    
    file_paths = []
    labels = []
    
    # Map folder names to emotion indices
    emotion_map = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'fear': 4,
        'disgust': 5,
        'surprise': 6
    }
    
    # Find all audio files
    for emotion, label in emotion_map.items():
        emotion_path = os.path.join(data_path, emotion)
        if os.path.exists(emotion_path):
            audio_files = glob.glob(os.path.join(emotion_path, '*.wav'))
            file_paths.extend(audio_files)
            labels.extend([label] * len(audio_files))
    
    df = pd.DataFrame({
        'file_path': file_paths,
        'label': labels,
        'emotion': [EMOTIONS[label] for label in labels],
        'dataset': 'SAMPLE'
    })
    
    print(f"✅ Loaded {len(df)} files from SAMPLE dataset")
    return df
'''
    
    with open('src/sample_dataset_loader.py', 'w') as f:
        f.write(loader_code)
    
    print("\n✅ Created sample dataset loader: src/sample_dataset_loader.py")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎙️  SPEECH EMOTION RECOGNITION - DATASET DOWNLOADER")
    print("="*70)
    
    download_tess_dataset()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Verify dataset: python demo.py")
    print("2. Extract features: See QUICKSTART.md")
    print("3. Train model: python src/train.py")
    print("="*70 + "\n")
