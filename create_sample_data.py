"""
Quick Sample Dataset Generator
Creates synthetic audio samples for testing
"""
import os
import numpy as np
import soundfile as sf

print("="*70)
print("🎨 CREATING SAMPLE DATASET FOR TESTING")
print("="*70)

# Create sample directory
sample_dir = 'data/raw/SAMPLE'
os.makedirs(sample_dir, exist_ok=True)

# Emotion configurations
emotions = {
    'neutral': {'freq': 250, 'amplitude': 0.3},
    'happy': {'freq': 500, 'amplitude': 0.5},
    'sad': {'freq': 200, 'amplitude': 0.2},
    'angry': {'freq': 400, 'amplitude': 0.7},
    'fear': {'freq': 450, 'amplitude': 0.6},
    'disgust': {'freq': 325, 'amplitude': 0.4},
    'surprise': {'freq': 550, 'amplitude': 0.6}
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
        freq_variation = np.random.uniform(0.8, 1.2)
        freq = config['freq'] * freq_variation
        
        # Create base signal
        audio = config['amplitude'] * np.sin(2 * np.pi * freq * t)
        
        # Add harmonics for richness
        audio += 0.3 * config['amplitude'] * np.sin(2 * np.pi * freq * 2 * t)
        audio += 0.2 * config['amplitude'] * np.sin(2 * np.pi * freq * 3 * t)
        
        # Add noise
        noise = np.random.normal(0, 0.05, len(audio))
        audio += noise
        
        # Apply envelope (fade in/out)
        fade_samples = int(0.1 * sr)
        envelope = np.ones_like(audio)
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

print(f"\n{'='*70}")
print(f"✅ SAMPLE DATASET CREATED SUCCESSFULLY!")
print(f"{'='*70}")
print(f"Total files: {total_files}")
print(f"Location: {sample_dir}")
print(f"\n⚠️  Note: This is a SYNTHETIC dataset for TESTING only.")
print("For real applications, download actual speech emotion datasets.")
print(f"{'='*70}\n")
