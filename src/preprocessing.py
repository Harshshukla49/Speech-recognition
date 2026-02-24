"""
Audio preprocessing module for Speech Emotion Recognition
"""
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import config


def load_audio(file_path, sr=None, duration=None):
    """
    Load audio file
    
    Args:
        file_path: Path to audio file
        sr: Sample rate (default: config.SAMPLE_RATE)
        duration: Duration to load in seconds
    
    Returns:
        audio: Audio time series
        sr: Sample rate
    """
    if sr is None:
        sr = config.SAMPLE_RATE
    
    try:
        audio, sr = librosa.load(file_path, sr=sr, duration=duration)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def normalize_audio(audio):
    """
    Normalize audio to [-1, 1] range
    
    Args:
        audio: Audio time series
    
    Returns:
        Normalized audio
    """
    if len(audio) == 0:
        return audio
    
    # L2 normalization
    norm = np.linalg.norm(audio)
    if norm > 0:
        audio = audio / norm
    
    return audio


def remove_silence(audio, sr, top_db=20):
    """
    Remove silence from audio
    
    Args:
        audio: Audio time series
        sr: Sample rate
        top_db: Threshold in decibels below reference
    
    Returns:
        Audio with silence removed
    """
    # Trim silence from beginning and end
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return audio_trimmed


def add_noise(audio, noise_factor=0.005):
    """
    Add random noise to audio for data augmentation
    
    Args:
        audio: Audio time series
        noise_factor: Amount of noise to add
    
    Returns:
        Audio with added noise
    """
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio


def shift_audio(audio, shift_max=0.2):
    """
    Shift audio left or right for data augmentation
    
    Args:
        audio: Audio time series
        shift_max: Maximum shift amount (fraction of total length)
    
    Returns:
        Shifted audio
    """
    shift = np.random.randint(int(len(audio) * shift_max))
    direction = np.random.choice([-1, 1])
    augmented_audio = np.roll(audio, shift * direction)
    return augmented_audio


def change_pitch(audio, sr, n_steps=2):
    """
    Change pitch of audio for data augmentation
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_steps: Number of semitones to shift pitch
    
    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def change_speed(audio, speed_factor=1.0):
    """
    Change speed of audio for data augmentation
    
    Args:
        audio: Audio time series
        speed_factor: Speed factor (>1 = faster, <1 = slower)
    
    Returns:
        Speed-changed audio
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def pad_audio(audio, max_length):
    """
    Pad audio to fixed length
    
    Args:
        audio: Audio time series
        max_length: Target length
    
    Returns:
        Padded audio
    """
    if len(audio) >= max_length:
        return audio[:max_length]
    else:
        padding = max_length - len(audio)
        return np.pad(audio, (0, padding), mode='constant')


def preprocess_audio(file_path, augment=False):
    """
    Complete preprocessing pipeline for audio file
    
    Args:
        file_path: Path to audio file
        augment: Whether to apply data augmentation
    
    Returns:
        Preprocessed audio
    """
    # Load audio
    audio, sr = load_audio(file_path, sr=config.SAMPLE_RATE, duration=config.DURATION)
    
    if audio is None:
        return None
    
    # Remove silence
    audio = remove_silence(audio, sr)
    
    # Apply augmentation if requested
    if augment:
        # Randomly apply augmentation techniques
        if np.random.random() > 0.5:
            audio = add_noise(audio)
        if np.random.random() > 0.5:
            audio = shift_audio(audio)
        if np.random.random() > 0.5:
            audio = change_pitch(audio, sr, n_steps=np.random.randint(-2, 3))
    
    # Normalize
    audio = normalize_audio(audio)
    
    # Pad to fixed length
    max_length = config.SAMPLE_RATE * config.DURATION
    audio = pad_audio(audio, max_length)
    
    return audio


def apply_bandpass_filter(audio, sr, lowcut=300, highcut=3000):
    """
    Apply bandpass filter to focus on speech frequencies
    
    Args:
        audio: Audio time series
        sr: Sample rate
        lowcut: Lower frequency bound (Hz)
        highcut: Upper frequency bound (Hz)
    
    Returns:
        Filtered audio
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = signal.butter(5, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def save_audio(audio, sr, output_path):
    """
    Save audio to file
    
    Args:
        audio: Audio time series
        sr: Sample rate
        output_path: Path to save audio
    """
    sf.write(output_path, audio, sr)
    print(f"✅ Audio saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Audio Preprocessing Module")
    print("=" * 50)
    
    # Test preprocessing on a sample file
    # Uncomment and modify the path to test
    # audio = preprocess_audio("path/to/audio.wav", augment=True)
    # print(f"Preprocessed audio shape: {audio.shape}")
