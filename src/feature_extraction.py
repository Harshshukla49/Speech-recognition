"""
Feature extraction module for Speech Emotion Recognition
"""
import numpy as np
import librosa
import config


def extract_mfcc(audio, sr):
    """
    Extract MFCC features from audio
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        MFCC features
    """
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    return np.mean(mfccs.T, axis=0)


def extract_mel_spectrogram(audio, sr):
    """
    Extract Mel-Spectrogram features from audio
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Mel-Spectrogram features
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_chroma(audio, sr):
    """
    Extract Chroma features from audio
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Chroma features
    """
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    return np.mean(chroma.T, axis=0)


def extract_zero_crossing_rate(audio):
    """
    Extract Zero Crossing Rate from audio
    
    Args:
        audio: Audio time series
    
    Returns:
        Zero Crossing Rate
    """
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=config.HOP_LENGTH)
    return np.mean(zcr)


def extract_spectral_centroid(audio, sr):
    """
    Extract Spectral Centroid from audio
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Spectral Centroid
    """
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    return np.mean(spectral_centroid)


def extract_spectral_rolloff(audio, sr):
    """
    Extract Spectral Rolloff from audio
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Spectral Rolloff
    """
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    return np.mean(spectral_rolloff)


def extract_rms(audio):
    """
    Extract RMS (Root Mean Square) energy from audio
    
    Args:
        audio: Audio time series
    
    Returns:
        RMS energy
    """
    rms = librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH)
    return np.mean(rms)


def extract_all_features(audio, sr):
    """
    Extract all features from audio
    
    Args:
        audio: Audio time series
        sr: Sample rate
    
    Returns:
        Dictionary of all features
    """
    features = {}
    
    # Extract MFCC
    features['mfcc'] = extract_mfcc(audio, sr)
    
    # Extract Chroma
    features['chroma'] = extract_chroma(audio, sr)
    
    # Extract other features
    features['zcr'] = extract_zero_crossing_rate(audio)
    features['spectral_centroid'] = extract_spectral_centroid(audio, sr)
    features['spectral_rolloff'] = extract_spectral_rolloff(audio, sr)
    features['rms'] = extract_rms(audio)
    
    # Combine all features
    combined_features = np.hstack([
        features['mfcc'],
        features['chroma'],
        [features['zcr']],
        [features['spectral_centroid']],
        [features['spectral_rolloff']],
        [features['rms']]
    ])
    
    return combined_features


def extract_mel_spectrogram_for_cnn(audio, sr, n_mels=128, max_len=128):
    """
    Extract Mel-Spectrogram for CNN input
    
    Args:
        audio: Audio time series
        sr: Sample rate
        n_mels: Number of mel bands
        max_len: Maximum length of time axis
    
    Returns:
        Mel-Spectrogram resized to (n_mels, max_len)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize to fixed dimensions
    if mel_spec_db.shape[1] < max_len:
        # Pad if too short
        pad_width = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if too long
        mel_spec_db = mel_spec_db[:, :max_len]
    
    return mel_spec_db


def prepare_features_for_cnn(mel_spec):
    """
    Prepare features for CNN model
    
    Args:
        mel_spec: Mel-Spectrogram
    
    Returns:
        Reshaped features for CNN input (height, width, channels)
    """
    # Normalize
    mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-8)
    
    # Add channel dimension
    mel_spec = mel_spec.reshape(mel_spec.shape[0], mel_spec.shape[1], 1)
    
    return mel_spec


def extract_features_from_file(file_path, feature_type='mel_spectrogram'):
    """
    Extract features from audio file
    
    Args:
        file_path: Path to audio file
        feature_type: Type of features to extract ('mel_spectrogram' or 'combined')
    
    Returns:
        Extracted features
    """
    from preprocessing import load_audio, preprocess_audio
    
    # Preprocess audio
    audio = preprocess_audio(file_path)
    
    if audio is None:
        return None
    
    sr = config.SAMPLE_RATE
    
    if feature_type == 'mel_spectrogram':
        # Extract mel-spectrogram for CNN
        mel_spec = extract_mel_spectrogram_for_cnn(audio, sr)
        features = prepare_features_for_cnn(mel_spec)
    else:
        # Extract combined features
        features = extract_all_features(audio, sr)
    
    return features


def batch_extract_features(file_paths, labels, feature_type='mel_spectrogram', verbose=True):
    """
    Extract features from multiple audio files
    
    Args:
        file_paths: List of file paths
        labels: List of corresponding labels
        feature_type: Type of features to extract
        verbose: Whether to print progress
    
    Returns:
        Features array, Labels array
    """
    from tqdm import tqdm
    
    features_list = []
    valid_labels = []
    
    iterator = tqdm(zip(file_paths, labels), total=len(file_paths)) if verbose else zip(file_paths, labels)
    
    for file_path, label in iterator:
        try:
            features = extract_features_from_file(file_path, feature_type)
            if features is not None:
                features_list.append(features)
                valid_labels.append(label)
        except Exception as e:
            if verbose:
                print(f"Error processing {file_path}: {e}")
            continue
    
    features_array = np.array(features_list)
    labels_array = np.array(valid_labels)
    
    if verbose:
        print(f"\n✅ Extracted features from {len(features_array)} files")
        print(f"Features shape: {features_array.shape}")
        print(f"Labels shape: {labels_array.shape}")
    
    return features_array, labels_array


if __name__ == "__main__":
    # Example usage
    print("Feature Extraction Module")
    print("=" * 50)
    
    # Test feature extraction on a sample file
    # Uncomment and modify the path to test
    # features = extract_features_from_file("path/to/audio.wav")
    # print(f"Extracted features shape: {features.shape}")
