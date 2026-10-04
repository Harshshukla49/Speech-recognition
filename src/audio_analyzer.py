"""
Audio Analyzer module for Speech Emotion Recognition
Provides advanced signal extraction, acoustic metrics, MFCC matrices, and segment-wise emotion timeline analysis.
"""
import numpy as np
import librosa
import soundfile as sf
import config


def compute_acoustic_features(audio, sr):
    """
    Compute rich acoustic descriptors from raw audio time series
    
    Args:
        audio: 1D numpy array of audio samples
        sr: Sampling rate in Hz
        
    Returns:
        dict containing calculated acoustic parameters
    """
    duration = float(len(audio) / sr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH)[0]
    mean_rms = float(np.mean(rms))
    max_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=config.HOP_LENGTH)[0]
    mean_zcr = float(np.mean(zcr))
    
    # Spectral Centroid & Rolloff
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=config.HOP_LENGTH)[0]
    mean_centroid = float(np.mean(spectral_centroid))
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=config.HOP_LENGTH)[0]
    mean_rolloff = float(np.mean(spectral_rolloff))
    
    # Silence Percentage (< -40 dB threshold)
    db_energy = librosa.amplitude_to_db(rms, ref=np.max)
    silence_frames = np.sum(db_energy < -35)
    silence_pct = float((silence_frames / len(db_energy)) * 100) if len(db_energy) > 0 else 0.0
    
    # Pitch / Fundamental Frequency (F0) Estimation via YIN algorithm
    try:
        f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr, hop_length=config.HOP_LENGTH)
        valid_f0 = f0[~np.isnan(f0)]
        mean_pitch = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
        pitch_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 0.0
    except Exception:
        mean_pitch = 0.0
        pitch_std = 0.0
        
    return {
        'duration_sec': round(duration, 2),
        'sample_rate': sr,
        'mean_rms': round(mean_rms, 4),
        'max_rms': round(max_rms, 4),
        'mean_zcr': round(mean_zcr, 4),
        'mean_spectral_centroid_hz': round(mean_centroid, 1),
        'mean_spectral_rolloff_hz': round(mean_rolloff, 1),
        'silence_percentage': round(silence_pct, 1),
        'estimated_mean_pitch_hz': round(mean_pitch, 1),
        'pitch_variation_hz': round(pitch_std, 1),
        'num_samples': len(audio)
    }


def extract_mfcc_matrix(audio, sr, n_mfcc=40):
    """
    Extract 2D MFCC matrix across time for heatmap visualization
    
    Args:
        audio: 1D numpy array
        sr: Sampling rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        2D numpy array of shape (n_mfcc, time_frames)
    """
    mfcc_mat = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH
    )
    return mfcc_mat


def segment_and_predict(file_path, predictor, segment_duration=3.0, hop_duration=1.5):
    """
    Segment audio file into overlapping windows and generate an emotion timeline
    
    Args:
        file_path: Path to audio file
        predictor: EmotionPredictor instance
        segment_duration: Duration of each analysis segment in seconds
        hop_duration: Step between consecutive segments
        
    Returns:
        list of segment dictionaries with timestamps and predicted emotions
    """
    try:
        audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio for segmentation: {e}")
        return []
        
    total_len_sec = len(audio) / sr
    
    # If audio is shorter than or equal to segment duration, single segment
    if total_len_sec <= segment_duration:
        emotion, probs = predictor.predict(file_path, return_probabilities=True)
        return [{
            'segment_index': 0,
            'start_time': 0.0,
            'end_time': round(total_len_sec, 2),
            'predicted_emotion': emotion,
            'confidence': float(probs[emotion]),
            'probabilities': probs
        }]
        
    segment_samples = int(segment_duration * sr)
    hop_samples = int(hop_duration * sr)
    
    segments = []
    seg_idx = 0
    
    start_samp = 0
    import tempfile
    
    while start_samp < len(audio):
        end_samp = min(start_samp + segment_samples, len(audio))
        chunk = audio[start_samp:end_samp]
        
        # If chunk is at least 0.8 seconds, analyze it
        if len(chunk) >= int(0.8 * sr):
            # Save chunk to temporary wav
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_chunk:
                tmp_chunk_path = tmp_chunk.name
            
            sf.write(tmp_chunk_path, chunk, sr)
            
            try:
                emotion, probs = predictor.predict(tmp_chunk_path, return_probabilities=True)
                start_t = round(start_samp / sr, 2)
                end_t = round(end_samp / sr, 2)
                
                segments.append({
                    'segment_index': seg_idx,
                    'start_time': start_t,
                    'end_time': end_t,
                    'predicted_emotion': emotion,
                    'confidence': float(probs[emotion]),
                    'probabilities': probs
                })
                seg_idx += 1
            except Exception as seg_e:
                print(f"Error analyzing segment {seg_idx}: {seg_e}")
            finally:
                if os.path.exists(tmp_chunk_path):
                    try:
                        os.remove(tmp_chunk_path)
                    except Exception:
                        pass
                        
        if end_samp >= len(audio):
            break
        start_samp += hop_samples
        
    return segments
