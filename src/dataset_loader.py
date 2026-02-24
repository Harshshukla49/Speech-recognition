"""
Dataset loading utilities for Speech Emotion Recognition
"""
import os
import glob
import re
from pathlib import Path
import pandas as pd
import config


def load_ravdess(data_path=None):
    """
    Load RAVDESS dataset
    
    RAVDESS filename format:
    Modality-Vocal channel-Emotion-Emotional intensity-Statement-Repetition-Actor
    Example: 03-01-05-02-01-01-12.wav
    Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    
    Args:
        data_path: Path to RAVDESS dataset
    
    Returns:
        DataFrame with file paths and labels
    """
    if data_path is None:
        data_path = config.RAVDESS_PATH
    
    if not os.path.exists(data_path):
        print(f"⚠️  RAVDESS path not found: {data_path}")
        return pd.DataFrame()
    
    # RAVDESS emotion mapping
    emotion_map = {
        '01': 0,  # neutral
        '02': 0,  # neutral (calm)
        '03': 1,  # happy
        '04': 2,  # sad
        '05': 3,  # angry
        '06': 4,  # fear
        '07': 5,  # disgust
        '08': 6   # surprise
    }
    
    file_paths = []
    labels = []
    
    # Find all audio files
    audio_files = glob.glob(os.path.join(data_path, '**', '*.wav'), recursive=True)
    
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                file_paths.append(file_path)
                labels.append(emotion_map[emotion_code])
    
    df = pd.DataFrame({
        'file_path': file_paths,
        'label': labels,
        'emotion': [config.EMOTIONS[label] for label in labels],
        'dataset': 'RAVDESS'
    })
    
    print(f"✅ Loaded {len(df)} files from RAVDESS")
    return df


def load_crema_d(data_path=None):
    """
    Load CREMA-D dataset
    
    CREMA-D filename format:
    ActorID_Sentence_Emotion_Intensity.wav
    Example: 1001_IEO_ANG_HI.wav
    Emotions: ANG=angry, DIS=disgust, FEA=fear, HAP=happy, NEU=neutral, SAD=sad
    
    Args:
        data_path: Path to CREMA-D dataset
    
    Returns:
        DataFrame with file paths and labels
    """
    if data_path is None:
        data_path = config.CREMA_D_PATH
    
    if not os.path.exists(data_path):
        print(f"⚠️  CREMA-D path not found: {data_path}")
        return pd.DataFrame()
    
    # CREMA-D emotion mapping
    emotion_map = {
        'ANG': 3,  # angry
        'DIS': 5,  # disgust
        'FEA': 4,  # fear
        'HAP': 1,  # happy
        'NEU': 0,  # neutral
        'SAD': 2   # sad
    }
    
    file_paths = []
    labels = []
    
    # Find all audio files
    audio_files = glob.glob(os.path.join(data_path, '*.wav'))
    
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                file_paths.append(file_path)
                labels.append(emotion_map[emotion_code])
    
    df = pd.DataFrame({
        'file_path': file_paths,
        'label': labels,
        'emotion': [config.EMOTIONS[label] for label in labels],
        'dataset': 'CREMA-D'
    })
    
    print(f"✅ Loaded {len(df)} files from CREMA-D")
    return df


def load_emodb(data_path=None):
    """
    Load EMO-DB dataset
    
    EMO-DB filename format:
    PositionSpeakerEmotionTextNumber.wav
    Example: 03a01Fa.wav
    Emotions: W=angry, L=boredom, E=disgust, A=fear, F=happy, T=sad, N=neutral
    
    Args:
        data_path: Path to EMO-DB dataset
    
    Returns:
        DataFrame with file paths and labels
    """
    if data_path is None:
        data_path = config.EMO_DB_PATH
    
    if not os.path.exists(data_path):
        print(f"⚠️  EMO-DB path not found: {data_path}")
        return pd.DataFrame()
    
    # EMO-DB emotion mapping
    emotion_map = {
        'W': 3,  # angry (Wut)
        'L': 0,  # neutral (Langeweile/boredom)
        'E': 5,  # disgust (Ekel)
        'A': 4,  # fear (Angst)
        'F': 1,  # happy (Freude)
        'T': 2,  # sad (Trauer)
        'N': 0   # neutral
    }
    
    file_paths = []
    labels = []
    
    # Find all audio files
    audio_files = glob.glob(os.path.join(data_path, '*.wav'))
    
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        # Extract emotion code (6th character)
        if len(filename) >= 6:
            emotion_code = filename[5]
            if emotion_code in emotion_map:
                file_paths.append(file_path)
                labels.append(emotion_map[emotion_code])
    
    df = pd.DataFrame({
        'file_path': file_paths,
        'label': labels,
        'emotion': [config.EMOTIONS[label] for label in labels],
        'dataset': 'EMO-DB'
    })
    
    print(f"✅ Loaded {len(df)} files from EMO-DB")
    return df


def load_sample(data_path=None):
    """
    Load SAMPLE dataset (synthetic data for testing)
    
    Args:
        data_path: Path to SAMPLE dataset
    
    Returns:
        DataFrame with file paths and labels
    """
    if data_path is None:
        data_path = os.path.join(config.RAW_DATA_DIR, 'SAMPLE')
    
    if not os.path.exists(data_path):
        print(f"⚠️  SAMPLE path not found: {data_path}")
        return pd.DataFrame()
    
    # Emotion mapping
    emotion_map = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'fear': 4,
        'disgust': 5,
        'surprise': 6
    }
    
    file_paths = []
    labels = []
    
    # Find all audio files in emotion folders
    for emotion, label in emotion_map.items():
        emotion_path = os.path.join(data_path, emotion)
        if os.path.exists(emotion_path):
            audio_files = glob.glob(os.path.join(emotion_path, '*.wav'))
            file_paths.extend(audio_files)
            labels.extend([label] * len(audio_files))
    
    df = pd.DataFrame({
        'file_path': file_paths,
        'label': labels,
        'emotion': [config.EMOTIONS[label] for label in labels],
        'dataset': 'SAMPLE'
    })
    
    print(f"✅ Loaded {len(df)} files from SAMPLE dataset")
    return df


def load_all_datasets():
    """
    Load all available datasets
    
    Returns:
        Combined DataFrame with all datasets
    """
    print("\n" + "="*50)
    print("LOADING DATASETS")
    print("="*50)
    
    datasets = []
    
    # Load SAMPLE dataset first (synthetic for testing)
    df_sample = load_sample()
    if not df_sample.empty:
        datasets.append(df_sample)
    
    # Load RAVDESS
    df_ravdess = load_ravdess()
    if not df_ravdess.empty:
        datasets.append(df_ravdess)
    
    # Load CREMA-D
    df_crema = load_crema_d()
    if not df_crema.empty:
        datasets.append(df_crema)
    
    # Load EMO-DB
    df_emodb = load_emodb()
    if not df_emodb.empty:
        datasets.append(df_emodb)
    
    if not datasets:
        print("❌ No datasets found!")
        return pd.DataFrame()
    
    # Combine all datasets
    df_combined = pd.concat(datasets, ignore_index=True)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total files: {len(df_combined)}")
    print("\nFiles per dataset:")
    print(df_combined['dataset'].value_counts())
    print("\nFiles per emotion:")
    print(df_combined['emotion'].value_counts())
    print("="*50 + "\n")
    
    return df_combined


def save_dataset_info(df, output_path=None):
    """
    Save dataset information to CSV
    
    Args:
        df: DataFrame with dataset information
        output_path: Path to save CSV
    """
    if output_path is None:
        output_path = os.path.join(config.PROCESSED_DATA_DIR, 'dataset_info.csv')
    
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset info saved to: {output_path}")


if __name__ == "__main__":
    # Load all datasets
    df = load_all_datasets()
    
    if not df.empty:
        # Save dataset info
        save_dataset_info(df)
        
        # Display sample
        print("\nSample data:")
        print(df.head())
