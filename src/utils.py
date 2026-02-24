"""
Utility functions for the Speech Emotion Recognition project
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import config


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.CHECKPOINTS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ All directories created successfully!")


def plot_waveform(audio, sr, title="Waveform"):
    """Plot audio waveform"""
    plt.figure(figsize=(12, 4))
    plt.plot(np.linspace(0, len(audio) / sr, len(audio)), audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    return plt


def plot_spectrogram(spectrogram, sr, hop_length, title="Spectrogram"):
    """Plot spectrogram"""
    import librosa.display
    
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length,
                            x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    return plt


def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False):
    """Plot confusion matrix"""
    if classes is None:
        classes = list(config.EMOTIONS.values())
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt


def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return plt


def print_evaluation_metrics(y_true, y_pred):
    """Print detailed evaluation metrics"""
    classes = list(config.EMOTIONS.values())
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=classes))
    
    # Calculate overall accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}\n")


def save_features(features, labels, features_path=None, labels_path=None):
    """Save extracted features and labels"""
    if features_path is None:
        features_path = config.FEATURES_FILE
    if labels_path is None:
        labels_path = config.LABELS_FILE
    
    np.save(features_path, features)
    np.save(labels_path, labels)
    print(f"✅ Features saved to: {features_path}")
    print(f"✅ Labels saved to: {labels_path}")


def load_features(features_path=None, labels_path=None):
    """Load extracted features and labels"""
    if features_path is None:
        features_path = config.FEATURES_FILE
    if labels_path is None:
        labels_path = config.LABELS_FILE
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    print(f"✅ Features loaded from: {features_path}")
    print(f"✅ Labels loaded from: {labels_path}")
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    return features, labels


def get_emotion_color(emotion):
    """Get color for each emotion for visualization"""
    colors = {
        'neutral': '#808080',
        'happy': '#FFD700',
        'sad': '#4169E1',
        'angry': '#DC143C',
        'fear': '#9370DB',
        'disgust': '#228B22',
        'surprise': '#FF69B4'
    }
    return colors.get(emotion, '#000000')
