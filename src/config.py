"""
Configuration file for Speech Emotion Recognition
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

# Emotion Labels
EMOTIONS = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'angry',
    4: 'fear',
    5: 'disgust',
    6: 'surprise'
}

EMOTION_TO_IDX = {v: k for k, v in EMOTIONS.items()}

# Audio Parameters
SAMPLE_RATE = 22050
DURATION = 3  # seconds
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

# Model Parameters
INPUT_SHAPE = (128, 128, 1)
NUM_CLASSES = len(EMOTIONS)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Training Parameters
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
CHECKPOINT_MONITOR = 'val_accuracy'
CHECKPOINT_MODE = 'max'

# Dataset Paths (update these with your dataset paths)
RAVDESS_PATH = os.path.join(RAW_DATA_DIR, 'RAVDESS')
CREMA_D_PATH = os.path.join(RAW_DATA_DIR, 'CREMA-D')
EMO_DB_PATH = os.path.join(RAW_DATA_DIR, 'EMO-DB')

# Feature Extraction
FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, 'features.npy')
LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, 'labels.npy')

# Model Save Path
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'best_model.h5')
MODEL_WEIGHTS_PATH = os.path.join(MODELS_DIR, 'best_weights.h5')

# Random Seed
RANDOM_SEED = 42
