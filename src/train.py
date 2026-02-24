"""
Training script for Speech Emotion Recognition
"""
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.utils import to_categorical
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt

import config
from model import get_model
from utils import (
    load_features, plot_training_history,
    plot_confusion_matrix, print_evaluation_metrics
)


def prepare_data(features, labels, validation_split=None, test_split=None):
    """
    Prepare data for training
    
    Args:
        features: Feature array
        labels: Label array
        validation_split: Validation split ratio
        test_split: Test split ratio
    
    Returns:
        Train, validation, and test datasets
    """
    if validation_split is None:
        validation_split = config.VALIDATION_SPLIT
    if test_split is None:
        test_split = config.TEST_SPLIT
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels,
        test_size=test_split,
        random_state=config.RANDOM_SEED,
        stratify=labels
    )
    
    # Split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=validation_split / (1 - test_split),
        random_state=config.RANDOM_SEED,
        stratify=y_train_val
    )
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=config.NUM_CLASSES)
    y_val_cat = to_categorical(y_val, num_classes=config.NUM_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=config.NUM_CLASSES)
    
    print("\n" + "="*50)
    print("DATA SPLIT SUMMARY")
    print("="*50)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature shape: {X_train.shape[1:]}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print("="*50 + "\n")
    
    return (X_train, y_train, y_train_cat), (X_val, y_val, y_val_cat), (X_test, y_test, y_test_cat)


def get_callbacks(model_path=None, checkpoint_path=None):
    """
    Get training callbacks
    
    Args:
        model_path: Path to save best model
        checkpoint_path: Path to save checkpoints
    
    Returns:
        List of callbacks
    """
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, 'checkpoint_{epoch:02d}_{val_accuracy:.2f}.h5')
    
    callbacks = [
        # Model checkpoint
        ModelCheckpoint(
            filepath=model_path,
            monitor=config.CHECKPOINT_MONITOR,
            mode=config.CHECKPOINT_MODE,
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=os.path.join(config.CHECKPOINTS_DIR, 'logs'),
            histogram_freq=1
        )
    ]
    
    return callbacks


def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y_train: Training labels
    
    Returns:
        Class weight dictionary
    """
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n" + "="*50)
    print("CLASS WEIGHTS")
    print("="*50)
    for emotion_id, weight in class_weight_dict.items():
        emotion_name = config.EMOTIONS[emotion_id]
        print(f"{emotion_name}: {weight:.2f}")
    print("="*50 + "\n")
    
    return class_weight_dict


def train_model(model_type='cnn_lstm', epochs=None, batch_size=None):
    """
    Train the emotion recognition model
    
    Args:
        model_type: Type of model to train
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        Trained model and history
    """
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    
    # Load features
    features, labels = load_features()
    
    # Prepare data
    (X_train, y_train, y_train_cat), (X_val, y_val, y_val_cat), (X_test, y_test, y_test_cat) = prepare_data(
        features, labels
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    
    print("\n" + "="*50)
    print("CREATING MODEL")
    print("="*50)
    
    # Create model
    model = get_model(model_type, input_shape=X_train.shape[1:])
    model.summary()
    
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train model
    history = model.fit(
        X_train, y_train_cat,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Generate predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Print detailed metrics
    print_evaluation_metrics(y_test, y_pred)
    
    # Plot confusion matrix
    plt_cm = plot_confusion_matrix(y_test, y_pred)
    plt_cm.savefig(os.path.join(config.MODELS_DIR, 'confusion_matrix.png'))
    print(f"✅ Confusion matrix saved to: {os.path.join(config.MODELS_DIR, 'confusion_matrix.png')}")
    
    # Plot training history
    plt_history = plot_training_history(history)
    plt_history.savefig(os.path.join(config.MODELS_DIR, 'training_history.png'))
    print(f"✅ Training history saved to: {os.path.join(config.MODELS_DIR, 'training_history.png')}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"✅ Best model saved to: {config.MODEL_SAVE_PATH}")
    
    return model, history


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Model')
    parser.add_argument('--model', type=str, default='cnn_lstm',
                       choices=['cnn', 'lstm', 'cnn_lstm', 'attention_cnn_lstm'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("SPEECH EMOTION RECOGNITION - TRAINING")
    print("="*50)
    print(f"Model Type: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("="*50)
    
    # Train model
    model, history = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
