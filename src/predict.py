"""
Prediction script for Speech Emotion Recognition
"""
import os
import argparse
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

import config
from feature_extraction import extract_features_from_file
from utils import get_emotion_color


class EmotionPredictor:
    """Emotion Predictor class"""
    
    def __init__(self, model_path=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
        """
        if model_path is None:
            model_path = config.MODEL_SAVE_PATH
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        print("✅ Model loaded successfully!")
    
    def predict(self, audio_path, return_probabilities=False):
        """
        Predict emotion from audio file
        
        Args:
            audio_path: Path to audio file
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Predicted emotion (and probabilities if requested)
        """
        # Extract features
        print(f"Processing: {audio_path}")
        features = extract_features_from_file(audio_path)
        
        if features is None:
            print("❌ Error extracting features")
            return None
        
        # Expand dimensions for batch
        features = np.expand_dims(features, axis=0)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_emotion = config.EMOTIONS[predicted_class]
        
        if return_probabilities:
            probabilities = {
                config.EMOTIONS[i]: float(predictions[0][i])
                for i in range(config.NUM_CLASSES)
            }
            return predicted_emotion, probabilities
        
        return predicted_emotion
    
    def predict_batch(self, audio_paths):
        """
        Predict emotions for multiple audio files
        
        Args:
            audio_paths: List of audio file paths
        
        Returns:
            List of predictions
        """
        predictions = []
        
        for audio_path in audio_paths:
            try:
                emotion = self.predict(audio_path)
                predictions.append({
                    'file': audio_path,
                    'emotion': emotion
                })
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                predictions.append({
                    'file': audio_path,
                    'emotion': 'error'
                })
        
        return predictions
    
    def visualize_prediction(self, audio_path, save_path=None):
        """
        Visualize prediction with probabilities
        
        Args:
            audio_path: Path to audio file
            save_path: Path to save visualization
        """
        emotion, probabilities = self.predict(audio_path, return_probabilities=True)
        
        # Sort probabilities
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        emotions_sorted = [x[0] for x in sorted_probs]
        probs_sorted = [x[1] for x in sorted_probs]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        colors = [get_emotion_color(e) for e in emotions_sorted]
        bars = ax.barh(emotions_sorted, probs_sorted, color=colors, alpha=0.8)
        
        # Add percentage labels
        for i, (emotion_name, prob) in enumerate(sorted_probs):
            ax.text(prob + 0.01, i, f'{prob*100:.1f}%', va='center')
        
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_ylabel('Emotion', fontsize=12)
        ax.set_title(f'Emotion Prediction\nPredicted: {emotion.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ Visualization saved to: {save_path}")
        else:
            plt.show()
        
        return fig


def predict_from_audio(audio_path, model_path=None, visualize=False):
    """
    Predict emotion from audio file
    
    Args:
        audio_path: Path to audio file
        model_path: Path to trained model
        visualize: Whether to visualize prediction
    
    Returns:
        Predicted emotion
    """
    # Create predictor
    predictor = EmotionPredictor(model_path)
    
    # Make prediction
    emotion, probabilities = predictor.predict(audio_path, return_probabilities=True)
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Audio File: {audio_path}")
    print(f"Predicted Emotion: {emotion.upper()}")
    print("\nClass Probabilities:")
    print("-" * 50)
    
    # Sort and display probabilities
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for emotion_name, prob in sorted_probs:
        bar = '█' * int(prob * 40)
        print(f"{emotion_name:12s}: {bar:40s} {prob*100:5.2f}%")
    
    print("="*50 + "\n")
    
    # Visualize if requested
    if visualize:
        predictor.visualize_prediction(audio_path)
    
    return emotion


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Predict Emotion from Speech')
    parser.add_argument('--audio_path', type=str, required=True,
                       help='Path to audio file')
    parser.add_argument('--model_path', type=str, default=config.MODEL_SAVE_PATH,
                       help='Path to trained model')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize prediction probabilities')
    parser.add_argument('--save_viz', type=str, default=None,
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("SPEECH EMOTION RECOGNITION - PREDICTION")
    print("="*50)
    
    # Check if file exists
    if not os.path.exists(args.audio_path):
        print(f"❌ Audio file not found: {args.audio_path}")
        return
    
    # Create predictor
    predictor = EmotionPredictor(args.model_path)
    
    # Make prediction
    emotion, probabilities = predictor.predict(args.audio_path, return_probabilities=True)
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Audio File: {args.audio_path}")
    print(f"Predicted Emotion: {emotion.upper()}")
    print("\nClass Probabilities:")
    print("-" * 50)
    
    # Sort and display probabilities
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for emotion_name, prob in sorted_probs:
        bar = '█' * int(prob * 40)
        print(f"{emotion_name:12s}: {bar:40s} {prob*100:5.2f}%")
    
    print("="*50 + "\n")
    
    # Visualize if requested
    if args.visualize or args.save_viz:
        predictor.visualize_prediction(args.audio_path, save_path=args.save_viz)


if __name__ == "__main__":
    main()
