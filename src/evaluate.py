"""
Standalone Model Evaluation CLI Script
Runs full validation on test dataset and prints classification report & confusion matrix.
"""
import os
import sys

# Configure UTF-8 encoding for standard streams on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from evaluation import get_model_evaluation_metrics


def main():
    print("=" * 70)
    print("[*] SPEECH EMOTION RECOGNITION - MODEL EVALUATION SUITE")
    print("=" * 70)
    
    print("\n[+] Evaluating trained model on dataset...")
    metrics = get_model_evaluation_metrics()
    
    if metrics.get('status') == 'no_data':
        print(f"\n[-] {metrics.get('message')}")
        return
    elif metrics.get('status') == 'error':
        print(f"\n[-] Error evaluating model: {metrics.get('message')}")
        return
        
    print(f"\n[+] Total Samples Evaluated: {metrics['total_samples']}")
    print(f"[+] Overall Accuracy:        {metrics['accuracy']*100:.2f}%")
    print(f"[+] Macro Precision:         {metrics['macro_precision']*100:.2f}%")
    print(f"[+] Macro Recall:            {metrics['macro_recall']*100:.2f}%")
    print(f"[+] Macro F1-Score:          {metrics['macro_f1']*100:.2f}%")
    print(f"[+] Weighted F1-Score:       {metrics['weighted_f1']*100:.2f}%")
    
    print("\n" + "-" * 70)
    print("PER-CLASS EVALUATION METRICS:")
    print("-" * 70)
    print(f"{'Emotion':14s} | {'Precision':10s} | {'Recall':10s} | {'F1-Score':10s} | {'Support':8s}")
    print("-" * 70)
    for c_name, scores in metrics['per_class'].items():
        print(f"{c_name.capitalize():14s} | {scores['precision']*100:9.2f}% | {scores['recall']*100:9.2f}% | {scores['f1_score']*100:9.2f}% | {scores['support']:8d}")
    print("-" * 70)
    
    print("\n[+] Evaluation successfully completed!")


if __name__ == "__main__":
    main()
