"""
Comprehensive Test Suite for Speech Emotion Recognition Platform
Tests audio DSP feature extraction, SQLite database operations, model inference, and PDF report generation.
"""
import os
import sys
import unittest
import numpy as np
import io
import soundfile as sf
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import config
from src.database import (
    init_db, save_analysis, get_all_analyses, get_analysis_by_id,
    delete_analysis, get_summary_stats, export_history_dataframe
)
from src.audio_analyzer import compute_acoustic_features, extract_mfcc_matrix
from src.report_generator import generate_pdf_report
from src.predict import EmotionPredictor
from src.evaluation import get_model_evaluation_metrics, get_model_architecture_comparison


class TestSpeechEmotionPlatform(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Create sample audio fixture for testing"""
        cls.sr = 22050
        cls.duration = 3.0
        # Generate 3 seconds of 440 Hz sinusoidal audio
        t = np.linspace(0, cls.duration, int(cls.sr * cls.duration), endpoint=False)
        cls.test_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        # Save temporary wav
        cls.temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(cls.temp_wav.name, cls.test_audio, cls.sr)
        cls.temp_wav.close()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixture"""
        if os.path.exists(cls.temp_wav.name):
            os.remove(cls.temp_wav.name)

    def test_01_acoustic_features_computation(self):
        """Test acoustic signal metrics extraction"""
        metrics = compute_acoustic_features(self.test_audio, self.sr)
        self.assertIsInstance(metrics, dict)
        self.assertIn('duration_sec', metrics)
        self.assertIn('mean_rms', metrics)
        self.assertIn('mean_zcr', metrics)
        self.assertIn('estimated_mean_pitch_hz', metrics)
        self.assertAlmostEqual(metrics['duration_sec'], 3.0, places=1)
        self.assertGreater(metrics['mean_rms'], 0.0)

    def test_02_mfcc_matrix_extraction(self):
        """Test 2D MFCC matrix generation"""
        mfcc_mat = extract_mfcc_matrix(self.test_audio, self.sr, n_mfcc=40)
        self.assertEqual(mfcc_mat.shape[0], 40)
        self.assertGreater(mfcc_mat.shape[1], 0)

    def test_03_database_crud(self):
        """Test SQLite database initialization, insertion, retrieval, and deletion"""
        init_db()
        test_id = "test_uuid_12345"
        sample_probs = {e: 0.14 for e in config.EMOTIONS.values()}
        sample_metrics = {'duration_sec': 3.0, 'mean_rms': 0.35}
        
        # Save
        save_analysis(
            analysis_id=test_id,
            filename="unit_test_audio.wav",
            source_type="upload",
            duration=3.0,
            predicted_emotion="happy",
            confidence=0.88,
            is_low_confidence=False,
            probabilities=sample_probs,
            acoustic_metrics=sample_metrics,
            model_version="CNN-LSTM Hybrid v2.0",
            processing_time=0.045
        )
        
        # Retrieve
        record = get_analysis_by_id(test_id)
        self.assertIsNotNone(record)
        self.assertEqual(record['filename'], "unit_test_audio.wav")
        self.assertEqual(record['predicted_emotion'], "happy")
        self.assertAlmostEqual(record['confidence'], 0.88)
        
        # Summary stats
        stats = get_summary_stats()
        self.assertGreaterEqual(stats['total_analyses'], 1)
        
        # Export dataframe
        df = export_history_dataframe()
        self.assertFalse(df.empty)
        
        # Delete
        delete_analysis(test_id)
        deleted_record = get_analysis_by_id(test_id)
        self.assertIsNone(deleted_record)

    def test_04_pdf_report_generation(self):
        """Test automated ReportLab PDF report generation"""
        sample_record = {
            'id': 'test-report-id-999',
            'timestamp': '2026-10-03 12:00:00',
            'filename': 'test_sample.wav',
            'duration': 3.0,
            'predicted_emotion': 'happy',
            'confidence': 0.92,
            'is_low_confidence': False,
            'probabilities': {e: 0.14 for e in config.EMOTIONS.values()},
            'acoustic_metrics': {'mean_rms': 0.25, 'estimated_mean_pitch_hz': 220.0, 'silence_percentage': 5.0},
            'model_version': 'CNN-LSTM Hybrid v2.0',
            'processing_time': 0.05
        }
        
        pdf_buf = generate_pdf_report(
            analysis_record=sample_record,
            audio_time_series=self.test_audio,
            sample_rate=self.sr
        )
        pdf_bytes = pdf_buf.getvalue()
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b'%PDF'))

    def test_05_model_loading_and_prediction(self):
        """Test model loading and real neural feedforward inference"""
        if os.path.exists(config.MODEL_SAVE_PATH):
            predictor = EmotionPredictor(config.MODEL_SAVE_PATH)
            emotion, probs = predictor.predict(self.temp_wav.name, return_probabilities=True)
            self.assertIn(emotion, config.EMOTIONS.values())
            self.assertEqual(len(probs), config.NUM_CLASSES)
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

    def test_06_model_architecture_comparison(self):
        """Test architecture specs generator"""
        comps = get_model_architecture_comparison()
        self.assertEqual(len(comps), 3)
        self.assertIn('CNN-LSTM Hybrid', [c['name'].split(' (')[0] for c in comps])


if __name__ == '__main__':
    unittest.main()
