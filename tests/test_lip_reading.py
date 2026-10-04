"""
Comprehensive Test Suite for AI Lip-Reading and Hinglish Video-to-Text Conversion System.
Tests Hinglish NLP engine, visual viseme decoding, audio-visual alignment, and subtitle exporters.
"""
import os
import sys
import numpy as np

# Ensure src and project root are on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hinglish_engine import HinglishEngine
from src.visual_speech_recognizer import VisualSpeechRecognizerAdapter
from src.audio_transcriber import AudioTranscriber, AlignedTranscriptSegment
from src.subtitle_exporter import SubtitleExporter
from src.lip_sync_detector import LipTrackingPoint


def test_hinglish_engine_core_examples():
    """Test standard benchmark sentences required in specifications."""
    # 1. Hindi question
    s1 = "आप कैसे हो? क्या कर रहे हो?"
    h1 = HinglishEngine.convert_to_hinglish(s1)
    assert "Aap" in h1
    assert "kaise" in h1.lower()
    assert "kya" in h1.lower()
    assert "ho" in h1.lower()

    # 2. Hindi sentence
    s2 = "मुझे यह प्रोजेक्ट बहुत अच्छा लगा।"
    h2 = HinglishEngine.convert_to_hinglish(s2)
    assert "Mujhe" in h2 or "mujhe" in h2.lower()
    assert "project" in h2.lower() or "projekat" in h2.lower()
    assert "achha" in h2.lower() or "laga" in h2.lower()

    # 3. Pure English sentence (preserved intact)
    s3 = "Today we are going to discuss artificial intelligence."
    h3 = HinglishEngine.convert_to_hinglish(s3)
    assert h3 == s3

    # 4. Mixed Hindi-English code-switched sentence
    s4 = "आज हम machine learning के बारे में सीखेंगे।"
    h4 = HinglishEngine.convert_to_hinglish(s4)
    assert "machine learning" in h4
    assert "Aaj" in h4 or "aaj" in h4.lower()
    assert "seekhenge" in h4.lower() or "baare" in h4.lower()


def test_hinglish_multilingual_outputs():
    """Test generating Hinglish, Devanagari, and English translation dictionaries."""
    raw = "नमस्ते आप कैसे हो?"
    res = HinglishEngine.generate_multilingual_outputs(raw)
    assert 'hinglish' in res
    assert 'devanagari' in res
    assert 'english_translation' in res
    assert 'Namaste' in res['hinglish']
    assert res['devanagari'] == raw


def test_visual_speech_recognizer_visemes_and_decoding():
    """Test visual viseme classification and speaker tracking."""
    adapter = VisualSpeechRecognizerAdapter()
    
    # Test viseme classifications
    # Open mouth -> V5
    assert adapter.classify_frame_viseme(0.55, 0.2, np.zeros((88, 88))) == 'V5'
    # Closed mouth -> V0
    assert adapter.classify_frame_viseme(0.04, 0.1, np.zeros((88, 88))) == 'V0'

    # Test multi-speaker detection on dummy image
    img = np.full((300, 400, 3), 40, dtype=np.uint8)
    speakers = adapter.detect_visible_speakers(img)
    assert len(speakers) >= 1
    assert hasattr(speakers[0], 'speaker_id')
    assert hasattr(speakers[0], 'mouth_box')

    # Test visual-only decoding
    dummy_frames = [{'timestamp': i * 0.04, 'frame_idx': i, 'image': img} for i in range(25)]
    dummy_points = [
        LipTrackingPoint(
            timestamp=i * 0.04, frame_idx=i, mouth_aspect_ratio=0.35 if 5 <= i <= 20 else 0.05,
            lip_velocity=0.6 if 5 <= i <= 20 else 0.0, is_speaking=(5 <= i <= 20),
            face_detected=True, tracking_confidence=0.9, mouth_box=(100, 100, 40, 30)
        ) for i in range(25)
    ]

    decoded = adapter.decode_visual_speech(dummy_frames, dummy_points, selected_speaker_idx=1)
    assert 'transcript' in decoded
    assert 'segments' in decoded
    assert decoded['modality'] == 'Visual Lip-Reading (Visual Only)'


def test_audio_transcriber_multimodal_alignment():
    """Test multimodal reconciliation between visual segments and audio transcript."""
    transcriber = AudioTranscriber()
    
    vis_results = {
        'transcript': 'aap kaise ho',
        'segments': [
            {'word': 'aap', 'start_time': 0.0, 'end_time': 0.8, 'confidence': 0.88, 'viseme_sequence': 'V5-V0'},
            {'word': 'kaise ho', 'start_time': 0.9, 'end_time': 2.0, 'confidence': 0.90, 'viseme_sequence': 'V4-V7-V2'}
        ],
        'mean_confidence': 0.89
    }

    audio_results = {
        'raw_transcript': 'आप कैसे हो',
        'detected_language': 'hi-IN',
        'confidence': 0.92,
        'is_success': True
    }

    aligned = transcriber.align_and_reconcile_multimodal(
        visual_results=vis_results,
        audio_results=audio_results,
        total_duration=2.5,
        mode="Audio + Lip Reading"
    )

    assert len(aligned) >= 1
    assert isinstance(aligned[0], AlignedTranscriptSegment)
    assert "Aap" in aligned[0].hinglish_text
    assert "Multimodal" in aligned[0].source_modality or "Audio" in aligned[0].source_modality


def test_subtitle_exporter_all_formats():
    """Test subtitle generation in SRT, VTT, TXT, CSV, and PDF formats."""
    sample_segments = [
        {
            'segment_id': 1,
            'start_time': 0.5,
            'end_time': 2.2,
            'hinglish_text': 'Aap kaise ho? Kya kar rahe ho?',
            'devanagari_text': 'आप कैसे हो? क्या कर रहे हो?',
            'english_translation': 'How are you? What are you doing?',
            'confidence': 0.94,
            'source_modality': 'Multimodal Fused (A+V)'
        },
        {
            'segment_id': 2,
            'start_time': 2.5,
            'end_time': 4.8,
            'hinglish_text': 'Mujhe yeh project bahut achha laga.',
            'devanagari_text': 'मुझे यह प्रोजेक्ट बहुत अच्छा लगा।',
            'english_translation': 'I really liked this project.',
            'confidence': 0.92,
            'source_modality': 'Visual Lip-Reading'
        }
    ]

    # SRT
    srt_out = SubtitleExporter.export_to_srt(sample_segments)
    assert "00:00:00,500 --> 00:00:02,200" in srt_out
    assert "Aap kaise ho?" in srt_out

    # VTT
    vtt_out = SubtitleExporter.export_to_vtt(sample_segments)
    assert "WEBVTT" in vtt_out
    assert "00:00:00.500 --> 00:00:02.200" in vtt_out

    # TXT
    txt_out = SubtitleExporter.export_to_txt(sample_segments)
    assert "[0.50s - 2.20s]" in txt_out

    # CSV
    csv_out = SubtitleExporter.export_to_csv(sample_segments)
    assert "segment_id,start_time,end_time" in csv_out
    assert "Aap kaise ho?" in csv_out

    # PDF
    pdf_buf = SubtitleExporter.export_to_pdf(
        segments=sample_segments,
        video_name="test_speech.mp4",
        total_duration=5.0,
        analysis_mode="Audio + Lip Reading",
        detected_emotion="happy"
    )
    pdf_bytes = pdf_buf.getvalue()
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")


if __name__ == '__main__':
    print("Running AI Lip-Reading & Hinglish Conversion Test Suite...")
    test_hinglish_engine_core_examples()
    print("✅ test_hinglish_engine_core_examples passed")
    test_hinglish_multilingual_outputs()
    print("✅ test_hinglish_multilingual_outputs passed")
    test_visual_speech_recognizer_visemes_and_decoding()
    print("✅ test_visual_speech_recognizer_visemes_and_decoding passed")
    test_audio_transcriber_multimodal_alignment()
    print("✅ test_audio_transcriber_multimodal_alignment passed")
    test_subtitle_exporter_all_formats()
    print("✅ test_subtitle_exporter_all_formats passed")
    print("\n🎉 ALL 5 LIP-READING & HINGLISH CONVERSION TESTS PASSED SUCCESSFULLY!")
