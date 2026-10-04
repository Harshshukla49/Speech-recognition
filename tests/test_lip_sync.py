"""
Unit and Integration Tests for AI Lip-Sync Detection, Visual Speech Analysis,
and Multimodal Emotion Fusion Platform.
"""
import os
import sys
import tempfile
import numpy as np
import cv2
import soundfile as sf
import av

# Ensure src and project root are on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.video_processor import VideoProcessor
from src.lip_sync_detector import LipSyncDetector, LipTrackingPoint
from src.visual_speech_recognizer import VisualSpeechRecognizerAdapter
from src.multimodal_fusion import MultimodalFusionEngine
from src.database import init_db, save_analysis, get_analysis_by_id, delete_analysis
from src.report_generator import generate_pdf_report


def create_synthetic_test_video(duration_sec=2.0, fps=25, sr=22050):
    """
    Helper function to construct a synthetic test video container with video frames
    and synchronized audio waveform.
    """
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, f"test_synth_video_{int(np.random.randint(10000, 99999))}.mp4")

    total_frames = int(duration_sec * fps)
    total_audio_samples = int(duration_sec * sr)
    width, height = 320, 240

    container = av.open(video_path, mode='w')
    v_stream = container.add_stream('libx264', rate=fps)
    v_stream.width = width
    v_stream.height = height
    v_stream.pix_fmt = 'yuv420p'

    a_stream = container.add_stream('aac', rate=sr)
    a_stream.layout = 'mono'

    # Generate synthetic audio tone (sine wave modulating in loudness)
    t_a = np.linspace(0, duration_sec, total_audio_samples, endpoint=False)
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t_a) * (1.0 + 0.5 * np.sin(2 * np.pi * 2 * t_a))).astype(np.float32)

    # Encode video frames (draw face and moving mouth)
    for i in range(total_frames):
        img = np.full((height, width, 3), 30, dtype=np.uint8)
        # Draw face circle
        cv2.circle(img, (160, 120), 60, (200, 180, 150), -1)
        # Draw eyes
        cv2.circle(img, (140, 100), 6, (40, 30, 20), -1)
        cv2.circle(img, (180, 100), 6, (40, 30, 20), -1)
        # Draw dynamic mouth opening (oscillating MAR)
        mouth_h = int(10 + 15 * abs(np.sin(2 * np.pi * 2 * (i / fps))))
        cv2.ellipse(img, (160, 150), (25, mouth_h), 0, 0, 360, (50, 30, 30), -1)

        frame = av.VideoFrame.from_ndarray(img, format='rgb24')
        for packet in v_stream.encode(frame):
            container.mux(packet)

    for packet in v_stream.encode():
        container.mux(packet)

    # Encode audio frames
    frame_size = 1024
    for start in range(0, total_audio_samples, frame_size):
        chunk = audio_data[start:start+frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        
        a_frame = av.AudioFrame(format='flt', layout='mono', samples=frame_size)
        a_frame.rate = sr
        a_frame.planes[0].update(chunk.tobytes())
        for packet in a_stream.encode(a_frame):
            container.mux(packet)

    for packet in a_stream.encode():
        container.mux(packet)

    container.close()
    return video_path


def test_video_processor_metadata_and_extraction():
    """Test video processor metadata, audio demuxing, and frame sampling."""
    video_path = create_synthetic_test_video(duration_sec=1.5, fps=25, sr=22050)
    try:
        proc = VideoProcessor(target_sample_rate=22050, max_fps=25)
        
        # Metadata check
        meta = proc.get_metadata(video_path)
        assert meta['is_valid'] is True
        assert meta['has_audio'] is True
        assert meta['width'] == 320
        assert meta['height'] == 240
        assert meta['duration_sec'] > 1.0

        # Audio Extraction
        audio_arr, sr = proc.extract_audio(video_path)
        assert sr == 22050
        assert isinstance(audio_arr, np.ndarray)
        assert len(audio_arr) > 0
        assert audio_arr.dtype == np.float32

        # Frame Extraction
        frames, summary = proc.extract_frames(video_path, target_fps=25.0)
        assert len(frames) > 10
        assert 'timestamp' in frames[0]
        assert 'image' in frames[0]
        assert frames[0]['image'].shape == (240, 320, 3)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def test_lip_sync_detector_tracking_and_cross_correlation():
    """Test lip movement tracking and cross-modal cross-correlation synchronization."""
    detector = LipSyncDetector(fps=25.0)
    
    # Generate synthetic sequence of frames
    frames = []
    fps = 25.0
    for i in range(50):
        img = np.full((200, 200, 3), 40, dtype=np.uint8)
        # Draw face
        cv2.circle(img, (100, 100), 50, (200, 180, 150), -1)
        mouth_h = int(6 + 10 * abs(np.sin(2 * np.pi * 1.5 * (i / fps))))
        cv2.ellipse(img, (100, 125), (20, mouth_h), 0, 0, 360, (40, 20, 20), -1)
        frames.append({
            'timestamp': i / fps,
            'frame_idx': i,
            'image': img
        })

    lip_points = detector.track_lip_movement(frames)
    assert len(lip_points) == 50
    assert isinstance(lip_points[0], LipTrackingPoint)
    assert lip_points[0].mouth_aspect_ratio >= 0.0

    # Test Cross-Correlation with synthetic audio
    sr = 22050
    dur = 50 / fps
    t_a = np.linspace(0, dur, int(dur * sr))
    audio_syn = (0.6 * np.sin(2 * np.pi * 300 * t_a) * (1.0 + 0.5 * np.sin(2 * np.pi * 1.5 * t_a))).astype(np.float32)

    sync_results = detector.analyze_synchronization(lip_points, audio_syn, sr)
    assert 'sync_quality_index' in sync_results
    assert 'time_offset_ms' in sync_results
    assert 'sync_status' in sync_results
    assert isinstance(sync_results['sync_quality_index'], float)
    assert len(sync_results['lags_ms']) > 0

    # Test Frame Annotation
    annotated = detector.annotate_frame(frames[0]['image'], lip_points[0], sync_status="In-Sync")
    assert annotated.shape == frames[0]['image'].shape


def test_visual_speech_recognizer_adapter_diagnostics():
    """Test VSR adapter tensor preprocessing and zero-hallucination diagnostics."""
    adapter = VisualSpeechRecognizerAdapter()
    specs = adapter.get_system_specifications()
    
    assert specs['module_name'] == 'Visual Speech Recognition (VSR / Lip-Reading)'
    assert len(specs['supported_architectures']) >= 2

    # Create dummy frame and point
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    p = LipTrackingPoint(
        timestamp=0.0, frame_idx=0, mouth_aspect_ratio=0.3,
        lip_velocity=0.1, is_speaking=True, face_detected=True,
        tracking_confidence=0.9, mouth_box=(20, 40, 40, 30)
    )

    rois = adapter.preprocess_mouth_rois([{'image': img}], [p])
    assert rois.shape == (1, 88, 88)
    assert rois.dtype == np.float32

    # Test transparent transcription output (zero hallucination when unweighted)
    out = adapter.transcribe_visual_speech([{'image': img}], [p])
    assert 'model_loaded' in out
    if not out['model_loaded']:
        assert out['transcript'] is None  # Never fakes text
        assert "AV-Hubert" in out['diagnostic_message'] or "pipeline is ready" in out['diagnostic_message']


def test_multimodal_emotion_fusion():
    """Test decision-level late fusion between acoustic model and visual kinematics."""
    engine = MultimodalFusionEngine()
    
    audio_probs = {
        'happy': 0.70,
        'neutral': 0.10,
        'sad': 0.05,
        'angry': 0.05,
        'fear': 0.04,
        'surprise': 0.04,
        'disgust': 0.02
    }

    # High velocity and MAR bursts characteristic of active happy speech
    lip_points = [
        LipTrackingPoint(
            timestamp=i*0.04, frame_idx=i, mouth_aspect_ratio=0.35 + 0.15*np.sin(i),
            lip_velocity=0.8, is_speaking=True, face_detected=True,
            tracking_confidence=0.95, mouth_box=(50, 50, 40, 30)
        ) for i in range(25)
    ]

    fusion = engine.fuse_predictions(audio_probs, lip_points)
    assert 'multimodal_emotion' in fusion
    assert 'multimodal_confidence' in fusion
    assert 'audio_only_emotion' in fusion
    assert 'visual_only_emotion' in fusion
    assert 'fusion_weights' in fusion
    assert fusion['multimodal_emotion'] in config.EMOTIONS.values()
    assert abs(sum(fusion['multimodal_probabilities'].values()) - 1.0) < 0.01


def test_database_and_pdf_report_with_lipsync_data():
    """Test storing video analysis in SQLite and generating multimodal PDF report."""
    init_db()
    test_id = f"test_vid_{int(np.random.randint(10000, 99999))}"

    sync_metrics = {
        'sync_quality_index': 88.5,
        'time_offset_ms': 12.0,
        'correlation_coefficient': 0.45,
        'sync_status': 'In-Sync (Broadcast Standard <45ms)',
        'speaking_activity_pct': 72.0,
        'tracking_consistency_pct': 98.0
    }

    save_analysis(
        analysis_id=test_id,
        filename="multimodal_speaker_test.mp4",
        source_type="Video Upload",
        duration=3.5,
        predicted_emotion="happy",
        confidence=0.85,
        is_low_confidence=False,
        probabilities={'happy': 0.85, 'neutral': 0.15},
        acoustic_metrics={'mean_rms': 0.05, 'mean_zcr': 0.08},
        segment_results=None,
        model_version="Multimodal Audio-Visual Late Fusion v2.0",
        processing_time=0.45,
        media_type="video",
        sync_offset_ms=12.0,
        sync_quality_score=88.5,
        visual_metrics=sync_metrics
    )

    rec = get_analysis_by_id(test_id)
    assert rec is not None
    assert rec['media_type'] == 'video'
    assert rec['sync_offset_ms'] == 12.0
    assert rec['sync_quality_score'] == 88.5

    # Generate PDF Report
    audio_test_series = np.sin(np.linspace(0, 3.5, 3500))
    pdf_buf = generate_pdf_report(
        analysis_record=rec,
        audio_time_series=audio_test_series,
        sample_rate=22050,
        sync_results=sync_metrics
    )
    pdf_bytes = pdf_buf.getvalue()
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

    # Cleanup
    delete_analysis(test_id)


if __name__ == '__main__':
    print("Running AI Lip-Sync and Multimodal Test Suite...")
    test_video_processor_metadata_and_extraction()
    print("✅ test_video_processor_metadata_and_extraction passed")
    test_lip_sync_detector_tracking_and_cross_correlation()
    print("✅ test_lip_sync_detector_tracking_and_cross_correlation passed")
    test_visual_speech_recognizer_adapter_diagnostics()
    print("✅ test_visual_speech_recognizer_adapter_diagnostics passed")
    test_multimodal_emotion_fusion()
    print("✅ test_multimodal_emotion_fusion passed")
    test_database_and_pdf_report_with_lipsync_data()
    print("✅ test_database_and_pdf_report_with_lipsync_data passed")
    print("\n🎉 ALL 5 LIP-SYNC AND MULTIMODAL TESTS PASSED SUCCESSFULLY!")
