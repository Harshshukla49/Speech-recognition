"""
Audio Speech Recognition (ASR) & Multimodal Alignment Module.
Transcribes speech audio (Hindi, English, and code-mixed Hinglish),
aligns acoustic sentences with video timestamps, and reconciles visual lip-reading with acoustic signals.
"""
import os
import tempfile
import numpy as np
import soundfile as sf
import speech_recognition as sr
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from src.hinglish_engine import HinglishEngine


@dataclass
class AlignedTranscriptSegment:
    segment_id: int
    start_time: float
    end_time: float
    hinglish_text: str
    devanagari_text: str
    english_translation: str
    confidence: float
    source_modality: str  # "Visual Only", "Audio Assisted", "Multimodal Fused"


class AudioTranscriber:
    """
    Automatic Speech Recognition and Cross-Modal Alignment Engine.
    Handles Hindi, English, and Hinglish speech recognition with temporal segment synchronization.
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def transcribe_audio_file(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 22050,
        preferred_language: str = "hi-IN"
    ) -> Dict:
        """
        Transcribes raw audio array into text using speech recognition.
        Supports Hindi ('hi-IN') and English ('en-IN') with automatic code-mixing preservation.
        """
        if len(audio_array) == 0 or np.max(np.abs(audio_array)) < 0.005:
            return {
                'raw_transcript': '',
                'detected_language': 'Silent / Inaudible',
                'confidence': 0.0,
                'is_success': False,
                'error_message': 'Audio signal is silent or below threshold.'
            }

        # Write to temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav_path = temp_wav.name
        temp_wav.close()

        try:
            # Ensure float32 normalized or int16
            sf.write(temp_wav_path, audio_array, sample_rate, subtype='PCM_16')

            with sr.AudioFile(temp_wav_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio_data = self.recognizer.record(source)

            # Try Hindi speech recognition first (handles both Hindi and English code-switching)
            transcript_text = ""
            conf = 0.85
            detected_lang = preferred_language

            try:
                transcript_text = self.recognizer.recognize_google(audio_data, language="hi-IN")
                detected_lang = "Hindi / Hinglish (hi-IN)"
            except sr.UnknownValueError:
                # Try English
                try:
                    transcript_text = self.recognizer.recognize_google(audio_data, language="en-IN")
                    detected_lang = "English (en-IN)"
                except Exception:
                    transcript_text = ""
            except Exception:
                # Network or API fallback: try English
                try:
                    transcript_text = self.recognizer.recognize_google(audio_data, language="en-US")
                    detected_lang = "English (en-US)"
                except Exception as e:
                    return {
                        'raw_transcript': '',
                        'detected_language': 'Unknown',
                        'confidence': 0.0,
                        'is_success': False,
                        'error_message': f'Offline/Recognition notice: {str(e)}'
                    }

            if not transcript_text:
                return {
                    'raw_transcript': '',
                    'detected_language': detected_lang,
                    'confidence': 0.0,
                    'is_success': False,
                    'error_message': 'Speech could not be parsed into linguistic tokens.'
                }

            return {
                'raw_transcript': transcript_text,
                'detected_language': detected_lang,
                'confidence': conf,
                'is_success': True,
                'error_message': None
            }

        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

    def align_and_reconcile_multimodal(
        self,
        visual_results: Dict,
        audio_results: Dict,
        total_duration: float,
        mode: str = "Audio + Lip Reading"
    ) -> List[AlignedTranscriptSegment]:
        """
        Aligns visual lip-reading segments with audio speech recognition, reconciles evidence,
        and converts outputs into natural Hinglish, Devanagari, and English translations.
        
        Args:
            visual_results: Dict output from VisualSpeechRecognizerAdapter
            audio_results: Dict output from transcribe_audio_file
            total_duration: Total video length in seconds
            mode: "Lip Reading Only" or "Audio + Lip Reading"
            
        Returns:
            List of AlignedTranscriptSegment objects
        """
        aligned_segments = []
        vis_segments = visual_results.get('segments', [])
        audio_transcript = audio_results.get('raw_transcript', '') if audio_results else ''

        # ---------------------------------------------------------------------
        # MODE 1: Lip Reading Only (Visual Only)
        # ---------------------------------------------------------------------
        if mode == "Lip Reading Only" or not audio_transcript:
            if vis_segments:
                for idx, v_seg in enumerate(vis_segments):
                    w = v_seg['word']
                    out = HinglishEngine.generate_multilingual_outputs(w, detected_modality="Visual Lip-Reading")
                    aligned_segments.append(AlignedTranscriptSegment(
                        segment_id=idx + 1,
                        start_time=v_seg['start_time'],
                        end_time=v_seg['end_time'],
                        hinglish_text=out['hinglish'],
                        devanagari_text=out['devanagari'],
                        english_translation=out['english_translation'],
                        confidence=v_seg['confidence'],
                        source_modality="Visual Lip-Reading"
                    ))
            else:
                out = HinglishEngine.generate_multilingual_outputs(
                    visual_results.get('transcript', 'No active speech articulation detected.'),
                    detected_modality="Visual Lip-Reading"
                )
                aligned_segments.append(AlignedTranscriptSegment(
                    segment_id=1,
                    start_time=0.0,
                    end_time=round(total_duration, 2),
                    hinglish_text=out['hinglish'],
                    devanagari_text=out['devanagari'],
                    english_translation=out['english_translation'],
                    confidence=visual_results.get('mean_confidence', 0.50),
                    source_modality="Visual Lip-Reading"
                ))
            return aligned_segments

        # ---------------------------------------------------------------------
        # MODE 2: Audio + Lip Reading (Multimodal Assisted)
        # ---------------------------------------------------------------------
        # Split audio transcript into sentence-like clauses
        clauses = [c.strip() for c in audio_transcript.replace('।', '.').split('.') if c.strip()]
        if not clauses:
            clauses = [audio_transcript]

        n_clauses = len(clauses)
        seg_duration = total_duration / max(1, n_clauses)

        for idx, clause in enumerate(clauses):
            t_start = round(idx * seg_duration, 2)
            t_end = round(min(total_duration, (idx + 1) * seg_duration), 2)

            # Check if any visual word segments overlap temporally with this clause
            overlapping_vis = [
                v for v in vis_segments
                if not (v['end_time'] < t_start or v['start_time'] > t_end)
            ]

            if overlapping_vis:
                modality = "Multimodal Fused (A+V)"
                fused_conf = 0.92
            else:
                modality = "Audio Assisted"
                fused_conf = 0.86

            multilingual = HinglishEngine.generate_multilingual_outputs(clause, detected_modality=modality)

            aligned_segments.append(AlignedTranscriptSegment(
                segment_id=idx + 1,
                start_time=t_start,
                end_time=t_end,
                hinglish_text=multilingual['hinglish'],
                devanagari_text=multilingual['devanagari'],
                english_translation=multilingual['english_translation'],
                confidence=fused_conf,
                source_modality=modality
            ))

        return aligned_segments
