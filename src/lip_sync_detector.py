"""
Lip-Sync Detection & Visual Speech Analysis Module.
Extracts facial & mouth articulation dynamics, calculates Mouth Aspect Ratio (MAR),
lip velocity, and performs audio-visual temporal cross-correlation for synchronization analysis.
"""
import os
import cv2
import numpy as np
import scipy.signal
from dataclasses import dataclass


@dataclass
class LipTrackingPoint:
    timestamp: float
    frame_idx: int
    mouth_aspect_ratio: float  # MAR (vertical / horizontal opening)
    lip_velocity: float        # |d(MAR)/dt|
    is_speaking: bool
    face_detected: bool
    tracking_confidence: float
    mouth_box: tuple[int, int, int, int] | None  # (x, y, w, h)


class LipSyncDetector:
    """
    Analyzes visual mouth articulation dynamics and evaluates audio-video synchronization
    using normalized cross-correlation between acoustic RMS energy and visual lip motion.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        # Initialize Haar cascade classifier if xml exists on disk
        cascade_dir = getattr(cv2.data, 'haarcascades', '')
        face_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml') if cascade_dir else ''
        if face_path and os.path.exists(face_path):
            try:
                self.face_cascade = cv2.CascadeClassifier(face_path)
                if self.face_cascade.empty():
                    self.face_cascade = None
            except Exception:
                self.face_cascade = None
        else:
            self.face_cascade = None

    def _extract_mouth_geometry_robust(self, rgb_frame: np.ndarray) -> tuple[float, tuple[int, int, int, int] | None, float]:
        """
        Extracts mouth bounding box and Mouth Aspect Ratio (MAR) using multi-tier computer vision:
        1. Cascade classifier (if available)
        2. YCrCb skin-color segmentation & morphological face-mouth spatial mapping
        3. Adaptive oral cavity contour & gradient analysis
        
        Returns:
            (mar, mouth_box, tracking_confidence)
        """
        h, w, _ = rgb_frame.shape
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        fx, fy, fw, fh = 0, 0, 0, 0
        face_found = False
        confidence = 0.0

        # Tier 1: Try Cascade Classifier
        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.15,
                    minNeighbors=5,
                    minSize=(int(w * 0.15), int(h * 0.15))
                )
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = faces[0]
                    face_found = True
                    confidence = 0.90
            except Exception:
                pass

        # Tier 2: Skin-color segmentation in YCrCb color space
        if not face_found:
            ycrcb = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2YCrCb)
            # Universal human skin chromaticity range
            skin_mask = cv2.inRange(ycrcb, (0, 130, 75), (255, 178, 130))
            
            # Morphological noise removal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                valid_c = [c for c in contours if cv2.contourArea(c) > (w * h * 0.03)]
                if valid_c:
                    largest_c = max(valid_c, key=cv2.contourArea)
                    fx, fy, fw, fh = cv2.boundingRect(largest_c)
                    face_found = True
                    confidence = 0.82

        # Tier 3: Center-quadrant fallback
        if not face_found:
            fx = int(w * 0.25)
            fy = int(h * 0.20)
            fw = int(w * 0.50)
            fh = int(h * 0.60)
            confidence = 0.40

        # Anatomical Mouth Region: lower 35% of face, center 60% horizontally
        mx = int(fx + fw * 0.20)
        my = int(fy + fh * 0.60)
        mw = int(fw * 0.60)
        mh = int(fh * 0.35)

        # Boundary clamping
        mx = max(0, min(mx, w - 1))
        my = max(0, min(my, h - 1))
        mw = max(10, min(w - mx, mw))
        mh = max(10, min(h - my, mh))

        mouth_roi_gray = gray[my:my+mh, mx:mx+mw]

        # Oral cavity detection via adaptive thresholding
        blur = cv2.GaussianBlur(mouth_roi_gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        opening_h = mh * 0.22  # baseline resting opening
        opening_w = mw * 0.68

        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > (mw * mh * 0.015)]
            if valid_contours:
                largest_c = max(valid_contours, key=cv2.contourArea)
                _, _, cw, ch = cv2.boundingRect(largest_c)
                opening_h = max(opening_h, float(ch))
                opening_w = max(opening_w, float(cw))

        mar = float(opening_h / max(opening_w, 1.0))
        mar = round(np.clip(mar, 0.05, 1.4), 4)

        return mar, (mx, my, mw, mh), confidence

    def track_lip_movement(self, frames_list: list[dict]) -> list[LipTrackingPoint]:
        """
        Processes a sequence of video frames to produce temporal lip landmark trajectories.
        
        Args:
            frames_list: List of dicts with keys 'timestamp', 'frame_idx', 'image' (RGB ndarray)
            
        Returns:
            List of LipTrackingPoint objects
        """
        if not frames_list:
            return []

        raw_points = []
        prev_mar = 0.1

        for idx, item in enumerate(frames_list):
            t = item['timestamp']
            f_idx = item['frame_idx']
            img = item['image']

            mar, mouth_box, conf = self._extract_mouth_geometry_robust(img)
            face_detected = (conf >= 0.50)

            # Compute smoothed lip velocity
            lip_vel = abs(mar - prev_mar) * self.fps if idx > 0 else 0.0
            prev_mar = mar

            raw_points.append(LipTrackingPoint(
                timestamp=t,
                frame_idx=f_idx,
                mouth_aspect_ratio=round(float(mar), 4),
                lip_velocity=round(float(lip_vel), 4),
                is_speaking=False,
                face_detected=face_detected,
                tracking_confidence=round(float(conf), 2),
                mouth_box=mouth_box
            ))

        # Smooth MAR trajectory using Savitzky-Golay filter
        mars = np.array([p.mouth_aspect_ratio for p in raw_points])
        
        if len(mars) >= 7:
            win_len = min(7, len(mars) if len(mars) % 2 != 0 else len(mars) - 1)
            smoothed_mars = scipy.signal.savgol_filter(mars, window_length=win_len, polyorder=2)
        else:
            smoothed_mars = mars

        smoothed_mars = np.clip(smoothed_mars, 0.05, 1.5)
        
        # Dynamic speaking threshold
        min_mar = np.percentile(smoothed_mars, 10)
        max_mar = np.percentile(smoothed_mars, 90)
        threshold_speaking = min_mar + 0.15 * max(max_mar - min_mar, 0.04)

        processed_points = []
        for i, p in enumerate(raw_points):
            s_mar = float(smoothed_mars[i])
            s_vel = abs(s_mar - smoothed_mars[i-1]) * self.fps if i > 0 else 0.0
            
            p.mouth_aspect_ratio = round(s_mar, 4)
            p.lip_velocity = round(float(s_vel), 4)
            p.is_speaking = bool(s_mar > threshold_speaking or s_vel > 0.25)
            processed_points.append(p)

        return processed_points

    def compute_audio_energy_envelope(
        self,
        audio: np.ndarray,
        sr: int,
        target_timestamps: np.ndarray,
        window_duration: float = 0.05
    ) -> np.ndarray:
        """
        Computes the acoustic RMS energy envelope aligned precisely to target video timestamps.
        """
        if len(audio) == 0 or len(target_timestamps) == 0:
            return np.zeros(len(target_timestamps), dtype=np.float32)

        window_samples = int(window_duration * sr)
        energy_env = np.zeros(len(target_timestamps), dtype=np.float32)

        for i, t in enumerate(target_timestamps):
            center_idx = int(t * sr)
            start_idx = max(0, center_idx - window_samples // 2)
            end_idx = min(len(audio), center_idx + window_samples // 2)

            if end_idx > start_idx:
                segment = audio[start_idx:end_idx]
                rms = np.sqrt(np.mean(segment ** 2))
                energy_env[i] = rms
            else:
                energy_env[i] = 0.0

        max_e = np.max(energy_env)
        if max_e > 0:
            energy_env = energy_env / max_e

        return energy_env

    def analyze_synchronization(
        self,
        lip_points: list[LipTrackingPoint],
        audio: np.ndarray,
        sr: int,
        max_lag_ms: float = 500.0
    ) -> dict:
        """
        Evaluates temporal audio-video synchronization using normalized cross-correlation.
        """
        if not lip_points or len(lip_points) < 5:
            return {
                'sync_quality_index': 0.0,
                'time_offset_ms': 0.0,
                'correlation_coefficient': 0.0,
                'sync_status': 'Indeterminate (Insufficient Frames)',
                'sync_badge_color': '#94A3B8',
                'speaking_activity_pct': 0.0,
                'tracking_consistency_pct': 0.0,
                'lags_ms': [],
                'cross_correlation': [],
                'is_in_sync': False,
                'disclaimer': 'Analysis inconclusive due to insufficient video duration or face detection.'
            }

        timestamps = np.array([p.timestamp for p in lip_points])
        mar_series = np.array([p.mouth_aspect_ratio for p in lip_points])
        vel_series = np.array([p.lip_velocity for p in lip_points])
        face_detected_arr = np.array([p.face_detected for p in lip_points])

        tracking_consistency = float(np.mean(face_detected_arr)) * 100.0
        speaking_activity = float(np.mean([p.is_speaking for p in lip_points])) * 100.0

        # Calculate matching Audio Energy Envelope
        audio_env = self.compute_audio_energy_envelope(audio, sr, timestamps)

        # Visual articulation signal
        visual_signal = 0.6 * (mar_series - np.mean(mar_series)) + 0.4 * (vel_series - np.mean(vel_series))
        v_std = np.std(visual_signal)
        if v_std > 1e-6:
            visual_signal = visual_signal / v_std

        a_signal = audio_env - np.mean(audio_env)
        a_std = np.std(a_signal)
        if a_std > 1e-6:
            a_signal = a_signal / a_std

        dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else (1.0 / self.fps)
        effective_fps = 1.0 / max(dt, 0.001)

        max_lag_frames = int(round((max_lag_ms / 1000.0) * effective_fps))
        max_lag_frames = max(1, min(max_lag_frames, len(timestamps) // 2))

        # Cross-correlation
        n = len(visual_signal)
        raw_corr = scipy.signal.correlate(a_signal, visual_signal, mode='full') / n
        lags = scipy.signal.correlation_lags(n, n, mode='full')

        valid_mask = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
        window_lags = lags[valid_mask]
        window_corr = raw_corr[valid_mask]

        if len(window_corr) == 0:
            best_lag_frames = 0
            max_corr_val = 0.0
        else:
            best_idx = np.argmax(window_corr)
            best_lag_frames = window_lags[best_idx]
            max_corr_val = float(window_corr[best_idx])

        time_offset_ms = round(float(best_lag_frames * (1000.0 / effective_fps)), 1)
        peak_corr = round(float(np.clip(max_corr_val, -1.0, 1.0)), 3)

        # Sync Quality Index (SQI) [0% to 100%]
        offset_penalty = max(0.0, 1.0 - (abs(time_offset_ms) / 250.0))
        corr_score = max(0.0, peak_corr)
        tracking_score = tracking_consistency / 100.0

        sqi = (0.50 * corr_score + 0.30 * offset_penalty + 0.20 * tracking_score) * 100.0
        sqi = round(float(np.clip(sqi, 0.0, 100.0)), 1)

        # Status Classification
        if tracking_consistency < 30.0:
            status = "Low Confidence (Occluded / Missing Face)"
            badge_color = "#F59E0B"
            is_in_sync = False
        elif abs(time_offset_ms) <= 45.0 and peak_corr >= 0.20:
            status = "In-Sync (Broadcast Standard <45ms)"
            badge_color = "#10B981"
            is_in_sync = True
        elif abs(time_offset_ms) <= 90.0 and peak_corr >= 0.15:
            status = f"Acceptable Sync (Offset: {time_offset_ms:+.1f} ms)"
            badge_color = "#06B6D4"
            is_in_sync = True
        elif time_offset_ms > 90.0:
            status = f"Audio Leads Video (+{time_offset_ms:.1f} ms)"
            badge_color = "#EF4444"
            is_in_sync = False
        elif time_offset_ms < -90.0:
            status = f"Video Leads Audio ({time_offset_ms:.1f} ms)"
            badge_color = "#F97316"
            is_in_sync = False
        else:
            status = "Weak Audio-Visual Correlation"
            badge_color = "#64748B"
            is_in_sync = False

        return {
            'sync_quality_index': sqi,
            'time_offset_ms': time_offset_ms,
            'correlation_coefficient': peak_corr,
            'sync_status': status,
            'sync_badge_color': badge_color,
            'speaking_activity_pct': round(speaking_activity, 1),
            'tracking_consistency_pct': round(tracking_consistency, 1),
            'lags_ms': [round(float(l * (1000.0 / effective_fps)), 1) for l in window_lags],
            'cross_correlation': [round(float(c), 4) for c in window_corr],
            'is_in_sync': is_in_sync,
            'timestamps': [round(float(t), 3) for t in timestamps],
            'audio_energy_envelope': [round(float(e), 4) for e in audio_env],
            'lip_mar_trajectory': [round(float(m), 4) for m in mar_series],
            'lip_velocity_trajectory': [round(float(v), 4) for v in vel_series],
            'disclaimer': (
                "Responsible AI Notice: A/V temporal offsets can result from hardware capture latency, "
                "Bluetooth codecs, container muxing, or video editing delays. "
                "Desynchronization metrics should not be used as sole evidence for deepfake or fraud detection."
            )
        }

    def annotate_frame(
        self,
        rgb_frame: np.ndarray,
        tracking_point: LipTrackingPoint,
        sync_status: str = "Analyzing"
    ) -> np.ndarray:
        """
        Draws visual HUD overlay on video frame highlighting mouth ROI, MAR gauge, and sync state.
        """
        annotated = rgb_frame.copy()
        h, w, _ = annotated.shape

        if tracking_point.mouth_box is not None:
            mx, my, mw, mh = tracking_point.mouth_box
            cv2.rectangle(annotated, (mx, my), (mx + mw, my + mh), (56, 189, 248), 2)
            corner_len = min(12, int(mw * 0.2))
            cv2.line(annotated, (mx, my), (mx + corner_len, my), (255, 255, 255), 2)
            cv2.line(annotated, (mx, my), (mx, my + corner_len), (255, 255, 255), 2)

            cx = mx + mw // 2
            cy = my + mh // 2
            cv2.circle(annotated, (cx, cy), 3, (16, 185, 129), -1)

            label = f"MAR: {tracking_point.mouth_aspect_ratio:.2f}"
            cv2.putText(
                annotated, label, (mx, max(18, my - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA
            )

        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (11, 17, 32), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        hud_text = f"T: {tracking_point.timestamp:.2f}s | Activity: {'SPEAKING' if tracking_point.is_speaking else 'REST'} | Status: {sync_status[:30]}"
        cv2.putText(
            annotated, hud_text, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 245, 249), 1, cv2.LINE_AA
        )

        return annotated
