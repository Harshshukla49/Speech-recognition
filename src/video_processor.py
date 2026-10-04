"""
Video Processing Module for Speech Emotion Recognition & Lip-Sync Platform.
Provides robust media extraction, container validation, audio demuxing, and frame sampling.
"""
import os
import io
import tempfile
import numpy as np
import av
import cv2


class VideoProcessor:
    """Handles video validation, frame extraction, and synchronized audio demuxing."""

    SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.m4v'}
    MAX_FILE_SIZE_MB = 150
    MAX_DURATION_SECONDS = 180  # 3 minutes for real-time interactive analysis

    def __init__(self, target_sample_rate: int = 22050, max_fps: int = 30):
        self.target_sample_rate = target_sample_rate
        self.max_fps = max_fps

    @staticmethod
    def save_temp_video(file_input, suffix: str = ".mp4") -> str:
        """
        Saves uploaded file bytes or Streamlit UploadedFile to a managed temporary file.
        Returns absolute path to the temporary video file.
        """
        temp_dir = os.path.join(tempfile.gettempdir(), "speech_emotion_video")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir)
        
        if hasattr(file_input, 'read'):
            content = file_input.read()
            if hasattr(file_input, 'seek'):
                file_input.seek(0)
            temp_file.write(content)
        elif isinstance(file_input, bytes):
            temp_file.write(file_input)
        elif isinstance(file_input, str) and os.path.exists(file_input):
            with open(file_input, 'rb') as f:
                temp_file.write(f.read())
        else:
            temp_file.close()
            raise ValueError("Unsupported video input type.")
            
        temp_file.flush()
        temp_file.close()
        return temp_file.name

    def get_metadata(self, video_path: str) -> dict:
        """
        Inspects video container and returns comprehensive stream metadata.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        metadata = {
            'file_path': video_path,
            'file_name': os.path.basename(video_path),
            'file_size_mb': round(file_size_mb, 2),
            'duration_sec': 0.0,
            'fps': 0.0,
            'frame_count': 0,
            'width': 0,
            'height': 0,
            'video_codec': 'unknown',
            'audio_codec': 'none',
            'has_audio': False,
            'audio_sample_rate': 0,
            'audio_channels': 0,
            'is_valid': True,
            'validation_message': 'Valid video file'
        }

        try:
            with av.open(video_path) as container:
                # Video Stream
                video_streams = [s for s in container.streams if s.type == 'video']
                if video_streams:
                    v_stream = video_streams[0]
                    metadata['video_codec'] = v_stream.codec_context.name
                    metadata['width'] = v_stream.codec_context.width or 0
                    metadata['height'] = v_stream.codec_context.height or 0
                    if v_stream.average_rate:
                        metadata['fps'] = float(v_stream.average_rate)
                    elif v_stream.rate:
                        metadata['fps'] = float(v_stream.rate)
                    
                    if v_stream.duration and v_stream.time_base:
                        metadata['duration_sec'] = float(v_stream.duration * v_stream.time_base)
                    elif container.duration:
                        metadata['duration_sec'] = float(container.duration / av.time.AV_TIME_BASE)
                        
                    metadata['frame_count'] = v_stream.frames or int(metadata['duration_sec'] * metadata['fps'])

                # Audio Stream
                audio_streams = [s for s in container.streams if s.type == 'audio']
                if audio_streams:
                    a_stream = audio_streams[0]
                    metadata['has_audio'] = True
                    metadata['audio_codec'] = a_stream.codec_context.name
                    metadata['audio_sample_rate'] = a_stream.codec_context.sample_rate or 0
                    metadata['audio_channels'] = a_stream.codec_context.channels or 1
                    if metadata['duration_sec'] == 0.0 and a_stream.duration and a_stream.time_base:
                        metadata['duration_sec'] = float(a_stream.duration * a_stream.time_base)

        except Exception as e:
            # Fallback to OpenCV metadata extraction
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    metadata['fps'] = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if metadata['fps'] > 0:
                        metadata['duration_sec'] = metadata['frame_count'] / metadata['fps']
                    cap.release()
            except Exception:
                pass
            metadata['validation_message'] = f"Metadata extraction partial notice: {str(e)}"

        # Validation Checks
        if metadata['duration_sec'] > self.MAX_DURATION_SECONDS:
            metadata['is_valid'] = False
            metadata['validation_message'] = f"Duration ({metadata['duration_sec']:.1f}s) exceeds maximum allowed ({self.MAX_DURATION_SECONDS}s)."
        elif file_size_mb > self.MAX_FILE_SIZE_MB:
            metadata['is_valid'] = False
            metadata['validation_message'] = f"File size ({file_size_mb:.1f}MB) exceeds limit of {self.MAX_FILE_SIZE_MB}MB."
        elif metadata['width'] == 0 or metadata['height'] == 0:
            metadata['is_valid'] = False
            metadata['validation_message'] = "Could not detect valid video stream or frame dimensions."

        return metadata

    def extract_audio(self, video_path: str) -> tuple[np.ndarray, int]:
        """
        Demuxes audio stream directly from video container and resamples to target sample rate.
        Returns (audio_time_series, sample_rate) as mono float32 numpy array.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            with av.open(video_path) as container:
                audio_streams = [s for s in container.streams if s.type == 'audio']
                if not audio_streams:
                    # Return silent audio array if video has no audio track
                    metadata = self.get_metadata(video_path)
                    dur = max(metadata.get('duration_sec', 1.0), 1.0)
                    silence = np.zeros(int(dur * self.target_sample_rate), dtype=np.float32)
                    return silence, self.target_sample_rate

                # Configure resampler to target sample rate, mono (1 channel), float32
                resampler = av.AudioResampler(
                    format='flt',
                    layout='mono',
                    rate=self.target_sample_rate
                )

                audio_chunks = []
                for frame in container.decode(audio_streams[0]):
                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        chunk = r_frame.to_ndarray()
                        if chunk.ndim > 1:
                            chunk = chunk.flatten()
                        audio_chunks.append(chunk)

                if audio_chunks:
                    audio_data = np.concatenate(audio_chunks).astype(np.float32)
                    # Normalize amplitude to [-1.0, 1.0]
                    max_val = np.max(np.abs(audio_data))
                    if max_val > 0:
                        audio_data = audio_data / max_val
                    return audio_data, self.target_sample_rate
                else:
                    return np.zeros(self.target_sample_rate, dtype=np.float32), self.target_sample_rate

        except Exception as e:
            # Fallback: try reading audio using soundfile/librosa or return synthetic silence
            import librosa
            try:
                y, sr = librosa.load(video_path, sr=self.target_sample_rate, mono=True)
                return y.astype(np.float32), sr
            except Exception:
                # Return 3s silence fallback
                return np.zeros(3 * self.target_sample_rate, dtype=np.float32), self.target_sample_rate

    def extract_frames(
        self,
        video_path: str,
        target_fps: float = None,
        max_frames: int = 1500,
        resize_dims: tuple[int, int] = None
    ) -> tuple[list[dict], dict]:
        """
        Extracts sampled video frames with precise timestamps.
        
        Returns:
            frames: List of dicts [{'timestamp': float, 'frame_idx': int, 'image': np.ndarray (RGB)}]
            meta: Dict containing extraction summary
        """
        metadata = self.get_metadata(video_path)
        src_fps = metadata.get('fps', 30.0)
        if src_fps <= 0:
            src_fps = 30.0
            
        effective_fps = target_fps if target_fps and target_fps < src_fps else min(src_fps, self.max_fps)
        step_interval = max(1, int(round(src_fps / effective_fps)))

        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        frame_idx = 0
        extracted_count = 0

        while cap.isOpened() and extracted_count < max_frames:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            if frame_idx % step_interval == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                
                if resize_dims is not None:
                    rgb_frame = cv2.resize(rgb_frame, resize_dims, interpolation=cv2.INTER_AREA)
                    
                timestamp = frame_idx / src_fps
                frames.append({
                    'timestamp': round(timestamp, 4),
                    'frame_idx': frame_idx,
                    'image': rgb_frame
                })
                extracted_count += 1

            frame_idx += 1

        cap.release()

        summary = {
            'total_source_frames': frame_idx,
            'extracted_frames': len(frames),
            'source_fps': src_fps,
            'sampling_fps': effective_fps,
            'duration_sec': metadata.get('duration_sec', frame_idx / src_fps)
        }
        
        return frames, summary
