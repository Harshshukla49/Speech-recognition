"""
Multimodal Emotion Fusion Module.
Combines 7-class deep acoustic emotion predictions with visual articulatory kinematics and facial dynamics.
Provides transparent decision-level late fusion with confidence weighting and modular comparison.
"""
import numpy as np
import config


class MultimodalFusionEngine:
    """
    Fuses acoustic emotion probabilities with visual facial dynamics.
    Enables comparative analysis across Audio-Only, Visual Kinematics, and Multimodal Consensus.
    """

    EMOTIONS = [config.EMOTIONS[i] for i in range(len(config.EMOTIONS))]

    def __init__(self, audio_weight_default: float = 0.70):
        self.audio_weight_default = audio_weight_default

    def extract_visual_kinematic_priors(
        self,
        lip_points: list,
        video_metadata: dict | None = None
    ) -> tuple[dict[str, float], float]:
        """
        Estimates a normalized 7-class visual kinematic distribution from mouth aspect ratios,
        articulation speed, and face tracking continuity.
        
        Returns:
            (visual_probabilities, visual_tracking_confidence)
        """
        if not lip_points or len(lip_points) < 5:
            # Uniform fallback distribution if no video points
            uniform = {emo: round(1.0 / len(self.EMOTIONS), 4) for emo in self.EMOTIONS}
            return uniform, 0.0

        mars = np.array([p.mouth_aspect_ratio for p in lip_points])
        velocities = np.array([p.lip_velocity for p in lip_points])
        tracking_ratios = np.array([p.face_detected for p in lip_points])

        mean_mar = float(np.mean(mars))
        std_mar = float(np.std(mars))
        max_mar = float(np.max(mars))
        mean_vel = float(np.mean(velocities))
        max_vel = float(np.max(velocities))
        tracking_conf = float(np.mean(tracking_ratios))

        # Empirical facial dynamics mapping
        # Surprise: High vertical opening (large MAR, large max_mar)
        surprise_score = np.clip((max_mar - 0.25) / 0.20, 0.0, 1.0) * 1.5

        # Happy: High velocity bursts, moderate-high MAR variance (smile dynamics)
        happy_score = np.clip((std_mar / 0.08) * 0.8 + (mean_vel / 0.5) * 0.5, 0.0, 1.5)

        # Angry: High velocity, sudden bursts, tense baseline
        angry_score = np.clip((max_vel / 1.2) * 0.9 + (std_mar / 0.10) * 0.6, 0.0, 1.5)

        # Fear: Rapid tremors, erratic velocity
        fear_score = np.clip((mean_vel / 0.6) * 0.7 + (std_mar / 0.06) * 0.5, 0.0, 1.2)

        # Disgust: Asymmetric or suppressed articulation
        disgust_score = np.clip((0.15 - mean_mar) / 0.10, 0.0, 0.8) + 0.2

        # Sad: Depressed articulation velocity, flat MAR
        sad_score = np.clip(1.0 - (mean_vel / 0.4), 0.0, 1.2) * np.clip(1.0 - (std_mar / 0.05), 0.0, 1.0)

        # Neutral: Moderate steady MAR, low variance
        neutral_score = np.clip(1.0 - (std_mar / 0.06), 0.0, 1.2) * np.clip(1.0 - (mean_vel / 0.5), 0.0, 1.0)

        raw_scores = {
            'neutral': max(0.1, float(neutral_score)),
            'happy': max(0.1, float(happy_score)),
            'sad': max(0.1, float(sad_score)),
            'angry': max(0.1, float(angry_score)),
            'fear': max(0.1, float(fear_score)),
            'disgust': max(0.1, float(disgust_score)),
            'surprise': max(0.1, float(surprise_score))
        }

        # Softmax normalization
        exp_scores = {k: np.exp(v) for k, v in raw_scores.items()}
        sum_exp = sum(exp_scores.values())
        visual_probs = {k: round(float(v / sum_exp), 4) for k, v in exp_scores.items()}

        return visual_probs, tracking_conf

    def fuse_predictions(
        self,
        audio_probabilities: dict[str, float],
        lip_points: list,
        user_audio_weight: float | None = None
    ) -> dict:
        """
        Performs adaptive decision-level fusion between acoustic model and visual kinematics.
        
        Args:
            audio_probabilities: Dict mapping emotion names to probabilities
            lip_points: List of LipTrackingPoint objects from video analysis
            user_audio_weight: Optional manual audio weight [0.0 - 1.0]
            
        Returns:
            Dict containing Audio-Only, Visual-Only, and Multimodal Fused results.
        """
        visual_probs, tracking_conf = self.extract_visual_kinematic_priors(lip_points)

        # Compute adaptive weights based on tracking reliability
        if user_audio_weight is not None:
            w_audio = np.clip(user_audio_weight, 0.1, 0.95)
        else:
            # If visual tracking was poor (occlusion / dark), increase audio reliance
            visual_reliability = tracking_conf
            w_audio = 0.50 + 0.40 * (1.0 - visual_reliability)
            w_audio = float(np.clip(w_audio, 0.50, 0.90))

        w_visual = 1.0 - w_audio

        fused_probs = {}
        for emo in self.EMOTIONS:
            p_a = audio_probabilities.get(emo, 0.0)
            p_v = visual_probs.get(emo, 0.0)
            fused_p = (w_audio * p_a) + (w_visual * p_v)
            fused_probs[emo] = round(float(fused_p), 4)

        # Normalize fused probabilities to sum to 1.0
        total_p = sum(fused_probs.values())
        if total_p > 0:
            fused_probs = {k: round(v / total_p, 4) for k, v in fused_probs.items()}

        top_audio_emo = max(audio_probabilities.items(), key=lambda x: x[1])[0]
        top_visual_emo = max(visual_probs.items(), key=lambda x: x[1])[0]
        top_fused_emo = max(fused_probs.items(), key=lambda x: x[1])[0]

        top_fused_conf = fused_probs[top_fused_emo]
        top_audio_conf = audio_probabilities[top_audio_emo]
        top_visual_conf = visual_probs[top_visual_emo]

        consensus_agreement = (top_audio_emo == top_visual_emo)

        return {
            'multimodal_emotion': top_fused_emo,
            'multimodal_confidence': round(top_fused_conf, 4),
            'multimodal_probabilities': fused_probs,
            'audio_only_emotion': top_audio_emo,
            'audio_only_confidence': round(top_audio_conf, 4),
            'audio_only_probabilities': audio_probabilities,
            'visual_only_emotion': top_visual_emo,
            'visual_only_confidence': round(top_visual_conf, 4),
            'visual_only_probabilities': visual_probs,
            'fusion_weights': {
                'audio_weight': round(w_audio, 3),
                'visual_weight': round(w_visual, 3)
            },
            'consensus_agreement': consensus_agreement,
            'visual_tracking_confidence': round(tracking_conf, 3),
            'methodology_note': (
                f"Dynamic decision-level late fusion applied with {w_audio*100:.1f}% acoustic weight "
                f"and {w_visual*100:.1f}% visual kinematic weight based on landmark continuity."
            )
        }
