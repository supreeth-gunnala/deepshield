"""
DeepShield Audio Quality & Deepfake Risk Analysis Engine
Calculates pre-prediction acoustic health metrics to evaluate audio quality and deepfake prediction reliability.
"""

import torch
import numpy as np
from typing import Dict, Any, Union

class AudioQualityAnalyzer:
    """
    Acoustic health analyzer measuring SNR, clipping, silence ratio, RMS energy, ZCR,
    dynamic range, and prediction reliability risk.
    """

    def __init__(self, silence_threshold_db: float = -40.0, clipping_threshold: float = 0.999):
        self.silence_threshold_db = silence_threshold_db
        self.clipping_threshold = clipping_threshold

    def analyze(self, waveform: Union[torch.Tensor, np.ndarray], sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Analyzes a preprocessed or raw audio waveform.

        Args:
            waveform: Tensor or numpy array of shape (1, Samples) or (Samples,).
            sample_rate: Sample rate in Hz.

        Returns:
            Dictionary containing calculated acoustic metrics and qualitative risk ratings.
        """
        if isinstance(waveform, torch.Tensor):
            samples = waveform.squeeze().detach().cpu().numpy()
        else:
            samples = np.squeeze(waveform)

        n_samples = len(samples)
        if n_samples == 0:
            return self._empty_response()

        # 1. Clipping Percentage (saturated samples near 1.0)
        clipped_samples = np.sum(np.abs(samples) >= self.clipping_threshold)
        clipping_pct = float((clipped_samples / n_samples) * 100.0)

        # 2. RMS Energy & Frame Analysis
        frame_size = int(0.025 * sample_rate) # 25ms frames
        hop_size = int(0.010 * sample_rate) # 10ms hop

        n_frames = max(1, (n_samples - frame_size) // hop_size + 1)
        frame_rms = []

        for i in range(n_frames):
            start = i * hop_size
            end = min(n_samples, start + frame_size)
            frame = samples[start:end]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
            frame_rms.append(rms)

        frame_rms = np.array(frame_rms)
        total_rms = float(np.sqrt(np.mean(samples ** 2) + 1e-10))

        # 3. Silence Percentage
        max_rms = np.max(frame_rms) if len(frame_rms) > 0 else 1e-5
        frame_db = 20 * np.log10(frame_rms / (max_rms + 1e-10) + 1e-10)
        silent_frames = np.sum(frame_db < self.silence_threshold_db)
        silence_pct = float((silent_frames / len(frame_db)) * 100.0)

        # 4. Signal-to-Noise Ratio (SNR) Estimation
        sorted_rms = np.sort(frame_rms)
        n_noise = max(1, int(0.10 * len(sorted_rms)))
        n_speech = max(1, int(0.20 * len(sorted_rms)))

        noise_floor_rms = np.mean(sorted_rms[:n_noise])
        speech_rms = np.mean(sorted_rms[-n_speech:])

        # If noise floor is close to speech RMS (signal has no quiet pauses)
        if noise_floor_rms >= 0.70 * speech_rms and total_rms > 0.05:
            # Measure high-frequency noise perturbation via 2nd-order differences
            hff = np.std(np.diff(samples, n=2))
            # Clean single tone / speech has low hff (~0.05 - 0.10); noise increases hff proportionally
            estimated_noise_rms = max(1e-4, (hff - 0.05) / np.sqrt(6.0)) if hff > 0.05 else 1e-4
            snr_linear = (total_rms ** 2) / (estimated_noise_rms ** 2 + 1e-10)
            snr_db = float(10 * np.log10(snr_linear + 1e-10))
        else:
            snr_linear = (speech_rms ** 2) / (noise_floor_rms ** 2 + 1e-10)
            snr_db = float(10 * np.log10(snr_linear + 1e-10))
        
        snr_db = float(np.clip(snr_db, 0.0, 45.0))

        # 5. Zero Crossing Rate (ZCR)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(samples)))))

        # 6. Dynamic Range (dB)
        peak_amp = np.max(np.abs(samples))
        dynamic_range_db = float(20 * np.log10((peak_amp + 1e-10) / (noise_floor_rms + 1e-10)))
        dynamic_range_db = float(np.clip(dynamic_range_db, 0.0, 80.0))

        # 7. Composite Quality Score (0 - 100)
        score = 100.0
        
        # Clipping penalty applies only if clipping percentage is elevated (> 3.0%)
        if clipping_pct > 3.0:
            score -= min(35.0, (clipping_pct - 3.0) * 15.0)

        # Low SNR penalty
        if snr_db < 20.0:
            score -= (20.0 - snr_db) * 2.0
            
        # Excessive silence penalty
        if silence_pct > 55.0:
            score -= (silence_pct - 55.0) * 0.5

        # Dynamic range penalty (only if SNR is low)
        if dynamic_range_db < 15.0 and snr_db < 20.0:
            score -= (15.0 - dynamic_range_db) * 1.5

        quality_score = float(np.clip(score, 0.0, 100.0))

        # 8. Qualitative Ratings & Reliability Risk Assessment
        if quality_score >= 85:
            quality_rating = "Excellent"
        elif quality_score >= 70:
            quality_rating = "Good"
        elif quality_score >= 50:
            quality_rating = "Moderate"
        else:
            quality_rating = "Poor"

        if snr_db >= 20.0:
            noise_level = "Low"
        elif snr_db >= 10.0:
            noise_level = "Moderate"
        else:
            noise_level = "High"

        # Reliability assessment
        if clipping_pct > 3.0 or snr_db < 5.0 or quality_score < 40:
            prediction_reliability = "Low"
            reliability_reason = "High clipping or severe background noise present"
        elif clipping_pct > 0.8 or snr_db < 12.0 or quality_score < 60:
            prediction_reliability = "Reduced"
            reliability_reason = "Moderate noise or minor waveform clipping detected"
        elif snr_db < 18.0 or silence_pct > 55.0:
            prediction_reliability = "Moderate"
            reliability_reason = "Slight acoustic corruption or long silent pauses"
        else:
            prediction_reliability = "High"
            reliability_reason = "Clean acoustic signal with optimal dynamic range"

        return {
            "snr_db": snr_db,
            "clipping_pct": clipping_pct,
            "silence_pct": silence_pct,
            "rms_energy": total_rms,
            "zcr": zcr,
            "dynamic_range_db": dynamic_range_db,
            "quality_score": quality_score,
            "quality_rating": quality_rating,
            "noise_level": noise_level,
            "prediction_reliability": prediction_reliability,
            "reliability_reason": reliability_reason
        }

    def _empty_response(self) -> Dict[str, Any]:
        return {
            "snr_db": 0.0,
            "clipping_pct": 0.0,
            "silence_pct": 0.0,
            "rms_energy": 0.0,
            "zcr": 0.0,
            "dynamic_range_db": 0.0,
            "quality_score": 0.0,
            "quality_rating": "Unknown",
            "noise_level": "Unknown",
            "prediction_reliability": "Low",
            "reliability_reason": "Empty audio signal"
        }
