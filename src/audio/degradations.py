"""
DeepShield Audio Degradation Engine Module
Simulates real-world acoustic channels and digital transmission corruptions:
- Additive Background Noise (SNR)
- Resampling Degradation (Downsampling)
- Narrowband Telephone / VoIP Bandpass Filtering
- Quantization & Compression Simulation
- Pitch Modification
- Combined Multi-Stage Degradation Pipeline
"""

import os
import logging
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as AF
import scipy.signal as signal
import numpy as np

logger = logging.getLogger("DeepShield.Degradations")

def add_noise(waveform: torch.Tensor, sample_rate: int = 16000, snr_db: float = 10.0, seed: int = None) -> tuple[torch.Tensor, int]:
    """Adds Gaussian additive background noise at specified Signal-to-Noise Ratio (SNR in dB)."""
    if seed is not None:
        torch.manual_seed(seed)

    signal_power = torch.mean(waveform ** 2)
    if signal_power.item() == 0.0:
        return waveform, sample_rate

    # Calculate noise power for target SNR: SNR_dB = 10 * log10(P_signal / P_noise)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)

    degraded = waveform + noise
    
    # Safe clipping to prevent out-of-range values
    degraded = torch.clamp(degraded, -1.0, 1.0)
    return degraded, sample_rate

def apply_resampling_degradation(waveform: torch.Tensor, sample_rate: int = 16000, target_sr: int = 8000) -> tuple[torch.Tensor, int]:
    """Simulates bandwidth reduction by downsampling to target_sr and upsampling back to sample_rate."""
    if target_sr >= sample_rate:
        return waveform, sample_rate

    downsampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
    upsampler = T.Resample(orig_freq=target_sr, new_freq=sample_rate)

    downsampled = downsampler(waveform)
    restored = upsampler(downsampled)

    # Ensure length matches original waveform tensor exactly
    if restored.size(-1) != waveform.size(-1):
        if restored.size(-1) < waveform.size(-1):
            restored = F.pad(restored, (0, waveform.size(-1) - restored.size(-1)))
        else:
            restored = restored[..., :waveform.size(-1)]

    return restored, sample_rate

def apply_telephone_degradation(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    low_cutoff: float = 300.0,
    high_cutoff: float = 3400.0
) -> tuple[torch.Tensor, int]:
    """Simulates G.711 / PSTN Telephone and VoIP narrowband channel filtering (300 Hz - 3400 Hz)."""
    np_wave = waveform.squeeze(0).cpu().numpy()

    # Design 4th-order Butterworth Bandpass Filter
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = min(high_cutoff / nyquist, 0.99)

    b, a = signal.butter(N=4, Wn=[low, high], btype='bandpass')
    filtered = signal.filtfilt(b, a, np_wave)

    filtered_tensor = torch.from_numpy(filtered.copy()).float().unsqueeze(0)
    
    # Resample down to 8kHz and back to 16kHz for authentic telephone codec effect
    restored_tensor, _ = apply_resampling_degradation(filtered_tensor, sample_rate=sample_rate, target_sr=8000)
    return restored_tensor, sample_rate

def apply_compression(waveform: torch.Tensor, sample_rate: int = 16000, bits: int = 8) -> tuple[torch.Tensor, int]:
    """Simulates lossy digital audio compression & uniform quantization artifacts."""
    # Quantize waveform into 2^bits discrete steps
    levels = 2 ** bits
    scaled = (waveform + 1.0) / 2.0 # [0, 1]
    quantized = torch.round(scaled * (levels - 1)) / (levels - 1)
    degraded = (quantized * 2.0) - 1.0 # [-1, 1]
    return degraded, sample_rate

def apply_pitch_shift(waveform: torch.Tensor, sample_rate: int = 16000, semitones: float = 2.0) -> tuple[torch.Tensor, int]:
    """Applies pitch modification (-2 to +2 semitones) using torchaudio or librosa."""
    try:
        # Torchaudio functional pitch shift
        shifted = AF.pitch_shift(waveform, sample_rate=sample_rate, n_steps=int(semitones))
        return shifted, sample_rate
    except Exception:
        import librosa
        np_wave = waveform.squeeze(0).cpu().numpy()
        shifted_np = librosa.effects.pitch_shift(np_wave, sr=sample_rate, n_steps=semitones)
        shifted_tensor = torch.from_numpy(shifted_np).float().unsqueeze(0)
        return shifted_tensor, sample_rate

def apply_combined_degradation(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    sequence: list[str] = None
) -> tuple[torch.Tensor, int]:
    """Applies multiple degradations in sequence (e.g. noise -> telephone -> compression)."""
    if sequence is None:
        sequence = ["noise", "telephone", "resample"]

    current_wf = waveform.clone()
    current_sr = sample_rate

    for deg_name in sequence:
        if deg_name == "noise":
            current_wf, current_sr = add_noise(current_wf, current_sr, snr_db=10.0)
        elif deg_name == "resample":
            current_wf, current_sr = apply_resampling_degradation(current_wf, current_sr, target_sr=8000)
        elif deg_name == "telephone":
            current_wf, current_sr = apply_telephone_degradation(current_wf, current_sr)
        elif deg_name == "compression":
            current_wf, current_sr = apply_compression(current_wf, current_sr, bits=8)
        elif deg_name == "pitch":
            current_wf, current_sr = apply_pitch_shift(current_wf, current_sr, semitones=2.0)

    return current_wf, current_sr

class DegradationRegistry:
    """Registry allowing selecting audio degradations by string name."""

    DEGRADATIONS = {
        "noise": add_noise,
        "resample": apply_resampling_degradation,
        "telephone": apply_telephone_degradation,
        "compression": apply_compression,
        "pitch": apply_pitch_shift,
        "combined": apply_combined_degradation
    }

    @classmethod
    def apply(cls, name: str, waveform: torch.Tensor, sample_rate: int = 16000, **kwargs) -> tuple[torch.Tensor, int]:
        if name not in cls.DEGRADATIONS:
            raise KeyError(f"Unknown degradation '{name}'. Available: {list(cls.DEGRADATIONS.keys())}")
        return cls.DEGRADATIONS[name](waveform, sample_rate, **kwargs)
