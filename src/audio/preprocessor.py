"""
DeepShield Audio Preprocessor Module
Standardizes audio into normalized, fixed-length PyTorch Tensors for deep learning feature extraction.
"""

import os
import yaml
import logging
import torch
import torch.nn.functional as F
from src.audio.validator import AudioValidator, AudioValidationError
from src.audio.audio_loader import AudioLoader

logger = logging.getLogger("DeepShield.AudioPreprocessor")

class AudioPreprocessor:
    """End-to-end audio preprocessor: loading, mono conversion, resampling, normalization, padding/cropping."""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_duration: float = 5.0,
        normalization_type: str = "peak",
        config_path: str = None
    ):
        # Load from config YAML if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f).get("audio", {})
            self.target_sample_rate = cfg.get("sample_rate", target_sample_rate)
            self.target_duration = cfg.get("duration_seconds", target_duration)
            self.normalization_type = cfg.get("normalization_type", normalization_type)
            min_duration = cfg.get("min_duration_seconds", 0.5)
            max_duration = cfg.get("max_duration_seconds", 30.0)
            silence_threshold = cfg.get("silence_threshold_db", -50.0)
        else:
            self.target_sample_rate = target_sample_rate
            self.target_duration = target_duration
            self.normalization_type = normalization_type
            min_duration = 0.5
            max_duration = 30.0
            silence_threshold = -50.0

        self.target_samples = int(self.target_sample_rate * self.target_duration)
        self.validator = AudioValidator(
            min_duration=min_duration,
            max_duration=max_duration,
            silence_threshold_db=silence_threshold
        )
        self.loader = AudioLoader(
            target_sample_rate=self.target_sample_rate,
            validator=self.validator
        )

    def normalize_amplitude(self, waveform: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Safely normalizes waveform amplitude to prevent clipping and scale variations."""
        if self.normalization_type == "peak":
            max_val = torch.max(torch.abs(waveform))
            if max_val > eps:
                waveform = waveform / max_val
        elif self.normalization_type == "rms":
            rms = torch.sqrt(torch.mean(waveform ** 2))
            if rms > eps:
                waveform = waveform / (rms * 4.0 + eps) # Scale RMS safely within [-1, 1]
        return waveform

    def fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pads short audio with zeros or crops long audio to exactly target_samples."""
        num_samples = waveform.size(-1)

        if num_samples < self.target_samples:
            # Pad with zeros at the end
            pad_length = self.target_samples - num_samples
            waveform = F.pad(waveform, (0, pad_length), mode="constant", value=0.0)
            logger.debug(f"Padded waveform from {num_samples} to {self.target_samples} samples.")
        elif num_samples > self.target_samples:
            # Crop to target_samples from the center/start
            waveform = waveform[:, :self.target_samples]
            logger.debug(f"Cropped waveform from {num_samples} to {self.target_samples} samples.")

        return waveform

    def process_file(self, file_path: str) -> tuple[torch.Tensor, int]:
        """
        Full pipeline: Load file -> mono -> resample 16kHz -> normalize -> crop/pad to fixed duration.
        Returns (waveform, sample_rate) where waveform shape is (1, target_samples).
        """
        waveform, sr = self.loader.load_audio(file_path)
        waveform = self.normalize_amplitude(waveform)
        waveform = self.fix_length(waveform)
        return waveform, sr

    def process_waveform(self, waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        """Preprocesses an in-memory waveform tensor."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        self.validator.validate_waveform(waveform, sample_rate)

        if sample_rate != self.target_sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        waveform = self.normalize_amplitude(waveform)
        waveform = self.fix_length(waveform)
        return waveform, sample_rate
