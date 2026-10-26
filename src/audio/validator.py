"""
DeepShield Audio Validator Module
Handles format, corruption, duration, and silence validation.
"""

import os
import logging
import torch

logger = logging.getLogger("DeepShield.AudioValidator")

class AudioValidationError(Exception):
    """Custom exception raised for invalid audio files."""
    pass

class AudioValidator:
    """Validates audio files against size, duration, format, and silence constraints."""

    SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

    def __init__(self, min_duration: float = 0.5, max_duration: float = 30.0, silence_threshold_db: float = -50.0):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.silence_threshold_db = silence_threshold_db

    def validate_filepath(self, file_path: str) -> None:
        """Validates file existence, format, and size."""
        if not os.path.exists(file_path):
            raise AudioValidationError(f"File not found at path: '{file_path}'")

        if os.path.getsize(file_path) == 0:
            raise AudioValidationError(f"Audio file is empty (0 bytes): '{file_path}'")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise AudioValidationError(
                f"Unsupported audio format '{ext}'. Supported formats: {sorted(list(self.SUPPORTED_EXTENSIONS))}"
            )

    def validate_waveform(self, waveform: torch.Tensor, sample_rate: int) -> None:
        """Validates tensor shape, non-emptiness, duration, and silence."""
        if waveform is None or waveform.numel() == 0:
            raise AudioValidationError("Audio waveform tensor is empty or None.")

        num_samples = waveform.size(-1)
        duration = num_samples / float(sample_rate)

        if duration < self.min_duration:
            raise AudioValidationError(
                f"Audio duration ({duration:.2f}s) is shorter than minimum required ({self.min_duration}s)."
            )

        if duration > self.max_duration:
            raise AudioValidationError(
                f"Audio duration ({duration:.2f}s) exceeds maximum allowed ({self.max_duration}s)."
            )

        # Check for absolute zero or complete silence
        rms_energy = torch.sqrt(torch.mean(waveform ** 2))
        if rms_energy.item() == 0.0:
            raise AudioValidationError("Audio content contains absolute silence (zero energy).")

        rms_db = 20 * torch.log10(rms_energy + 1e-9).item()
        if rms_db < self.silence_threshold_db:
            raise AudioValidationError(
                f"Audio level ({rms_db:.1f} dB) is below silence threshold ({self.silence_threshold_db} dB)."
            )

        logger.debug(f"Validation passed: Duration={duration:.2f}s, RMS={rms_db:.1f}dB")
