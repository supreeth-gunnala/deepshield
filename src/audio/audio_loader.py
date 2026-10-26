"""
DeepShield Audio Loader Module
Loads audio from disk, handles multi-channel to mono conversion, and resamples to target sample rate.
"""

import os
import logging
import torch
import soundfile as sf
import librosa
import torchaudio
import torchaudio.transforms as T
from src.audio.validator import AudioValidator, AudioValidationError

logger = logging.getLogger("DeepShield.AudioLoader")

class AudioLoader:
    """Loads audio files reliably into PyTorch Tensors, handling mono conversion and resampling."""

    def __init__(self, target_sample_rate: int = 16000, validator: AudioValidator = None):
        self.target_sample_rate = target_sample_rate
        self.validator = validator or AudioValidator()

    def load_audio(self, file_path: str) -> tuple[torch.Tensor, int]:
        """
        Loads an audio file and returns (waveform, sample_rate).
        Output waveform tensor shape: (1, num_samples) on CPU.
        """
        self.validator.validate_filepath(file_path)

        waveform = None
        sr = None

        # Strategy 1: Attempt direct loading with SoundFile (Fastest for WAV, FLAC, OGG)
        try:
            data, sr = sf.read(file_path, dtype='float32')
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0) # (1, N)
            else:
                waveform = torch.from_numpy(data.T) # (C, N)
            logger.debug(f"Loaded '{file_path}' via SoundFile. SR={sr}, Shape={waveform.shape}")
        except Exception as e_sf:
            logger.debug(f"SoundFile loader failed for '{file_path}': {e_sf}. Attempting Librosa/Torchaudio...")

            # Strategy 2: Fallback to Librosa (Supports MP3, AAC, etc.)
            try:
                y, sr = librosa.load(file_path, sr=None, mono=False)
                if y.ndim == 1:
                    waveform = torch.from_numpy(y).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(y)
                logger.debug(f"Loaded '{file_path}' via Librosa fallback. SR={sr}, Shape={waveform.shape}")
            except Exception as e_librosa:
                # Strategy 3: Torchaudio fallback
                try:
                    waveform, sr = torchaudio.load(file_path)
                    logger.debug(f"Loaded '{file_path}' via Torchaudio fallback. SR={sr}, Shape={waveform.shape}")
                except Exception as e_torch:
                    raise AudioValidationError(
                        f"Failed to load audio file '{file_path}' with all backends (SoundFile, Librosa, Torchaudio): {e_torch}"
                    )

        # Convert multi-channel audio to mono by averaging channels
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Validate loaded raw waveform
        self.validator.validate_waveform(waveform, sr)

        # Resample to target_sample_rate if necessary
        if sr != self.target_sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            sr = self.target_sample_rate
            logger.debug(f"Resampled audio to {sr} Hz.")

        return waveform, sr
