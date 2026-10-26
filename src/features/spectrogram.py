"""
DeepShield Log-Mel Spectrogram Feature Extractor Module
Converts standard 1D time-domain waveforms into normalized 2D Log-Mel Spectrogram tensors for 2D PyTorch CNNs.
"""

import os
import yaml
import logging
import torch
import torch.nn as nn
import torchaudio.transforms as T

logger = logging.getLogger("DeepShield.SpectrogramExtractor")

class LogMelSpectrogramExtractor(nn.Module):
    """Computes Log-Mel Spectrograms from preprocessed 1D PyTorch audio waveform tensors."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        normalize: bool = True,
        config_path: str = None
    ):
        super().__init__()

        # Load parameters from config YAML if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                full_cfg = yaml.safe_load(f) or {}
            cfg = full_cfg.get("features", {})
            audio_cfg = full_cfg.get("audio", {})

            self.sample_rate = audio_cfg.get("sample_rate", sample_rate)
            self.n_mels = cfg.get("n_mels", n_mels)
            self.n_fft = cfg.get("n_fft", n_fft)
            self.hop_length = cfg.get("hop_length", hop_length)
            self.win_length = cfg.get("win_length", win_length)
            self.f_min = cfg.get("f_min", f_min)
            self.f_max = cfg.get("f_max", f_max)
            self.normalize = cfg.get("normalize", normalize)
        else:
            self.sample_rate = sample_rate
            self.n_mels = n_mels
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.win_length = win_length
            self.f_min = f_min
            self.f_max = f_max
            self.normalize = normalize

        # Torchaudio transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels,
            power=2.0
        )
        self.amp_to_db = T.AmplitudeToDB(top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: waveform tensor of shape (Channels, Time) or (Batch, Channels, Time).
        Output: Log-Mel Spectrogram tensor of shape (Channels, N_Mels, Time_Frames) or (Batch, Channels, N_Mels, Time_Frames).
        """
        if waveform is None or waveform.numel() == 0:
            raise ValueError("Input waveform tensor is empty or None.")

        # Ensure correct batch/channel dimensions
        was_single_channel = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0) # -> (1, Time)
            was_single_channel = True
        elif waveform.dim() == 2:
            pass # (1, Time) or (Channels, Time)

        # 1. Compute Mel Spectrogram (Power scale)
        mel_spec = self.mel_transform(waveform)

        # 2. Convert to Decibel (Log) scale
        log_mel_spec = self.amp_to_db(mel_spec)

        # 3. Optional Feature Normalization (Instance-level Z-Score Normalization)
        if self.normalize:
            mean = log_mel_spec.mean(dim=(-2, -1), keepdim=True)
            std = log_mel_spec.std(dim=(-2, -1), keepdim=True) + 1e-8
            log_mel_spec = (log_mel_spec - mean) / std

        if was_single_channel:
            log_mel_spec = log_mel_spec.squeeze(0)

        logger.debug(f"Extracted Log-Mel Spectrogram shape: {log_mel_spec.shape}")
        return log_mel_spec

    def extract_from_file(self, file_path: str, preprocessor=None) -> torch.Tensor:
        """Helper method: Loads audio via preprocessor and extracts Log-Mel Spectrogram."""
        if preprocessor is None:
            from src.audio.preprocessor import AudioPreprocessor
            preprocessor = AudioPreprocessor()

        waveform, _ = preprocessor.process_file(file_path)
        return self.forward(waveform)
