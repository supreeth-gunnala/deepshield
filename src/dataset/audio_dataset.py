"""
DeepShield PyTorch Audio Dataset Module
Loads audio metadata, preprocesses waveforms, extracts Log-Mel Spectrogram features, and supports feature caching.
"""

import os
import hashlib
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor

logger = logging.getLogger("DeepShield.Dataset")

class DeepShieldDataset(Dataset):
    """PyTorch Dataset for DeepShield audio classification & feature extraction."""

    def __init__(
        self,
        metadata_path: str,
        split: str = None,
        label: str = None,
        generator: str = None,
        preprocessor: AudioPreprocessor = None,
        feature_extractor: LogMelSpectrogramExtractor = None,
        cache_features: bool = True,
        cache_dir: str = "data/processed/features",
        config_path: str = None
    ):
        super().__init__()

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata CSV file not found at: '{metadata_path}'")

        self.df = pd.read_csv(metadata_path)

        # Apply filtering by split, label, or generator
        if split:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if label:
            self.df = self.df[self.df["label"] == label].reset_index(drop=True)
        if generator:
            self.df = self.df[self.df["generator"] == generator].reset_index(drop=True)

        self.preprocessor = preprocessor or AudioPreprocessor(config_path=config_path)
        self.feature_extractor = feature_extractor or LogMelSpectrogramExtractor(config_path=config_path)

        self.cache_features = cache_features
        self.cache_dir = cache_dir
        if self.cache_features and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(f"Initialized DeepShieldDataset: {len(self.df)} samples (split='{split}').")

    def __len__(self) -> int:
        return len(self.df)

    def _get_cache_path(self, file_path: str) -> str:
        """Generates a unique deterministic cache file path for an audio file."""
        file_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"spec_{file_hash}.pt")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict]:
        row = self.df.iloc[index]
        file_path = row["file_path"]
        label_id = int(row["label_id"])
        generator = row["generator"]
        split = row.get("split", "unknown")

        cache_path = self._get_cache_path(file_path) if self.cache_features else None

        # 1. Check feature cache
        if cache_path and os.path.exists(cache_path):
            try:
                spec_tensor = torch.load(cache_path, weights_only=True)
                sample_meta = {
                    "file_path": file_path,
                    "filename": row["filename"],
                    "label": row["label"],
                    "label_id": label_id,
                    "generator": generator,
                    "split": split,
                    "cached": True
                }
                return spec_tensor, label_id, sample_meta
            except Exception as e:
                logger.warning(f"Failed to load cache file '{cache_path}': {e}. Recomputing feature...")

        # 2. Extract feature online
        try:
            waveform, sr = self.preprocessor.process_file(file_path)
            spec_tensor = self.feature_extractor(waveform)

            # Save to cache if enabled
            if cache_path:
                torch.save(spec_tensor, cache_path)

            sample_meta = {
                "file_path": file_path,
                "filename": row["filename"],
                "label": row["label"],
                "label_id": label_id,
                "generator": generator,
                "split": split,
                "cached": False
            }
            return spec_tensor, label_id, sample_meta

        except Exception as e:
            logger.error(f"Error processing audio sample at index {index} ('{file_path}'): {e}")
            raise RuntimeError(f"Dataset item loading error for file '{file_path}': {e}")
