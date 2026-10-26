"""
DeepShield Dataset Metadata Validator Module
Validates audio files during dataset indexing and extracts format/duration metadata.
"""

import os
import logging
import soundfile as sf
import librosa
from src.audio.validator import AudioValidator, AudioValidationError

logger = logging.getLogger("DeepShield.MetadataValidator")

class MetadataValidator:
    """Extracts technical metadata and performs integrity checks on audio dataset files."""

    def __init__(self, audio_validator: AudioValidator = None):
        self.audio_validator = audio_validator or AudioValidator()

    def inspect_file(self, file_path: str) -> dict:
        """
        Inspects an audio file and returns metadata info dict:
        {
            "file_path": ...,
            "filename": ...,
            "duration": float,
            "sample_rate": int,
            "channels": int,
            "file_format": str,
            "validation_status": "valid" | "corrupt" | "silent" | "unsupported"
        }
        """
        file_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        info = {
            "file_path": file_path,
            "filename": filename,
            "duration": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "file_format": ext.replace(".", ""),
            "validation_status": "valid"
        }

        # 1. Path & format validation
        try:
            self.audio_validator.validate_filepath(file_path)
        except AudioValidationError as e:
            info["validation_status"] = f"invalid_path: {e}"
            return info

        # 2. Extract metadata & validate audio waveform
        try:
            sf_info = sf.info(file_path)
            info["duration"] = round(sf_info.duration, 3)
            info["sample_rate"] = sf_info.samplerate
            info["channels"] = sf_info.channels
        except Exception as e_sf:
            try:
                # Librosa fallback for info
                duration = librosa.get_duration(path=file_path)
                sr = librosa.get_samplerate(file_path)
                info["duration"] = round(duration, 3)
                info["sample_rate"] = sr
                info["channels"] = 1
            except Exception as e_lib:
                info["validation_status"] = f"corrupt: {e_sf}"
                return info

        return info
