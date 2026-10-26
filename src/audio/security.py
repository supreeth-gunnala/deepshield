"""
DeepShield Enterprise Security & File Integrity Module
Handles SHA-256 audio hashing, file size limit enforcement, header validation, temporary file cleanup, and privacy notices.
"""

import os
import io
import hashlib
import soundfile as sf
from typing import Dict, Any, Union, List, Tuple

class AudioSecurityEngine:
    """
    Enterprise security engine for audio validation, SHA-256 fingerprinting, and privacy enforcement.
    """

    ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    @staticmethod
    def compute_sha256(data: Union[str, os.PathLike, bytes, io.BytesIO]) -> str:
        """
        Computes SHA-256 hash fingerprint for an audio file or byte buffer.

        Args:
            data: Filepath string, Path object, bytes, or BytesIO buffer.

        Returns:
            sha256_hex: 64-character SHA-256 hex string.
        """
        hasher = hashlib.sha256()

        if isinstance(data, (str, os.PathLike)):
            if not os.path.exists(data):
                return "0" * 64
            with open(data, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
        elif isinstance(data, bytes):
            hasher.update(data)
        elif isinstance(data, io.BytesIO):
            pos = data.tell()
            data.seek(0)
            hasher.update(data.read())
            data.seek(pos)
        else:
            return "0" * 64

        return hasher.hexdigest()

    @classmethod
    def validate_file_security(
        cls,
        filepath_or_bytes: Union[str, os.PathLike, bytes],
        filename: str = "upload.wav",
        max_size_mb: float = 50.0
    ) -> Dict[str, Any]:
        """
        Validates audio file size, extension, header integrity, and security status.

        Args:
            filepath_or_bytes: Input audio filepath or bytes data.
            filename: Original filename.
            max_size_mb: Maximum allowed file size in MB (default: 50.0 MB).

        Returns:
            Dictionary containing validation status, file size MB, SHA-256 hash, and error message if any.
        """
        ext = os.path.splitext(filename)[1].lower()
        sha256_hash = cls.compute_sha256(filepath_or_bytes)

        # Check Extension
        if ext not in cls.ALLOWED_EXTENSIONS:
            return {
                "is_valid": False,
                "error_message": f"Unsupported file extension '{ext}'. Allowed extensions: {list(cls.ALLOWED_EXTENSIONS)}",
                "file_size_mb": 0.0,
                "sha256_hash": sha256_hash
            }

        # Check File Size
        if isinstance(filepath_or_bytes, (str, os.PathLike)):
            file_size_bytes = os.path.getsize(filepath_or_bytes)
        else:
            file_size_bytes = len(filepath_or_bytes)

        file_size_mb = float(file_size_bytes / (1024 * 1024))
        if file_size_mb > max_size_mb:
            return {
                "is_valid": False,
                "error_message": f"File size ({file_size_mb:.2f} MB) exceeds maximum security limit of {max_size_mb:.1f} MB.",
                "file_size_mb": file_size_mb,
                "sha256_hash": sha256_hash
            }

        # Validate Header Integrity via SoundFile
        try:
            if isinstance(filepath_or_bytes, (str, os.PathLike)):
                info = sf.info(filepath_or_bytes)
            else:
                buf = io.BytesIO(filepath_or_bytes)
                info = sf.info(buf)

            if info.duration <= 0.01:
                return {
                    "is_valid": False,
                    "error_message": "Audio duration is zero or corrupted.",
                    "file_size_mb": file_size_mb,
                    "sha256_hash": sha256_hash
                }

        except Exception as e:
            return {
                "is_valid": False,
                "error_message": f"Corrupted audio header or invalid encoding: {e}",
                "file_size_mb": file_size_mb,
                "sha256_hash": sha256_hash
            }

        return {
            "is_valid": True,
            "error_message": "File passed all security and header integrity checks.",
            "file_size_mb": file_size_mb,
            "sha256_hash": sha256_hash,
            "duration": float(info.duration),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels)
        }

    @staticmethod
    def cleanup_temp_files(temp_paths: List[str]):
        """Safely removes temporary files from disk."""
        for path in temp_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    @staticmethod
    def get_privacy_notice() -> str:
        """Returns standard privacy notice text."""
        return (
            "🔒 **Privacy & Data Security Notice:** Uploaded audio is processed in ephemeral memory "
            "for deepfake model inference. DeepShield does not permanently store audio files or share data."
        )
