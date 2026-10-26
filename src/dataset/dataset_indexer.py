"""
DeepShield Dataset Indexer Module
Recursively scans raw audio dataset directories and builds a consolidated dataset index DataFrame.
"""

import os
import logging
import pandas as pd
from src.dataset.metadata_validator import MetadataValidator

logger = logging.getLogger("DeepShield.DatasetIndexer")

class DatasetIndexer:
    """Recursively scans directory trees to build audio dataset metadata tables."""

    def __init__(self, metadata_validator: MetadataValidator = None):
        self.validator = metadata_validator or MetadataValidator()

    def scan_directory(self, raw_dir: str) -> pd.DataFrame:
        """
        Scans raw_dir recursively. Expected directory structure:
        data/raw/real/ -> label='real', generator='human'
        data/raw/fake/generator_A/ -> label='fake', generator='generator_A'
        data/raw/fake/generator_B/ -> label='fake', generator='generator_B'
        """
        raw_dir = os.path.abspath(raw_dir)
        records = []

        if not os.path.exists(raw_dir):
            logger.warning(f"Raw dataset directory '{raw_dir}' does not exist.")
            return pd.DataFrame()

        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in MetadataValidator().audio_validator.SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)

                    # Determine label and generator based on directory hierarchy
                    rel_path = os.path.relpath(file_path, raw_dir)
                    parts = rel_path.split(os.sep)
                    top_folder = parts[0].lower() if parts else ""

                    if top_folder == "real_voice":
                        label = "real"
                        label_id = 0
                        generator = "human"
                    elif top_folder == "fake_voice":
                        label = "fake"
                        label_id = 1
                        generator = "asvspoof_eval"
                    elif top_folder == "real_bg":
                        label = "background_noise"
                        label_id = -1
                        generator = "environmental_noise"
                    elif top_folder == "fake_bg":
                        label = "background_noise"
                        label_id = -1
                        generator = "synthetic_ambient_noise"
                    elif top_folder == "real":
                        label = "real"
                        label_id = 0
                        generator = "human"
                    elif top_folder == "fake":
                        label = "fake"
                        label_id = 1
                        generator = parts[1] if len(parts) >= 3 else "unknown"
                    else:
                        label = "unknown"
                        label_id = -1
                        generator = "unknown"

                    # Inspect file metadata & validation status
                    meta = self.validator.inspect_file(file_path)
                    meta["label"] = label
                    meta["label_id"] = label_id
                    meta["generator"] = generator
                    meta["dataset_source"] = parts[0] if parts else "unknown"
                    meta["split"] = "unassigned"

                    records.append(meta)

        df = pd.DataFrame(records)
        logger.info(f"Indexed {len(df)} files from '{raw_dir}'.")
        return df
