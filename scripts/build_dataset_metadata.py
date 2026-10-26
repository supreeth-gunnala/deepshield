"""
DeepShield Dataset Metadata Builder Script (Phase 4)
Scans raw dataset directories, validates audio files, applies generator-disjoint splitting, and exports metadata.
"""

import os
import sys
import yaml
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataset_indexer import DatasetIndexer
from src.dataset.splitter import GeneratorDisjointSplitter
from scripts.create_mock_dataset import generate_mock_dataset

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")

def build_metadata():
    print("==================================================")
    print("      DEEPSHIELD DATASET METADATA INDEXER          ")
    print("==================================================")

    # 1. Load config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    raw_dir_cfg = ds_cfg.get("raw_dir", "data/raw")
    raw_dir = raw_dir_cfg if os.path.isabs(raw_dir_cfg) else os.path.join(PROJECT_ROOT, raw_dir_cfg)
    metadata_csv = os.path.join(PROJECT_ROOT, ds_cfg.get("metadata_csv", "data/metadata/dataset_metadata.csv"))

    # 2. Check if raw dataset exists, otherwise create mock dataset for protocol testing
    indexer = DatasetIndexer()
    df = indexer.scan_directory(raw_dir)

    if df.empty:
        print(f"[NOTICE] No raw dataset found in '{raw_dir}'. Creating synthetic mock dataset...")
        generate_mock_dataset(raw_dir, num_samples_per_category=10)
        df = indexer.scan_directory(raw_dir)

    print(f"\n[INFO] Successfully indexed {len(df)} audio files.")

    # 3. Apply Generator-Disjoint Splitting
    splitter = GeneratorDisjointSplitter(
        train_generators=ds_cfg.get("train_generators", ["generator_A", "generator_B"]),
        val_generators=ds_cfg.get("val_generators", ["generator_A", "generator_B"]),
        unseen_test_generators=ds_cfg.get("unseen_test_generators", ["generator_C"]),
        val_ratio=ds_cfg.get("val_split_ratio", 0.15),
        test_ratio=ds_cfg.get("test_split_ratio", 0.20),
        random_seed=cfg.get("project", {}).get("seed", 42)
    )

    df = splitter.apply_generator_disjoint_split(df)

    # 4. Save metadata CSV
    os.makedirs(os.path.dirname(metadata_csv), exist_ok=True)
    df.to_csv(metadata_csv, index=False)
    print(f"[SUCCESS] Exported metadata CSV to: '{metadata_csv}'")

    # 5. Print summary statistics
    print("\n--------------------------------------------------")
    print("               DATASET SUMMARY STATS               ")
    print("--------------------------------------------------")
    print(f"Total Files Indexed  : {len(df)}")
    print(f"REAL Speech Files    : {len(df[df['label'] == 'real'])}")
    print(f"AI-Generated Files   : {len(df[df['label'] == 'fake'])}")
    print("\n--- GENERATOR DISTRIBUTION ---")
    print(df["generator"].value_counts().to_string())
    print("\n--- SPLIT DISTRIBUTION ---")
    print(df["split"].value_counts().to_string())
    print("\n--- CLASS PER SPLIT BREAKDOWN ---")
    print(pd.crosstab(df["split"], df["generator"]))
    print("--------------------------------------------------\n")

    return df

if __name__ == "__main__":
    build_metadata()
