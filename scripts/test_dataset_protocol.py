"""
DeepShield Protocol Test Script (Phase 4)
Verifies generator-disjoint dataset splitting, checks for data leakage, and asserts strict test isolation.
"""

import os
import sys
import yaml
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.splitter import GeneratorDisjointSplitter
from scripts.build_dataset_metadata import build_metadata

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")

def run_protocol_test():
    print("==================================================")
    print("  DEEPSHIELD GENERATOR-DISJOINT PROTOCOL TEST    ")
    print("==================================================")

    # 1. Load config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    metadata_csv = os.path.join(PROJECT_ROOT, ds_cfg.get("metadata_csv", "data/metadata/dataset_metadata.csv"))

    if not os.path.exists(metadata_csv):
        print(f"[NOTICE] Metadata CSV not found. Building metadata first...")
        df = build_metadata()
    else:
        df = pd.read_csv(metadata_csv)

    print(f"\n[INFO] Loaded metadata index with {len(df)} records.")

    # 2. Verify Generator-Disjoint Splitting & Leakage
    splitter = GeneratorDisjointSplitter(
        train_generators=ds_cfg.get("train_generators", ["generator_A", "generator_B"]),
        val_generators=ds_cfg.get("val_generators", ["generator_A", "generator_B"]),
        unseen_test_generators=ds_cfg.get("unseen_test_generators", ["generator_C"]),
        val_ratio=ds_cfg.get("val_split_ratio", 0.15),
        test_ratio=ds_cfg.get("test_split_ratio", 0.20),
        random_seed=cfg.get("project", {}).get("seed", 42)
    )

    verification = splitter.verify_no_leakage(df)

    print("\n--------------------------------------------------")
    print("           LEAKAGE VERIFICATION REPORT            ")
    print("--------------------------------------------------")
    print(f"Is Generator-Disjoint Split Compliant? : {verification['is_disjoint']}")
    print(f"Train Generators                       : {verification['train_generators']}")
    print(f"Validation Generators                  : {verification['val_generators']}")
    print(f"Test Generators                        : {verification['test_generators']}")
    print(f"Train vs Unseen Test Overlap           : {verification['train_unseen_overlap']}")
    print(f"Duplicate File Paths                   : {verification['duplicate_file_paths']}")
    print(f"Duplicate Filenames                    : {verification['duplicate_filenames']}")
    print("--------------------------------------------------\n")

    # 3. Protocol Assertions
    assert len(df) > 0, "Dataset metadata is empty!"
    assert verification["is_disjoint"] == True, "PROTOCOL FAILURE: Train/Val generators overlap with unseen test generator!"
    assert verification["duplicate_file_paths"] == 0, "DATA LEAKAGE: Duplicate file paths detected in dataset metadata!"

    # Ensure generator_C is present in test set and absent from train/val sets
    unseen_test_gens = ds_cfg.get("unseen_test_generators", ["generator_C"])
    for gen in unseen_test_gens:
        assert gen not in verification["train_generators"], f"Leakage: Unseen generator {gen} found in train set!"
        assert gen not in verification["val_generators"], f"Leakage: Unseen generator {gen} found in val set!"

    print("==================================================")
    print("   GENERATOR-DISJOINT PROTOCOL TEST PASSED!       ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_protocol_test()
    sys.exit(0 if success else 1)
