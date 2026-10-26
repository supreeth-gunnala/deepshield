"""
DeepShield Phase 5 Test Script
Tests PyTorch Dataset & DataLoader pipeline, verifies batch dimensions, checks for NaNs/Infs, and validates caching.
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.audio_dataset import DeepShieldDataset
from src.dataset.dataloader_factory import create_dataloaders

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
METADATA_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "dataset_metadata.csv")

def run_dataset_loader_test():
    print("==================================================")
    print("   DEEPSHIELD PHASE 5 DATASET LOADER TEST        ")
    print("==================================================")

    if not os.path.exists(METADATA_PATH):
        print("[NOTICE] Metadata CSV missing. Running Phase 4 build script first...")
        from scripts.build_dataset_metadata import build_metadata
        build_metadata()

    # 1. Test Dataset Instance Creation & Filtering
    print(f"\n[STEP 1] Instantiating DeepShieldDataset splits...")
    train_ds = DeepShieldDataset(METADATA_PATH, split="train", config_path=CONFIG_PATH)
    val_ds = DeepShieldDataset(METADATA_PATH, split="val", config_path=CONFIG_PATH)
    test_ds = DeepShieldDataset(METADATA_PATH, split="test", config_path=CONFIG_PATH)

    print(f"Dataset Lengths -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 2. Test Single Item Fetching
    print(f"\n[STEP 2] Fetching sample item from Train dataset...")
    spec_tensor, label_id, meta = train_ds[0]

    print("--------------------------------------------------")
    print("               SINGLE SAMPLE STATS                ")
    print("--------------------------------------------------")
    print(f"Feature Tensor Shape : {spec_tensor.shape} (C, Mels, Frames)")
    print(f"Feature Tensor Dtype : {spec_tensor.dtype}")
    print(f"Label ID             : {label_id} ({meta['label'].upper()})")
    print(f"Generator Identity   : {meta['generator']}")
    print(f"File Path            : {meta['file_path']}")
    print(f"Feature Cached?      : {meta['cached']}")
    print("--------------------------------------------------")

    # 3. Test DataLoaders and Batching
    print(f"\n[STEP 3] Testing DataLoaders & Batch Collation...")
    train_loader, val_loader, test_loader = create_dataloaders(config_path=CONFIG_PATH)

    for split_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        if len(loader) == 0:
            continue

        batch_features, batch_labels, batch_meta = next(iter(loader))

        print(f"\n--- {split_name} DataLoader Batch Output ---")
        print(f"Batch Feature Tensor Shape : {batch_features.shape} (Batch, C, Mels, Frames)")
        print(f"Batch Label Tensor Shape   : {batch_labels.shape} (Batch)")
        print(f"Batch Label IDs            : {batch_labels.tolist()}")
        print(f"Batch Generators           : {[m['generator'] for m in batch_meta]}")

        # Assertions
        assert not torch.isnan(batch_features).any(), f"NaN values detected in {split_name} batch!"
        assert not torch.isinf(batch_features).any(), f"Infinite values detected in {split_name} batch!"
        assert batch_features.dim() == 4, f"Expected 4D batch tensor [B, C, F, T], got {batch_features.dim()}D"
        assert batch_features.shape[1] == 1, f"Expected 1 channel, got {batch_features.shape[1]}"
        assert batch_features.shape[2] == 128, f"Expected 128 Mel bins, got {batch_features.shape[2]}"

    print("\n==================================================")
    print("   DATASET LOADER & FEATURE PIPELINE PASSED!      ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_dataset_loader_test()
    sys.exit(0 if success else 1)
