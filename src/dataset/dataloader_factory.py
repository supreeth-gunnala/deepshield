"""
DeepShield DataLoader Factory Module
Creates train, validation, and test PyTorch DataLoaders with custom collation and configuration.
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader
from src.dataset.audio_dataset import DeepShieldDataset

def deepshield_collate_fn(batch):
    """
    Custom collate function for DeepShield batch creation:
    Stacks feature tensors into [Batch_Size, Channels, Mels, Time_Frames] and collects metadata list.
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    meta_list = [item[2] for item in batch]

    features_tensor = torch.stack(features, dim=0) # [B, 1, 128, 94]
    labels_tensor = torch.tensor(labels, dtype=torch.long) # [B]

    return features_tensor, labels_tensor, meta_list

def create_dataloaders(config_path: str = "config/default_config.yaml") -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function returning (train_loader, val_loader, test_loader) configured via YAML.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    dl_cfg = cfg.get("dataloader", {})
    feat_cfg = cfg.get("features", {})

    metadata_path = ds_cfg.get("metadata_csv", "data/metadata/dataset_metadata.csv")
    batch_size = dl_cfg.get("batch_size", 8)
    num_workers = dl_cfg.get("num_workers", 0)
    shuffle_train = dl_cfg.get("shuffle_train", True)
    pin_memory = dl_cfg.get("pin_memory", False)
    cache_features = feat_cfg.get("cache_features", True)
    cache_dir = feat_cfg.get("cache_dir", "data/processed/features")

    # Train Dataset & DataLoader
    train_dataset = DeepShieldDataset(
        metadata_path=metadata_path,
        split="train",
        cache_features=cache_features,
        cache_dir=cache_dir,
        config_path=config_path
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=deepshield_collate_fn
    )

    # Validation Dataset & DataLoader
    val_dataset = DeepShieldDataset(
        metadata_path=metadata_path,
        split="val",
        cache_features=cache_features,
        cache_dir=cache_dir,
        config_path=config_path
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=deepshield_collate_fn
    )

    # Test Dataset & DataLoader
    test_dataset = DeepShieldDataset(
        metadata_path=metadata_path,
        split="test",
        cache_features=cache_features,
        cache_dir=cache_dir,
        config_path=config_path
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=deepshield_collate_fn
    )

    return train_loader, val_loader, test_loader
