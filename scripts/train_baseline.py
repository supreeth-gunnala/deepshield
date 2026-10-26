"""
DeepShield Baseline CNN Training Script (Phase 7)
Trains the baseline 2D CNN model on the generator-disjoint training dataset and validates per epoch.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader_factory import create_dataloaders
from src.models.model_factory import build_model, get_device
from src.training.trainer import Trainer
from src.training.checkpoint import CheckpointManager
from src.training.utils import set_seed

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")

def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    device = get_device()

    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 5)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    patience = train_cfg.get("early_stopping_patience", 5)
    ckpt_dir = os.path.join(PROJECT_ROOT, train_cfg.get("checkpoint_dir", "models_saved/checkpoints"))

    print("\n[DEEPSHIELD BASELINE TRAINING]")
    print(f"Device               : {device}")
    print(f"Epochs               : {epochs}")
    print(f"Learning Rate        : {lr}")
    print(f"Weight Decay         : {weight_decay}")
    print(f"Early Stop Patience  : {patience}")
    print(f"Checkpoint Directory : {ckpt_dir}\n")

    # 1. Load DataLoaders
    train_loader, val_loader, _ = create_dataloaders(config_path=CONFIG_PATH)

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("Train or Val DataLoader is empty. Please check metadata CSV.")

    # 2. Build Model
    model = build_model(config_path=CONFIG_PATH, device=device)

    # 3. Setup Optimizer & Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    checkpoint_manager = CheckpointManager(checkpoint_dir=ckpt_dir)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_manager=checkpoint_manager,
        early_stopping_patience=patience
    )

    # 4. Train Model
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    print(f"[SUCCESS] DeepShield Baseline CNN training finished. Best checkpoint saved to '{ckpt_dir}/best_model.pt'.")

if __name__ == "__main__":
    main()
