"""
DeepShield Advanced CRNN Training Script (Phase 11)
Trains the CRNN model on identical dataset splits and saves checkpoints to 'models_saved/checkpoints/crnn_advanced/'.
"""

import os
import sys
import yaml
import torch

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
        cfg = yaml.safe_load(f) or {}

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    device = get_device()

    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 5)
    lr = train_cfg.get("learning_rate", 0.001)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    patience = train_cfg.get("early_stopping_patience", 5)

    # Separate checkpoint directory for advanced CRNN model
    ckpt_dir = os.path.join(PROJECT_ROOT, "models_saved", "checkpoints", "asvspoof_voice")

    print("\n==================================================")
    print("      DEEPSHIELD ADVANCED CRNN TRAINING           ")
    print("==================================================")
    print(f"Model Architecture   : DeepShieldCRNN")
    print(f"Target Device        : {device}")
    print(f"Epochs               : {epochs}")
    print(f"Learning Rate        : {lr}")
    print(f"Checkpoint Directory : {ckpt_dir}\n")

    # 1. Load DataLoaders
    train_loader, val_loader, _ = create_dataloaders(config_path=CONFIG_PATH)

    # 2. Build Advanced CRNN Model
    model = build_model(model_name="crnn_advanced", config_path=CONFIG_PATH, device=device)

    # 3. Optimizer & Loss Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_manager = CheckpointManager(checkpoint_dir=ckpt_dir)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_manager=checkpoint_manager,
        early_stopping_patience=patience
    )

    # 4. Fit Model
    history = trainer.fit(train_loader, val_loader, epochs=epochs)
    print(f"[SUCCESS] Advanced CRNN training completed. Checkpoints saved to '{ckpt_dir}'.")

if __name__ == "__main__":
    main()
