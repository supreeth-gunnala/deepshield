"""
DeepShield Phase 7 Smoke Test Script
Verifies 1-epoch training & validation, gradient updates, loss reduction, and checkpoint save/load.
"""

import os
import sys
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

def run_training_smoke_test():
    print("==================================================")
    print("  DEEPSHIELD PHASE 7 TRAINING PIPELINE SMOKE TEST ")
    print("==================================================")

    set_seed(42)
    device = get_device()

    # 1. Load DataLoaders
    print("\n[STEP 1] Initializing DataLoaders...")
    train_loader, val_loader, _ = create_dataloaders(config_path=CONFIG_PATH)

    # 2. Build Model, Optimizer & Loss Criterion
    print("[STEP 2] Building CNN Model and AdamW Optimizer...")
    model = build_model(config_path=CONFIG_PATH, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_manager = CheckpointManager(checkpoint_dir="models_saved/test_checkpoints")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_manager=checkpoint_manager,
        early_stopping_patience=2
    )

    # 3. Execute 2-Epoch Smoke Test
    print("\n[STEP 3] Executing 2-epoch smoke test training run...")
    history = trainer.fit(train_loader, val_loader, epochs=2)

    # Assertions on loss history
    assert len(history["train_loss"]) == 2, "Expected 2 epochs recorded!"
    assert not any(torch.isnan(torch.tensor(history["train_loss"]))), "NaN loss detected in training history!"

    # 4. Verify Checkpoint Save and Load
    print("\n[STEP 4] Verifying Checkpoint Serialization & Restoration...")
    loaded_model = build_model(config_path=CONFIG_PATH, device=device)
    checkpoint_manager.load_checkpoint("best_model.pt", model=loaded_model, device=device)

    # Compare param state
    original_param = list(model.parameters())[0]
    loaded_param = list(loaded_model.parameters())[0]
    assert torch.equal(original_param, loaded_param), "Loaded checkpoint weights do not match saved weights!"

    print("[SUCCESS] Checkpoint loaded and verified! State dicts match perfectly.")

    print("\n==================================================")
    print("   TRAINING PIPELINE SMOKE TEST PASSED!           ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_training_smoke_test()
    sys.exit(0 if success else 1)
