
# Smoke test for DeepShield CRNN training on a tiny subset

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
from torch.utils.data import DataLoader, Subset

from src.dataset.dataloader_factory import create_dataloaders
from src.models.model_factory import build_model, get_device
from src.training.trainer import Trainer
from src.training.checkpoint import CheckpointManager
from src.training.utils import set_seed


# Configuration
CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}"
    )

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    cfg = yaml.safe_load(file) or {}


# Reproducibility
seed = cfg.get("project", {}).get("seed", 42)
set_seed(seed)


# Device
device = get_device()
print(f"Project root: {PROJECT_ROOT}")
print(f"Device: {device}")
print(f"Config: {CONFIG_PATH}")


# Load dataset
train_loader_full, val_loader_full, _ = create_dataloaders(
    config_path=str(CONFIG_PATH)
)

if len(train_loader_full.dataset) == 0:
    raise RuntimeError("Training dataset is empty.")

if len(val_loader_full.dataset) == 0:
    raise RuntimeError("Validation dataset is empty.")


# Create small subsets
subset_size = 8

train_count = min(subset_size, len(train_loader_full.dataset))
val_count = min(subset_size, len(val_loader_full.dataset))

train_subset = Subset(
    train_loader_full.dataset,
    list(range(train_count))
)

val_subset = Subset(
    val_loader_full.dataset,
    list(range(val_count))
)

print(f"Training subset size: {len(train_subset)}")
print(f"Validation subset size: {len(val_subset)}")


# Create DataLoaders
batch_size = cfg.get("dataloader", {}).get("batch_size", 8)
batch_size = max(1, min(batch_size, subset_size))

train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=getattr(train_loader_full, "collate_fn", None),
)

val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=getattr(val_loader_full, "collate_fn", None),
)


# Build a fresh CRNN model
model = build_model(
    model_name="crnn_advanced",
    config_path=str(CONFIG_PATH),
    device=device,
)


# Optimizer and loss
training_cfg = cfg.get("training", {})

learning_rate = training_cfg.get("learning_rate", 0.001)
weight_decay = training_cfg.get("weight_decay", 0.0001)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)

criterion = torch.nn.CrossEntropyLoss()


# Use a dedicated checkpoint directory
CHECKPOINT_DIR = (
    PROJECT_ROOT / "smoke_test_outputs" / "checkpoints"
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_manager = CheckpointManager(
    checkpoint_dir=str(CHECKPOINT_DIR)
)


# Run a short training test
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    checkpoint_manager=checkpoint_manager,
    early_stopping_patience=5,
)

print("\nStarting smoke-test training...")
print("Epochs: 2")
print(f"Training samples: {train_count}")
print(f"Validation samples: {val_count}")

history = trainer.fit(
    train_loader,
    val_loader,
    epochs=2,
)

print("\nTraining history:")
print(history)



# Evaluate using the Trainer's existing validation method
print("\nEvaluating validation subset...")

val_loss, val_metrics = trainer.validate(val_loader)

print(f"Validation loss: {val_loss}")
print(f"Validation metrics: {val_metrics}")

print("\nSmoke test finished.")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")

