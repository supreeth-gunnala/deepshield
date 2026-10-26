import os
import sys
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm

# Project root and path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project utilities
from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.dataset.dataloader_factory import create_dataloaders
from src.evaluation.metrics import compute_classification_metrics

def main():
    # Load configuration
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}

    device = get_device()

    # Build the model (same architecture used for training)
    model = build_model(model_name="crnn_advanced", config_path=CONFIG_PATH, device=device)

    # Load the best checkpoint from the new checkpoint directory
    ckpt_dir = os.path.join(PROJECT_ROOT, "models_saved", "checkpoints", "asvspoof_voice")
    ckpt_manager = CheckpointManager(checkpoint_dir=ckpt_dir)
    ckpt_manager.load_checkpoint(filename="best_model.pt", model=model, device=device)

    # Put model in evaluation mode
    model.eval()
    torch.set_grad_enabled(False)

    # Get test loader (third returned loader)
    _, _, test_loader = create_dataloaders(config_path=CONFIG_PATH)

    all_labels = []
    all_preds = []
    all_probs = []

    for batch in tqdm(test_loader, desc="Evaluating test set"):
        features, labels, _ = batch
        features = features.to(device)
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    # Concatenate results
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    # Compute metrics (ROC‑AUC expects probability for class 1)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    # Save metrics JSON
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "asvspoof_voice_test_metrics.json")
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- Test Set Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Metrics saved to: {results_path}")

if __name__ == "__main__":
    main()
