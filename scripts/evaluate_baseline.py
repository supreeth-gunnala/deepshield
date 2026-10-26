"""
DeepShield Baseline Evaluation Script (Phase 8)
Evaluates trained CNN model checkpoint on the test set and prints generator-wise performance reports.
"""

import os
import sys
import yaml
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader_factory import create_dataloaders
from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.evaluation.evaluator import DeepShieldEvaluator

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")

def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    train_cfg = cfg.get("training", {})
    ckpt_dir = os.path.join(PROJECT_ROOT, train_cfg.get("checkpoint_dir", "models_saved/checkpoints"))
    output_dir = os.path.join(PROJECT_ROOT, "experiments", "baseline_results")

    # 1. Load DataLoaders
    _, _, test_loader = create_dataloaders(config_path=CONFIG_PATH)

    if len(test_loader) == 0:
        raise RuntimeError("Test DataLoader is empty!")

    # 2. Build Model & Load Best Checkpoint
    model = build_model(config_path=CONFIG_PATH, device=device)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    ckpt_info = ckpt_mgr.load_checkpoint("best_model.pt", model=model, device=device)

    # 3. Instantiate Evaluator & Run Evaluation
    evaluator = DeepShieldEvaluator(model=model, device=device)
    metrics = evaluator.evaluate(test_loader, output_dir=output_dir)

    # 4. Print Summary Report
    print("\n==================================================")
    print("      DEEPSHIELD BASELINE EVALUATION REPORT       ")
    print("==================================================")
    print(f"Model Architecture  : DeepShieldCNN")
    print(f"Checkpoint Loaded   : models_saved/checkpoints/best_model.pt (Epoch {ckpt_info.get('epoch', 'N/A')})")
    print(f"Test Set Size       : {len(test_loader.dataset)} samples")
    print(f"Overall Accuracy    : {metrics['accuracy']:.4f}")
    print(f"Precision           : {metrics['precision']:.4f}")
    print(f"Recall              : {metrics['recall']:.4f}")
    print(f"F1-Score            : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC             : {metrics['roc_auc']}")

    print("\n--------------------------------------------------")
    print("         PER-GENERATOR PERFORMANCE BREAKDOWN       ")
    print("--------------------------------------------------")
    gen_df = pd.DataFrame(metrics["generator_breakdown"]).T
    print(gen_df.to_string())

    print("\n--------------------------------------------------")
    print("              CLASSIFICATION REPORT               ")
    print("--------------------------------------------------")
    print(metrics["classification_report"])

    print("==================================================")
    print(f"All evaluation artifacts exported to: '{output_dir}'")
    print("==================================================\n")

if __name__ == "__main__":
    main()
