"""
DeepShield Robustness Experiments Script (Phase 10)
Runs clean vs degraded evaluation experiments across all 11 conditions, computes Robustness Drop, and exports CSV/PNG artifacts.
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
from src.evaluation.robustness_evaluator import RobustnessEvaluator

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")

def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    train_cfg = cfg.get("training", {})
    ckpt_dir = os.path.join(PROJECT_ROOT, train_cfg.get("checkpoint_dir", "models_saved/checkpoints"))
    output_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "robustness")

    # 1. Load DataLoaders
    _, _, test_loader = create_dataloaders(config_path=CONFIG_PATH)

    # 2. Build Model & Load Checkpoint
    model = build_model(config_path=CONFIG_PATH, device=device)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    ckpt_info = ckpt_mgr.load_checkpoint("best_model.pt", model=model, device=device)

    print("\n==================================================")
    print("  DEEPSHIELD REAL-WORLD ROBUSTNESS EXPERIMENTS    ")
    print("==================================================")
    print(f"Model Checkpoint : best_model.pt (Epoch {ckpt_info.get('epoch', 'N/A')})")
    print(f"Test Set Size    : {len(test_loader.dataset)} samples")
    print(f"Results Output   : '{output_dir}'\n")

    # 3. Run Robustness Suite
    evaluator = RobustnessEvaluator(model=model, device=device, config_path=CONFIG_PATH)
    df_summary = evaluator.run_full_robustness_suite(test_loader, output_dir=output_dir)

    print("\n--------------------------------------------------")
    print("     FINAL ROBUSTNESS & DEGRADATION REPORT MATRIX ")
    print("--------------------------------------------------")
    print(df_summary[["condition", "accuracy", "precision", "recall", "f1_score", "roc_auc", "robustness_drop", "percentage_drop"]].to_string(index=False))
    print("--------------------------------------------------\n")

    print(f"[SUCCESS] All robustness experiment artifacts saved to: '{output_dir}'")

if __name__ == "__main__":
    main()
