"""
DeepShield Phase 10 Robustness Pipeline Test Script
Verifies multi-degradation processing, Robustness Drop calculations, prediction stability checks, and artifact exports.
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader_factory import create_dataloaders
from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.evaluation.robustness_evaluator import RobustnessEvaluator

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
CKPT_DIR = os.path.join(PROJECT_ROOT, "models_saved", "checkpoints")
TEST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "test_robustness_results")

def run_robustness_test():
    print("==================================================")
    print("  DEEPSHIELD PHASE 10 ROBUSTNESS PIPELINE TEST   ")
    print("==================================================")

    device = get_device()

    # 1. Load DataLoaders
    print("\n[STEP 1] Loading Test DataLoader...")
    _, _, test_loader = create_dataloaders(config_path=CONFIG_PATH)

    # 2. Build Model & Load Checkpoint
    print("[STEP 2] Loading Best Checkpoint into CNN Model...")
    model = build_model(config_path=CONFIG_PATH, device=device)
    ckpt_mgr = CheckpointManager(checkpoint_dir=CKPT_DIR)
    ckpt_mgr.load_checkpoint("best_model.pt", model=model, device=device)

    # 3. Instantiate Robustness Evaluator
    evaluator = RobustnessEvaluator(model=model, device=device, config_path=CONFIG_PATH)

    # 4. Run Full Robustness Test Suite
    print("\n[STEP 3] Running Robustness Degradation Experiments & Stability Checks...")
    df_summary = evaluator.run_full_robustness_suite(test_loader, output_dir=TEST_OUTPUT_DIR)

    print("\n--------------------------------------------------")
    print("          ROBUSTNESS SUMMARY MATRIX               ")
    print("--------------------------------------------------")
    print(df_summary[["condition", "accuracy", "f1_score", "robustness_drop", "percentage_drop", "flip_rate"]].to_string())
    print("--------------------------------------------------")

    # Assertions
    assert len(df_summary) > 0, "Robustness summary table is empty!"
    assert "Clean Audio" in df_summary["condition"].values, "Clean baseline condition missing!"
    assert not df_summary["f1_score"].isnull().any(), "NaN detected in F1 scores!"
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "robustness_summary.csv")), "Missing robustness_summary.csv"
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "clean_vs_degraded_f1.png")), "Missing clean_vs_degraded_f1.png"

    print("\n==================================================")
    print("   ROBUSTNESS PIPELINE SMOKE TEST PASSED!        ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_robustness_test()
    sys.exit(0 if success else 1)
