"""
DeepShield Phase 8 Evaluation Smoke Test Script
Verifies model inference, probability normalization, metric computation, generator breakdown, and file exports.
"""

import os
import sys
import torch
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader_factory import create_dataloaders
from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.evaluation.evaluator import DeepShieldEvaluator

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
CKPT_DIR = os.path.join(PROJECT_ROOT, "models_saved", "checkpoints")
TEST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "test_results")

def run_evaluation_smoke_test():
    print("==================================================")
    print("  DEEPSHIELD PHASE 8 EVALUATION SMOKE TEST        ")
    print("==================================================")

    device = get_device()

    # 1. Load DataLoaders
    print("\n[STEP 1] Loading Test DataLoader...")
    _, _, test_loader = create_dataloaders(config_path=CONFIG_PATH)

    # 2. Build Model & Load Best Checkpoint
    print("[STEP 2] Loading Best Checkpoint into CNN Model...")
    model = build_model(config_path=CONFIG_PATH, device=device)
    ckpt_mgr = CheckpointManager(checkpoint_dir=CKPT_DIR)
    
    ckpt_path = os.path.join(CKPT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print("[NOTICE] Checkpoint missing. Running baseline training first...")
        from scripts.train_baseline import main as train_main
        train_main()

    ckpt_mgr.load_checkpoint("best_model.pt", model=model, device=device)

    # 3. Instantiate Evaluator
    evaluator = DeepShieldEvaluator(model=model, device=device)

    # 4. Predict & Assert Probability Normalization
    print("\n[STEP 3] Running Test Set Inference & Probability Normalization Checks...")
    df_preds, y_true, y_pred, y_prob = evaluator.predict_loader(test_loader)

    print(f"Total Test Samples Evaluated: {len(df_preds)}")
    print(df_preds[["filename", "label", "pred_label", "prob_real", "prob_fake", "confidence"]].head())

    prob_sums = df_preds["prob_real"] + df_preds["prob_fake"]
    assert (abs(prob_sums - 1.0) < 1e-3).all(), "Probability normalization failed! (prob_real + prob_fake != 1.0)"
    print("[SUCCESS] Probability normalization verified (prob_real + prob_fake = 1.0 for all samples).")

    # 5. Run Full Evaluation Export
    print("\n[STEP 4] Computing Metrics and Saving Output Reports...")
    metrics = evaluator.evaluate(test_loader, output_dir=TEST_OUTPUT_DIR)

    # Assert output files exist
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "baseline_metrics.json")), "Missing baseline_metrics.json"
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "baseline_predictions.csv")), "Missing baseline_predictions.csv"
    assert os.path.exists(os.path.join(TEST_OUTPUT_DIR, "confusion_matrix.png")), "Missing confusion_matrix.png"

    print("\n==================================================")
    print("   EVALUATION PIPELINE SMOKE TEST PASSED!        ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_evaluation_smoke_test()
    sys.exit(0 if success else 1)
