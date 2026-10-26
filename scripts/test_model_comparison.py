"""
DeepShield Phase 11 Model Comparison & Ablation Test Script
Verifies side-by-side evaluation of CNN Baseline vs. Advanced CRNN, runs ablation experiments, and asserts file exports.
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader_factory import create_dataloaders
from src.models.model_factory import build_model, get_device
from src.evaluation.ablation_runner import ModelComparisonRunner

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "model_comparison")

def run_comparison_test():
    print("==================================================")
    print("  DEEPSHIELD PHASE 11 MODEL COMPARISON TEST      ")
    print("==================================================")

    device = get_device()

    # 1. Load Test DataLoader
    print("\n[STEP 1] Loading Test DataLoader...")
    _, _, test_loader = create_dataloaders(config_path=CONFIG_PATH)

    # 2. Build & Verify Architectures
    print("[STEP 2] Verifying Architecture Instantiations...")
    cnn_model = build_model(model_name="cnn_baseline", config_path=CONFIG_PATH, device=device)
    crnn_model = build_model(model_name="crnn_advanced", config_path=CONFIG_PATH, device=device)

    print(f"CNN Baseline Parameters  : {sum(p.numel() for p in cnn_model.parameters()):,}")
    print(f"Advanced CRNN Parameters : {sum(p.numel() for p in crnn_model.parameters()):,}")

    # 3. Execute Comparison Runner
    print("\n[STEP 3] Running Head-to-Head Model Comparison Suite...")
    runner = ModelComparisonRunner(device=device, config_path=CONFIG_PATH)
    df_comp = runner.run_comparison(test_loader, output_dir=OUTPUT_DIR)

    print("\n--------------------------------------------------")
    print("          HEAD-TO-HEAD MODEL COMPARISON           ")
    print("--------------------------------------------------")
    print(df_comp[["model", "condition", "accuracy", "f1_score", "robustness_drop"]].to_string(index=False))
    print("--------------------------------------------------")

    # 4. Execute Ablation Studies
    print("\n[STEP 4] Executing Ablation Studies...")
    df_ablation = runner.run_ablations(test_loader, output_dir=OUTPUT_DIR)

    print("\n--------------------------------------------------")
    print("               ABLATION STUDY SUMMARY             ")
    print("--------------------------------------------------")
    print(df_ablation.to_string(index=False))
    print("--------------------------------------------------")

    # Assertions
    assert os.path.exists(os.path.join(OUTPUT_DIR, "model_comparison.csv")), "Missing model_comparison.csv"
    assert os.path.exists(os.path.join(OUTPUT_DIR, "ablation_summary.csv")), "Missing ablation_summary.csv"
    assert os.path.exists(os.path.join(OUTPUT_DIR, "robustness_comparison.png")), "Missing robustness_comparison.png"

    print("\n==================================================")
    print("   MODEL COMPARISON & ABLATION TEST PASSED!      ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_comparison_test()
    sys.exit(0 if success else 1)
