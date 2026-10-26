"""
DeepShield Test Script: Experiment Registry System Test
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.registry import ExperimentRegistry

def main():
    print("=" * 50)
    print("   DEEPSHIELD EXPERIMENT REGISTRY SYSTEM TEST    ")
    print("=" * 50)

    registry = ExperimentRegistry(registry_dir=os.path.join(PROJECT_ROOT, "experiments", "registry"))

    print("\n[STEP 1] Registering test experiment entry...")
    exp_id = registry.register_experiment({
        "model_architecture": "Advanced CRNN",
        "train_generators": "generator_A, generator_B",
        "unseen_generator": "generator_C",
        "clean_f1": 0.9420,
        "noise_f1": 0.8350,
        "compression_f1": 0.8610,
        "degraded_f1_avg": 0.7980,
        "robustness_drop": 0.1440,
        "roc_auc": 0.9710,
        "status": "Test Run Verified"
    })

    print(f" -> Generated Experiment ID: {exp_id}")

    print("\n[STEP 2] Fetching experiment history...")
    df_history = registry.get_history()
    print(f" -> Total Experiments Logged: {len(df_history)}")
    print(df_history[["experiment_id", "model_architecture", "clean_f1", "robustness_drop", "status"]].tail(2))

    assert len(df_history) >= 2, "Expected at least 2 entries in registry!"
    assert exp_id.startswith("EXP-"), "Experiment ID format invalid!"

    print("\n" + "=" * 50)
    print(" EXPERIMENT REGISTRY TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
