"""
DeepShield Unseen Generator Evaluation Script (Phase 10)
Evaluates out-of-distribution generalization on unseen AI speech generators (Generator C).
"""

import os
import sys
import yaml
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.dataloader_factory import create_dataloaders
from src.dataset.splitter import GeneratorDisjointSplitter
from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.evaluation.evaluator import DeepShieldEvaluator

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")

def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("dataset", {})

    ckpt_dir = os.path.join(PROJECT_ROOT, train_cfg.get("checkpoint_dir", "models_saved/checkpoints"))
    output_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "unseen_generator")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load DataLoaders
    _, _, test_loader = create_dataloaders(config_path=CONFIG_PATH)

    # 2. Verify Generator Leakage Compliance
    df_meta = test_loader.dataset.df
    full_meta = pd.read_csv(os.path.join(PROJECT_ROOT, ds_cfg.get("metadata_csv", "data/metadata/dataset_metadata.csv")))

    splitter = GeneratorDisjointSplitter(
        train_generators=ds_cfg.get("train_generators", ["generator_A", "generator_B"]),
        val_generators=ds_cfg.get("val_generators", ["generator_A", "generator_B"]),
        unseen_test_generators=ds_cfg.get("unseen_test_generators", ["generator_C"]),
        val_ratio=ds_cfg.get("val_split_ratio", 0.15),
        test_ratio=ds_cfg.get("test_split_ratio", 0.20),
        random_seed=cfg.get("project", {}).get("seed", 42)
    )
    verification = splitter.verify_no_leakage(full_meta)

    # 3. Build Model & Load Checkpoint
    model = build_model(config_path=CONFIG_PATH, device=device)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    ckpt_info = ckpt_mgr.load_checkpoint("best_model.pt", model=model, device=device)

    # 4. Evaluate Unseen Generator Test Set
    evaluator = DeepShieldEvaluator(model=model, device=device)
    df_preds, y_true, y_pred, y_prob = evaluator.predict_loader(test_loader)
    metrics = evaluator.evaluate(test_loader, output_dir=output_dir)

    print("\n==================================================")
    print("   UNSEEN-GENERATOR OUT-OF-DISTRIBUTION REPORT    ")
    print("==================================================")
    print(f"Train Generators        : {verification['train_generators']}")
    print(f"Unseen Test Generators  : {verification['test_generators']}")
    print(f"Generator Leakage Check : {'PASSED (Disjoint)' if verification['is_disjoint'] else 'FAILED'}")
    print(f"Total Test Samples      : {len(df_preds)}")
    print(f"Unseen Generator F1     : {metrics['f1_score']:.4f}")
    print(f"Unseen Generator Accuracy: {metrics['accuracy']:.4f}")

    print("\n--------------------------------------------------")
    print("         PER-GENERATOR ACCURACY BREAKDOWN         ")
    print("--------------------------------------------------")
    gen_df = pd.DataFrame(metrics["generator_breakdown"]).T
    print(gen_df.to_string())
    print("--------------------------------------------------\n")

    # Write Text Report
    report_path = os.path.join(output_dir, "unseen_generator_report.txt")
    with open(report_path, "w") as f:
        f.write("DEEPSHIELD UNSEEN GENERATOR REPORT\n")
        f.write(f"Train Generators: {verification['train_generators']}\n")
        f.write(f"Unseen Test Generators: {verification['test_generators']}\n")
        f.write(f"Overall Test Accuracy: {metrics['accuracy']}\n")
        f.write(f"Overall Test F1: {metrics['f1_score']}\n\n")
        f.write(gen_df.to_string())

    print(f"[SUCCESS] Exported unseen generator report to '{output_dir}'.")

if __name__ == "__main__":
    main()
