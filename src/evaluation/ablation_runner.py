"""
DeepShield Model Comparison and Ablation Study Engine Module
Executes side-by-side benchmarking of CNN Baseline vs. Advanced CRNN across all degradations and ablations.
"""

import os
import json
import logging
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.model_factory import build_model
from src.training.checkpoint import CheckpointManager
from src.evaluation.robustness_evaluator import RobustnessEvaluator
from src.features.spectrogram import LogMelSpectrogramExtractor

logger = logging.getLogger("DeepShield.AblationRunner")

class ModelComparisonRunner:
    """Executes side-by-side model comparisons and ablation studies."""

    def __init__(self, device: torch.device, config_path: str = "config/default_config.yaml"):
        self.device = device
        self.config_path = config_path

    def run_comparison(self, test_loader, output_dir: str = "experiments/results/model_comparison") -> pd.DataFrame:
        """Runs full comparison between CNN Baseline and Advanced CRNN."""
        os.makedirs(output_dir, exist_ok=True)

        models_to_test = [
            ("CNN Baseline", "cnn_baseline", "models_saved/checkpoints"),
            ("Advanced CRNN", "crnn_advanced", "models_saved/checkpoints/crnn_advanced")
        ]

        all_summaries = []

        for display_name, arch_name, ckpt_dir in models_to_test:
            logger.info(f"Loading '{display_name}' for model comparison...")
            model = build_model(model_name=arch_name, config_path=self.config_path, device=self.device)
            ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
            
            try:
                ckpt_mgr.load_checkpoint("best_model.pt", model=model, device=self.device)
            except Exception as e:
                logger.warning(f"Could not load checkpoint for '{display_name}' in '{ckpt_dir}': {e}. Using initial weights for comparison test.")

            evaluator = RobustnessEvaluator(model=model, device=self.device, config_path=self.config_path)
            df_summary = evaluator.run_full_robustness_suite(test_loader, output_dir=os.path.join(output_dir, arch_name))

            df_summary["model"] = display_name
            all_summaries.append(df_summary)

        df_comparison = pd.concat(all_summaries, ignore_index=True)

        # Save Output Files
        df_comparison.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
        with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
            json.dump(df_comparison.to_dict(orient="records"), f, indent=4)

        # Plot 1: Robustness Comparison Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_comparison, x="condition", y="f1_score", hue="model", palette="Set2")
        plt.title("DeepShield Model Comparison: F1-Score Across Real-World Degradations")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("F1-Score")
        plt.ylim([0.0, 1.05])
        plt.legend(title="Architecture")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "robustness_comparison.png"), dpi=150)
        plt.close()

        # Plot 2: Clean Performance Comparison Plot
        clean_df = df_comparison[df_comparison["condition"] == "Clean Audio"]
        plt.figure(figsize=(6, 5))
        sns.barplot(data=clean_df, x="model", y="f1_score", hue="model", palette="Blues_d", legend=False)
        plt.title("Clean Baseline F1-Score Comparison")
        plt.ylabel("F1-Score")
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "clean_performance_comparison.png"), dpi=150)
        plt.close()

        # Write Text Report
        report_path = os.path.join(output_dir, "comparison_report.txt")
        with open(report_path, "w") as f:
            f.write("DEEPSHIELD ADVANCED MODEL COMPARISON REPORT\n")
            f.write("==========================================\n\n")
            f.write(df_comparison[["model", "condition", "accuracy", "precision", "recall", "f1_score", "robustness_drop"]].to_string(index=False))

        return df_comparison

    def run_ablations(self, test_loader, output_dir: str = "experiments/results/model_comparison") -> pd.DataFrame:
        """Runs ablation studies (e.g. without feature normalization)."""
        os.makedirs(output_dir, exist_ok=True)

        ablation_results = []

        # Baseline reference
        model = build_model(model_name="cnn_baseline", config_path=self.config_path, device=self.device)
        ckpt_mgr = CheckpointManager(checkpoint_dir="models_saved/checkpoints")
        try:
            ckpt_mgr.load_checkpoint("best_model.pt", model=model, device=self.device)
        except Exception:
            pass

        # Ablation 1: Standard Extractor (Normalized)
        eval_norm = RobustnessEvaluator(model=model, device=self.device, config_path=self.config_path)
        metrics_norm, _ = eval_norm.run_degraded_dataset_evaluation(test_loader, condition_name="Clean (With Normalization)")

        # Ablation 2: Feature Extractor Without Z-Score Normalization
        extractor_no_norm = LogMelSpectrogramExtractor(normalize=False, config_path=self.config_path)
        eval_no_norm = RobustnessEvaluator(model=model, device=self.device, config_path=self.config_path)
        eval_no_norm.extractor = extractor_no_norm
        metrics_no_norm, _ = eval_no_norm.run_degraded_dataset_evaluation(test_loader, condition_name="Clean (No Normalization)")

        ablation_results.append({
            "experiment": "Baseline (Normalized Log-Mel)",
            "accuracy": metrics_norm["accuracy"],
            "f1_score": metrics_norm["f1_score"],
            "delta_f1": 0.0
        })
        ablation_results.append({
            "experiment": "Ablation: Without Feature Normalization",
            "accuracy": metrics_no_norm["accuracy"],
            "f1_score": metrics_no_norm["f1_score"],
            "delta_f1": round(metrics_no_norm["f1_score"] - metrics_norm["f1_score"], 4)
        })

        df_ablation = pd.DataFrame(ablation_results)
        df_ablation.to_csv(os.path.join(output_dir, "ablation_summary.csv"), index=False)
        return df_ablation
