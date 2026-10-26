"""
DeepShield Evaluator Engine Module
Executes model inference, computes probabilities and metrics, and exports visual reports.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from src.evaluation.metrics import compute_classification_metrics, compute_generator_breakdown

logger = logging.getLogger("DeepShield.Evaluator")

class DeepShieldEvaluator:
    """Evaluation manager for DeepShield models."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def predict_loader(self, dataloader) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Runs batch inference over a DataLoader, returning predictions DataFrame and probability arrays."""
        records = []
        all_true = []
        all_pred = []
        all_probs = []

        with torch.no_grad():
            for batch_features, batch_labels, batch_meta in dataloader:
                batch_features = batch_features.to(self.device)
                
                # Forward pass -> raw logits
                logits = self.model(batch_features)
                
                # Apply Softmax to obtain probabilities
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                for i in range(len(batch_meta)):
                    meta = batch_meta[i]
                    p_real = float(probs[i, 0])
                    p_fake = float(probs[i, 1])
                    conf = float(np.max(probs[i]))

                    rec = {
                        "file_path": meta["file_path"],
                        "filename": meta["filename"],
                        "label": meta["label"],
                        "label_id": int(batch_labels[i].item()),
                        "pred_label_id": int(preds[i]),
                        "pred_label": "fake" if preds[i] == 1 else "real",
                        "prob_real": round(p_real, 4),
                        "prob_fake": round(p_fake, 4),
                        "confidence": round(conf, 4),
                        "generator": meta["generator"],
                        "split": meta["split"]
                    }
                    records.append(rec)
                    all_true.append(batch_labels[i].item())
                    all_pred.append(preds[i])
                    all_probs.append([p_real, p_fake])

        df_preds = pd.DataFrame(records)
        return df_preds, np.array(all_true), np.array(all_pred), np.array(all_probs)

    def evaluate(self, dataloader, output_dir: str = "experiments/results") -> dict:
        """Executes full evaluation, computes metrics & generator breakdown, and saves result artifacts."""
        os.makedirs(output_dir, exist_ok=True)

        df_preds, y_true, y_pred, y_prob = self.predict_loader(dataloader)

        if len(df_preds) == 0:
            raise ValueError("Evaluation DataLoader is empty!")

        # 1. Compute Overall Metrics
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)

        # 2. Compute Generator Breakdown
        gen_breakdown = compute_generator_breakdown(df_preds)
        metrics["generator_breakdown"] = gen_breakdown

        # 3. Save Predictions CSV
        csv_path = os.path.join(output_dir, "baseline_predictions.csv")
        df_preds.to_csv(csv_path, index=False)

        # 4. Save JSON Metrics
        json_path = os.path.join(output_dir, "baseline_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        # 5. Save Classification Report Text
        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(metrics["classification_report"])

        # 6. Plot & Save Confusion Matrix PNG
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            np.array(metrics["confusion_matrix"]),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["REAL", "AI-GENERATED"],
            yticklabels=["REAL", "AI-GENERATED"]
        )
        plt.title("DeepShield Baseline Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Ground Truth Label")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()

        # 7. Plot & Save ROC Curve PNG (if valid)
        if metrics["roc_auc"] != "N/A" and len(np.unique(y_true)) > 1:
            roc_path = os.path.join(output_dir, "roc_curve.png")
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {metrics['roc_auc']})")
            plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=150)
            plt.close()

        logger.info(f"Saved evaluation artifacts to '{output_dir}'.")
        return metrics
