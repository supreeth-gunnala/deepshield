"""
DeepShield Threshold Analyzer Module
Analyzes classification performance across adjustable decision thresholds and generates Precision-Recall-F1 curves.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

class ThresholdAnalyzer:
    """
    Computes precision, recall, and F1 across decision thresholds and generates tradeoff plots.
    """

    def __init__(self, default_threshold: float = 0.5):
        self.default_threshold = default_threshold

    def evaluate_sample_threshold(self, prob_fake: float, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Classifies a single sample using a custom decision threshold.

        Args:
            prob_fake: Softmax probability of AI-generated class.
            threshold: Decision threshold in [0.0, 1.0].

        Returns:
            Dictionary containing prediction label, threshold, and confidence distance.
        """
        pred_label = "AI-GENERATED" if prob_fake >= threshold else "REAL HUMAN SPEECH"
        distance_from_threshold = abs(prob_fake - threshold)

        return {
            "threshold": threshold,
            "prob_fake": prob_fake,
            "prediction_label": pred_label,
            "distance_from_threshold": distance_from_threshold,
            "margin_certainty": "High" if distance_from_threshold >= 0.20 else "Borderline"
        }

    def compute_threshold_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        thresholds: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Computes precision, recall, and F1 across a range of decision thresholds.

        Args:
            y_true: Ground truth binary labels (0 for REAL, 1 for AI-GENERATED).
            y_probs: Model output probabilities of AI-GENERATED class.
            thresholds: Optional array of thresholds to evaluate.

        Returns:
            Dictionary containing optimal threshold, threshold grid, metric arrays, and PR curve figure.
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 17)

        precisions = []
        recalls = []
        f1s = []

        for tau in thresholds:
            y_pred = (y_probs >= tau).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)

        best_idx = int(np.argmax(f1s))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1s[best_idx])

        # Plot figure
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(thresholds, precisions, "b-o", label="Precision", linewidth=1.8, markersize=4)
        ax.plot(thresholds, recalls, "r-s", label="Recall", linewidth=1.8, markersize=4)
        ax.plot(thresholds, f1s, "g-^", label="F1-Score", linewidth=2.0, markersize=5)

        ax.axvline(x=best_threshold, color="purple", linestyle="--", linewidth=1.2, label=f"Optimal τ = {best_threshold:.2f}")
        ax.set_title("Decision Threshold Tradeoff (Precision / Recall / F1)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Decision Threshold (τ)")
        ax.set_ylabel("Metric Value")
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=9)
        plt.tight_layout()

        # Buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)

        return {
            "optimal_threshold": best_threshold,
            "optimal_f1": best_f1,
            "thresholds": thresholds.tolist(),
            "precisions": precisions.tolist(),
            "recalls": recalls.tolist(),
            "f1_scores": f1s.tolist(),
            "figure_buffer": buf
        }
