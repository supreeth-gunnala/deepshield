"""
DeepShield Evaluation Metrics Module
Computes classification metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and Generator-wise breakdowns.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger("DeepShield.Metrics")

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Computes standard binary classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    roc_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            # y_prob[:, 1] is the probability of class 1 (AI-Generated)
            roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    report_str = classification_report(
        y_true, y_pred, target_names=["REAL", "AI-GENERATED"], zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str
    }

def compute_generator_breakdown(df_predictions: pd.DataFrame) -> dict:
    """Computes accuracy, precision, recall, and F1-score independently per generator."""
    breakdown = {}
    generators = df_predictions["generator"].unique()

    for gen in generators:
        sub_df = df_predictions[df_predictions["generator"] == gen]
        if len(sub_df) == 0:
            continue

        y_true = sub_df["label_id"].values
        y_pred = sub_df["pred_label_id"].values

        acc = float(accuracy_score(y_true, y_pred))
        
        # Calculate precision/recall/f1 with zero_division protection
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        breakdown[gen] = {
            "num_samples": int(len(sub_df)),
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4)
        }

    return breakdown
