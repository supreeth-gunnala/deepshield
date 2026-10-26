"""
DeepShield Generator-Wise Evaluation Module
Evaluates model performance independently for each synthetic generator and real speech.
"""

import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger("DeepShield.GeneratorEvaluator")

def evaluate_by_generator(df_preds: pd.DataFrame, condition_name: str = "clean") -> pd.DataFrame:
    """
    Computes per-generator metrics (Accuracy, Precision, Recall, F1) for a given prediction DataFrame.
    """
    records = []
    generators = df_preds["generator"].unique()

    for gen in generators:
        sub_df = df_preds[df_preds["generator"] == gen]
        if len(sub_df) == 0:
            continue

        y_true = sub_df["label_id"].values
        y_pred = sub_df["pred_label_id"].values

        acc = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        records.append({
            "condition": condition_name,
            "generator": gen,
            "num_samples": int(len(sub_df)),
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4)
        })

    return pd.DataFrame(records)
