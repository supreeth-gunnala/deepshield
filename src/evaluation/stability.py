"""
DeepShield Prediction Stability Analyzer Module
Measures sample-level prediction flip rates, confidence shifts, and prediction consistency between clean and degraded audio.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("DeepShield.Stability")

class PredictionStabilityAnalyzer:
    """Analyzes prediction flip rate and confidence reduction between paired clean and degraded predictions."""

    @staticmethod
    def analyze_pair(df_clean: pd.DataFrame, df_degraded: pd.DataFrame, condition_name: str) -> dict:
        """
        Compares clean vs degraded prediction DataFrames (matched by file_path).
        Returns a dictionary of stability metrics and a paired sample DataFrame.
        """
        merged = pd.merge(
            df_clean[["file_path", "filename", "generator", "label", "label_id", "pred_label_id", "confidence", "prob_fake"]],
            df_degraded[["file_path", "pred_label_id", "confidence", "prob_fake"]],
            on="file_path",
            suffixes=("_clean", "_degraded")
        )

        if len(merged) == 0:
            return {"condition": condition_name, "num_samples": 0, "flip_rate": 0.0, "mean_conf_change": 0.0}, merged

        # Prediction flip: 1 if clean prediction != degraded prediction, 0 otherwise
        merged["flipped"] = (merged["pred_label_id_clean"] != merged["pred_label_id_degraded"]).astype(int)
        
        # Confidence change: positive if confidence decreased under degradation
        merged["conf_change"] = merged["confidence_clean"] - merged["confidence_degraded"]
        merged["fake_prob_change"] = merged["prob_fake_clean"] - merged["prob_fake_degraded"]

        flip_rate = float(merged["flipped"].mean())
        mean_conf_change = float(merged["conf_change"].mean())
        conf_variance = float(merged["confidence_degraded"].var()) if len(merged) > 1 else 0.0

        summary = {
            "condition": condition_name,
            "num_samples": int(len(merged)),
            "flip_count": int(merged["flipped"].sum()),
            "flip_rate": round(flip_rate, 4),
            "flip_percentage": round(flip_rate * 100.0, 2),
            "mean_conf_clean": round(float(merged["confidence_clean"].mean()), 4),
            "mean_conf_degraded": round(float(merged["confidence_degraded"].mean()), 4),
            "mean_conf_change": round(mean_conf_change, 4),
            "conf_variance": round(conf_variance, 4)
        }
        return summary, merged
