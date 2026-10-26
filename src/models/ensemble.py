"""
DeepShield Model Ensemble Engine
Combines predictions from multiple trained models (e.g. 2D CNN Baseline and Advanced CRNN)
using weighted probability fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class ModelEnsemble:
    """
    Combines predictions from multiple DeepShield PyTorch models using weighted probability fusion.
    """

    def __init__(self, models: List[Tuple[str, nn.Module]], weights: Optional[List[float]] = None):
        """
        Args:
            models: List of tuples (model_name, nn.Module instance).
            weights: Optional list of fusion weights summing to 1.0. If None, uses uniform weights.
        """
        self.models = models
        for _, m in self.models:
            m.eval()

        n_models = len(models)
        if weights is None:
            self.weights = [1.0 / n_models] * n_models
        else:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]

    def predict(self, input_tensor: torch.Tensor, device: torch.device) -> Dict[str, Any]:
        """
        Runs inference across all ensemble models and fuses probabilities.

        Args:
            input_tensor: Tensor of shape (1, 1, Mel_Bins, Time_Frames).
            device: Execution hardware device (CPU/CUDA).

        Returns:
            Dictionary containing fused probabilities, ensemble prediction, agreement rate, and per-model breakdown.
        """
        x = input_tensor.to(device)
        model_results = []
        fused_probs = np.zeros(2, dtype=np.float64)

        with torch.no_grad():
            for idx, (name, model) in enumerate(self.models):
                logits = model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                w = self.weights[idx]

                fused_probs += w * probs
                pred_label = "AI-GENERATED" if probs[1] >= 0.5 else "REAL"

                model_results.append({
                    "model_name": name,
                    "prob_real": float(probs[0]),
                    "prob_fake": float(probs[1]),
                    "pred_label": pred_label,
                    "confidence": float(max(probs) * 100.0),
                    "weight": float(w)
                })

        p_real = float(fused_probs[0])
        p_fake = float(fused_probs[1])
        ensemble_pred = "AI-GENERATED" if p_fake >= 0.5 else "REAL HUMAN SPEECH"
        confidence = float(max(p_real, p_fake) * 100.0)

        # Model Agreement Rate
        predictions = [r["pred_label"] for r in model_results]
        matching = sum(1 for p in predictions if (p == "AI-GENERATED" if p_fake >= 0.5 else p == "REAL"))
        agreement_pct = float((matching / len(predictions)) * 100.0)

        # Variance across model fake probabilities
        fake_probs_list = [r["prob_fake"] for r in model_results]
        prob_std = float(np.std(fake_probs_list))

        return {
            "ensemble_prediction": ensemble_pred,
            "prob_real": p_real,
            "prob_fake": p_fake,
            "confidence": confidence,
            "agreement_pct": agreement_pct,
            "prob_std": prob_std,
            "per_model_breakdown": model_results
        }
