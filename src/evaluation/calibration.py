"""
DeepShield Confidence Calibration Module
Implements Temperature Scaling, Expected Calibration Error (ECE), and Reliability Assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List

class ConfidenceCalibrator:
    """
    Calibrates model logits using Temperature Scaling and measures calibration error (ECE).
    """

    def __init__(self, temperature: float = 1.25):
        """
        Args:
            temperature: Temperature scaling factor T > 0 (default: 1.25).
        """
        self.temperature = max(0.1, temperature)

    def calibrate_logits(self, logits: torch.Tensor) -> Tuple[np.ndarray, float, float]:
        """
        Applies Temperature Scaling to raw unnormalized logits.

        Args:
            logits: Tensor of shape (1, 2) or (Batch, 2).

        Returns:
            calibrated_probs: Array of calibrated probabilities [P(REAL), P(AI)].
            raw_confidence: Maximum uncalibrated softmax probability (%).
            calibrated_confidence: Maximum calibrated softmax probability (%).
        """
        # Raw uncalibrated probabilities
        raw_probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        raw_conf = float(max(raw_probs) * 100.0)

        # Temperature scaled logits
        scaled_logits = logits / self.temperature
        cal_probs = F.softmax(scaled_logits, dim=1).detach().cpu().numpy()[0]
        cal_conf = float(max(cal_probs) * 100.0)

        return cal_probs, raw_conf, cal_conf

    def calculate_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculates Expected Calibration Error (ECE) across prediction probability bins.

        Args:
            probs: Predicted probability of positive class (N,).
            labels: Ground truth binary labels (N,).
            n_bins: Number of confidence bins (default: 10).

        Returns:
            ece: Expected Calibration Error value in [0.0, 1.0].
        """
        if len(probs) == 0:
            return 0.0

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(probs)

        confidences = np.maximum(probs, 1.0 - probs)
        predictions = (probs >= 0.5).astype(int)
        accuracies = (predictions == labels).astype(float)

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return float(ece)

    def evaluate_reliability(
        self,
        raw_confidence: float,
        calibrated_confidence: float,
        ece: float = 0.035
    ) -> Dict[str, Any]:
        """
        Categorizes prediction reliability based on calibrated confidence and ECE.

        Args:
            raw_confidence: Raw confidence %.
            calibrated_confidence: Calibrated confidence %.
            ece: Expected Calibration Error.

        Returns:
            Dictionary containing reliability level, calibration status, and summary message.
        """
        diff = raw_confidence - calibrated_confidence

        if calibrated_confidence >= 85.0 and diff <= 10.0:
            reliability_level = "High Confidence"
            reason = "Model confidence is strongly calibrated with low entropy."
        elif calibrated_confidence >= 70.0:
            reliability_level = "Moderate"
            reason = "Model exhibits moderate certainty under temperature adjustment."
        else:
            reliability_level = "Uncertain"
            reason = "High prediction uncertainty detected. Require human verification."

        return {
            "temperature": self.temperature,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
            "confidence_adjustment": diff,
            "ece": ece,
            "reliability_level": reliability_level,
            "reliability_reason": reason
        }
