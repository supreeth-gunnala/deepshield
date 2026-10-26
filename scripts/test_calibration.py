"""
DeepShield Test Script: Confidence Calibration & Reliability Test
"""

import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.cnn_baseline import DeepShieldCNN
from src.evaluation.calibration import ConfidenceCalibrator

def main():
    print("=" * 50)
    print("    DEEPSHIELD CONFIDENCE CALIBRATION TEST       ")
    print("=" * 50)

    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    model = DeepShieldCNN(config_path=config_path)
    model.eval()

    calibrator = ConfidenceCalibrator(temperature=1.25)

    # 1. Single sample logit calibration test
    print("\n[STEP 1] Testing single-sample Temperature Scaling...")
    sample_logits = torch.tensor([[1.2, 3.8]]) # Raw logits favor class 1
    cal_probs, raw_conf, cal_conf = calibrator.calibrate_logits(sample_logits)

    print(f" -> Raw Logits: {sample_logits.numpy()[0]}")
    print(f" -> Raw Softmax Confidence: {raw_conf:.2f}%")
    print(f" -> Calibrated Confidence (T=1.25): {cal_conf:.2f}%")
    print(f" -> Confidence Adjustment: -{raw_conf - cal_conf:.2f}%")

    assert cal_conf < raw_conf, "Temperature scaling should smooth confidence!"

    # 2. Reliability evaluation
    print("\n[STEP 2] Evaluating prediction reliability level...")
    rel_info = calibrator.evaluate_reliability(raw_conf, cal_conf, ece=0.028)
    print(f" -> Reliability Level: {rel_info['reliability_level']}")
    print(f" -> Reason: {rel_info['reliability_reason']}")

    # 3. ECE calculation test
    print("\n[STEP 3] Testing Expected Calibration Error (ECE) calculation...")
    dummy_probs = np.array([0.92, 0.85, 0.70, 0.60, 0.40, 0.20, 0.10])
    dummy_labels = np.array([1, 1, 1, 0, 0, 0, 0])
    ece = calibrator.calculate_ece(dummy_probs, dummy_labels)
    print(f" -> Expected Calibration Error (ECE): {ece:.4f}")

    assert 0.0 <= ece <= 1.0, "ECE should be in [0, 1] range!"

    print("\n" + "=" * 50)
    print(" CONFIDENCE CALIBRATION TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
