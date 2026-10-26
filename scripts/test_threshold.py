"""
DeepShield Test Script: Threshold Control & PR Curve Analysis Test
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.threshold_analyzer import ThresholdAnalyzer

def main():
    print("=" * 50)
    print("      DEEPSHIELD THRESHOLD ANALYZER TEST         ")
    print("=" * 50)

    analyzer = ThresholdAnalyzer(default_threshold=0.5)

    # 1. Single sample custom threshold test
    print("\n[STEP 1] Testing single-sample custom threshold classification...")
    prob_fake = 0.42
    res_05 = analyzer.evaluate_sample_threshold(prob_fake, threshold=0.5)
    res_03 = analyzer.evaluate_sample_threshold(prob_fake, threshold=0.3)

    print(f" -> P(AI) = {prob_fake:.2f}")
    print(f"    - Threshold tau = 0.50: Prediction = '{res_05['prediction_label']}'")
    print(f"    - Threshold tau = 0.30: Prediction = '{res_03['prediction_label']}'")

    assert res_05["prediction_label"] == "REAL HUMAN SPEECH"
    assert res_03["prediction_label"] == "AI-GENERATED"

    # 2. PR curve & optimal threshold computation test
    print("\n[STEP 2] Testing Precision-Recall-F1 threshold tradeoff curve...")
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    y_probs = np.array([0.91, 0.84, 0.76, 0.62, 0.48, 0.31, 0.22, 0.15, 0.55, 0.38])

    curve_res = analyzer.compute_threshold_curve(y_true, y_probs)
    print(f" -> Optimal Threshold tau*: {curve_res['optimal_threshold']:.2f}")
    print(f" -> Optimal F1-Score: {curve_res['optimal_f1']:.4f}")
    print(f" -> Generated PNG Buffer Size: {len(curve_res['figure_buffer'].getvalue()):,} bytes")

    assert 0.1 <= curve_res["optimal_threshold"] <= 0.9, "Invalid optimal threshold!"

    print("\n" + "=" * 50)
    print("    THRESHOLD ANALYZER TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
