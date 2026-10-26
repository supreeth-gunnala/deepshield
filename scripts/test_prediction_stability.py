"""
DeepShield Test Script: Single-File Prediction Stability Analysis
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor
from src.models.cnn_baseline import DeepShieldCNN
from src.evaluation.single_file_stability import SingleFileStabilityAnalyzer

def main():
    print("=" * 50)
    print(" DEEPSHIELD PREDICTION STABILITY ANALYZER TEST  ")
    print("=" * 50)

    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    preprocessor = AudioPreprocessor(config_path=config_path)
    extractor = LogMelSpectrogramExtractor(config_path=config_path)
    model = DeepShieldCNN(config_path=config_path)
    model.eval()

    analyzer = SingleFileStabilityAnalyzer(
        preprocessor=preprocessor,
        extractor=extractor,
        model=model,
        device=torch.device("cpu")
    )

    sample_path = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")
    print(f"\n[STEP 1] Preprocessing sample audio: '{sample_path}'")
    waveform, sr = preprocessor.process_file(sample_path)

    print("\n[STEP 2] Running multi-variant stability analysis across 5 acoustic perturbations...")
    res = analyzer.analyze(waveform, sr)

    print(f"\n -> Original Prediction: {res['original_prediction']}")
    print(f" -> Stable Predictions: {res['matching_predictions']}/{res['total_variants']}")
    print(f" -> Prediction Flips: {res['prediction_flips']}")
    print(f" -> Stability Percentage: {res['stability_percentage']:.1f}%")
    print(f" -> Generated Figure Buffer Size: {len(res['figure_buffer'].getvalue()):,} bytes")

    print("\n[STEP 3] Variant Breakdown:")
    for r in res["results_list"]:
        print(f"    - {r['variant_name']:<25}: {r['pred_label']:<15} (P(Fake)={r['prob_fake']:.4f}, Flipped={r['is_flipped']})")

    assert res["total_variants"] == 5, "Expected 5 acoustic variants!"
    assert 0.0 <= res["stability_percentage"] <= 100.0, "Invalid stability percentage!"

    print("\n" + "=" * 50)
    print(" PREDICTION STABILITY ANALYZER TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
