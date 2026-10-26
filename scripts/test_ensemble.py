"""
DeepShield Test Script: Model Ensemble Engine Test
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
from src.models.advanced_model import DeepShieldCRNN
from src.models.ensemble import ModelEnsemble

def main():
    print("=" * 50)
    print("        DEEPSHIELD MODEL ENSEMBLE TEST          ")
    print("=" * 50)

    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    preprocessor = AudioPreprocessor(config_path=config_path)
    extractor = LogMelSpectrogramExtractor(config_path=config_path)

    # Load 2 models
    cnn = DeepShieldCNN(config_path=config_path)
    crnn = DeepShieldCRNN(config_path=config_path)

    ensemble = ModelEnsemble(models=[("CNN Baseline", cnn), ("Advanced CRNN", crnn)], weights=[0.4, 0.6])

    sample_path = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")
    print(f"\n[STEP 1] Preprocessing sample audio: '{sample_path}'")
    waveform, sr = preprocessor.process_file(sample_path)
    spec_tensor = extractor(waveform).unsqueeze(0)

    print("\n[STEP 2] Running Model Ensemble inference...")
    res = ensemble.predict(spec_tensor, device=torch.device("cpu"))

    print(f" -> Ensemble Prediction: {res['ensemble_prediction']}")
    print(f" -> P(REAL): {res['prob_real']:.4f} | P(AI): {res['prob_fake']:.4f}")
    print(f" -> Confidence: {res['confidence']:.2f}%")
    print(f" -> Agreement Rate: {res['agreement_pct']:.0f}%")
    print(f" -> Probability Std Dev: {res['prob_std']:.4f}")

    print("\n[STEP 3] Per-Model Breakdown:")
    for m in res["per_model_breakdown"]:
        print(f"    - {m['model_name']:<20}: P(AI)={m['prob_fake']:.4f}, Confidence={m['confidence']:.2f}%, Weight={m['weight']:.2f}")

    assert 0.0 <= res["prob_fake"] <= 1.0, "Invalid fused fake probability!"
    assert len(res["per_model_breakdown"]) == 2, "Expected 2 model results in breakdown!"

    print("\n" + "=" * 50)
    print("     MODEL ENSEMBLE TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
