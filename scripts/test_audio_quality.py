"""
DeepShield Test Script: Audio Quality & Deepfake Risk Analysis
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.preprocessor import AudioPreprocessor
from src.audio.quality_analyzer import AudioQualityAnalyzer

def main():
    print("=" * 50)
    print("   DEEPSHIELD AUDIO QUALITY RISK ANALYZER TEST   ")
    print("=" * 50)

    preprocessor = AudioPreprocessor(config_path=os.path.join(PROJECT_ROOT, "config", "default_config.yaml"))
    analyzer = AudioQualityAnalyzer()

    # 1. Clean sample test
    clean_file = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")
    print(f"\n[TEST 1] Analyzing clean sample: '{clean_file}'")
    wf_clean, sr = preprocessor.process_file(clean_file)
    res_clean = analyzer.analyze(wf_clean, sr)

    print(f" -> Quality Score: {res_clean['quality_score']:.1f} ({res_clean['quality_rating']})")
    print(f" -> SNR: {res_clean['snr_db']:.1f} dB (Noise Level: {res_clean['noise_level']})")
    print(f" -> Clipping: {res_clean['clipping_pct']:.2f}%")
    print(f" -> Silence Ratio: {res_clean['silence_pct']:.1f}%")
    print(f" -> RMS Energy: {res_clean['rms_energy']:.4f}")
    print(f" -> ZCR: {res_clean['zcr']:.4f}")
    print(f" -> Dynamic Range: {res_clean['dynamic_range_db']:.1f} dB")
    print(f" -> Prediction Reliability: {res_clean['prediction_reliability']}")

    assert res_clean["quality_score"] > 60.0, "Clean sample scored too low!"

    # 2. Noisy sample test
    print("\n[TEST 2] Analyzing synthetic noisy sample...")
    noisy_wf = wf_clean + torch.randn_like(wf_clean) * 0.15
    res_noisy = analyzer.analyze(noisy_wf, sr)
    print(f" -> Noisy Sample SNR: {res_noisy['snr_db']:.1f} dB")
    print(f" -> Noisy Quality Rating: {res_noisy['quality_rating']}")
    print(f" -> Prediction Reliability: {res_noisy['prediction_reliability']}")

    assert res_noisy["snr_db"] < res_clean["snr_db"], "Noisy sample SNR should be lower than clean!"

    print("\n" + "=" * 50)
    print(" AUDIO QUALITY RISK ANALYZER TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
