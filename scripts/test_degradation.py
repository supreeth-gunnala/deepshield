"""
DeepShield Phase 9 Audio Degradation Test Script
Tests all real-world audio degradations, verifies Phase 2/3 compatibility, and exports degraded sample WAV files.
"""

import os
import sys
import torch
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor
from src.audio.degradations import (
    add_noise, apply_resampling_degradation, apply_telephone_degradation,
    apply_compression, apply_pitch_shift, apply_combined_degradation, DegradationRegistry
)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
CLEAN_SAMPLE = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")
DEGRADED_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "degraded_examples")

def run_degradation_tests():
    print("==================================================")
    print("   DEEPSHIELD PHASE 9 AUDIO DEGRADATION TEST      ")
    print("==================================================")

    os.makedirs(DEGRADED_OUTPUT_DIR, exist_ok=True)

    # 1. Load clean preprocessed waveform
    preprocessor = AudioPreprocessor(config_path=CONFIG_PATH)
    extractor = LogMelSpectrogramExtractor(config_path=CONFIG_PATH)

    print(f"\n[STEP 1] Loading clean source audio: '{CLEAN_SAMPLE}'")
    clean_wf, sr = preprocessor.process_file(CLEAN_SAMPLE)

    clean_rms = torch.sqrt(torch.mean(clean_wf ** 2)).item()
    clean_min = clean_wf.min().item()
    clean_max = clean_wf.max().item()

    print("--------------------------------------------------")
    print("              ORIGINAL CLEAN AUDIO STATS          ")
    print("--------------------------------------------------")
    print(f"Shape: {clean_wf.shape}, Sample Rate: {sr} Hz")
    print(f"RMS Energy: {clean_rms:.4f}, Range: [{clean_min:.4f}, {clean_max:.4f}]")
    print("--------------------------------------------------\n")

    test_cases = [
        ("Noise 10dB", "sample_noise_10db.wav", lambda wf, s: add_noise(wf, s, snr_db=10.0)),
        ("Resample 8kHz", "sample_resampled_8khz.wav", lambda wf, s: apply_resampling_degradation(wf, s, target_sr=8000)),
        ("Telephone VoIP", "sample_telephone.wav", lambda wf, s: apply_telephone_degradation(wf, s)),
        ("Pitch +2", "sample_pitch_plus2.wav", lambda wf, s: apply_pitch_shift(wf, s, semitones=2.0)),
        ("Compression 8-bit", "sample_compression_8bit.wav", lambda wf, s: apply_compression(wf, s, bits=8)),
        ("Combined", "sample_combined.wav", lambda wf, s: apply_combined_degradation(wf, s, sequence=["noise", "telephone", "resample"]))
    ]

    tests_passed = 0

    for name, out_filename, deg_fn in test_cases:
        try:
            deg_wf, deg_sr = deg_fn(clean_wf, sr)

            # Assertions
            assert not torch.isnan(deg_wf).any(), f"NaN detected in {name} degradation!"
            assert not torch.isinf(deg_wf).any(), f"Inf detected in {name} degradation!"
            assert deg_wf.shape == clean_wf.shape, f"Shape mismatch in {name}! Got {deg_wf.shape}"

            # Phase 2 Preprocessing & Phase 3 Spectrogram Extractor Pass-through Verification
            processed_wf, p_sr = preprocessor.process_waveform(deg_wf, deg_sr)
            spec_tensor = extractor(processed_wf)

            assert spec_tensor.shape == (1, 128, 94), f"Spectrogram shape error in {name}: {spec_tensor.shape}"

            deg_rms = torch.sqrt(torch.mean(deg_wf ** 2)).item()
            deg_min = deg_wf.min().item()
            deg_max = deg_wf.max().item()

            # Save degraded audio file
            out_path = os.path.join(DEGRADED_OUTPUT_DIR, out_filename)
            sf.write(out_path, deg_wf.squeeze(0).cpu().numpy(), deg_sr)

            print(f"[PASS] {name:<18} | Degraded RMS: {deg_rms:.4f} | Range: [{deg_min:.2f}, {deg_max:.2f}] | Spectrogram: {spec_tensor.shape}")
            tests_passed += 1

        except Exception as e:
            print(f"[FAIL] {name} failed: {e}")

    print("\n--------------------------------------------------")
    print("           DEGRADATION REGISTRY VERIFICATION      ")
    print("--------------------------------------------------")
    reg_wf, _ = DegradationRegistry.apply("noise", clean_wf, sr, snr_db=15.0)
    assert reg_wf is not None and not torch.isnan(reg_wf).any()
    print(f"[SUCCESS] DegradationRegistry successfully invoked 'noise' degradation.")

    print("\n==================================================")
    print(f"   SUMMARY: {tests_passed} / {len(test_cases)} DEGRADATION TESTS PASSED")
    print(f"   Saved degraded WAV files to: '{DEGRADED_OUTPUT_DIR}'")
    print("==================================================")
    return tests_passed == len(test_cases)

if __name__ == "__main__":
    success = run_degradation_tests()
    sys.exit(0 if success else 1)
