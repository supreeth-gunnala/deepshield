"""
DeepShield Phase 2 Test Script
Tests audio loading, mono conversion, 16kHz resampling, normalization, padding/cropping, and error handling.
"""

import os
import sys
import torch
import soundfile as sf
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.preprocessor import AudioPreprocessor
from src.audio.validator import AudioValidationError

TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test_samples")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")

def create_synthetic_audio_files():
    """Generates synthetic audio files for testing pipeline features and edge cases."""
    os.makedirs(TEST_DIR, exist_ok=True)
    sr = 16000

    # 1. Clean 16kHz Mono Sine Wave (3.0s)
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    sine_mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(os.path.join(TEST_DIR, "clean_mono_16k.wav"), sine_mono, sr)

    # 2. Stereo 44.1kHz Tone (2.0s)
    t_44k = np.linspace(0, 2.0, int(44100 * 2.0), endpoint=False)
    sine_left = 0.4 * np.sin(2 * np.pi * 440 * t_44k)
    sine_right = 0.4 * np.sin(2 * np.pi * 880 * t_44k)
    stereo_44k = np.vstack([sine_left, sine_right]).T
    sf.write(os.path.join(TEST_DIR, "stereo_44k.wav"), stereo_44k, 44100)

    # 3. Short Audio (1.0s) -> Should be padded
    t_short = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    short_audio = 0.5 * np.sin(2 * np.pi * 440 * t_short)
    sf.write(os.path.join(TEST_DIR, "short_1s.wav"), short_audio, sr)

    # 4. Long Audio (5.0s) -> Should be cropped
    t_long = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    long_audio = 0.5 * np.sin(2 * np.pi * 440 * t_long)
    sf.write(os.path.join(TEST_DIR, "long_5s.wav"), long_audio, sr)

    # 5. Silent Audio (3.0s zeros) -> Should fail validation
    silent_audio = np.zeros(int(sr * 3.0))
    sf.write(os.path.join(TEST_DIR, "silent.wav"), silent_audio, sr)

    # 6. Corrupt / Empty File -> Should fail validation
    with open(os.path.join(TEST_DIR, "corrupt.wav"), "w") as f:
        f.write("Not an audio file header")

    print(f"[TEST SETUP] Created synthetic test audio files in: {TEST_DIR}")

def run_tests():
    print("==================================================")
    print("      DEEPSHIELD PHASE 2 PREPROCESSING TEST       ")
    print("==================================================")

    create_synthetic_audio_files()

    preprocessor = AudioPreprocessor(config_path=CONFIG_PATH)

    tests_passed = 0
    total_tests = 0

    # Test 1: Clean Mono 16kHz
    total_tests += 1
    file_mono = os.path.join(TEST_DIR, "clean_mono_16k.wav")
    try:
        wf, sr = preprocessor.process_file(file_mono)
        assert wf.shape == (1, 48000), f"Expected shape (1, 48000), got {wf.shape}"
        assert sr == 16000, f"Expected SR 16000, got {sr}"
        print(f"[PASS] Test 1: Mono 16kHz audio -> Output Shape: {wf.shape}, SR: {sr}")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Test 1 failed: {e}")

    # Test 2: Stereo 44.1kHz Resampling & Mono Conversion
    total_tests += 1
    file_stereo = os.path.join(TEST_DIR, "stereo_44k.wav")
    try:
        wf, sr = preprocessor.process_file(file_stereo)
        assert wf.shape == (1, 48000), f"Expected shape (1, 48000), got {wf.shape}"
        assert sr == 16000, f"Expected SR 16000, got {sr}"
        print(f"[PASS] Test 2: Stereo 44.1kHz -> Resampled & Converted to Mono: {wf.shape}, SR: {sr}")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Test 2 failed: {e}")

    # Test 3: Short Audio Padding
    total_tests += 1
    file_short = os.path.join(TEST_DIR, "short_1s.wav")
    try:
        wf, sr = preprocessor.process_file(file_short)
        assert wf.shape == (1, 48000), f"Expected padded shape (1, 48000), got {wf.shape}"
        print(f"[PASS] Test 3: Short 1.0s audio -> Padded to 3.0s: {wf.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Test 3 failed: {e}")

    # Test 4: Long Audio Cropping
    total_tests += 1
    file_long = os.path.join(TEST_DIR, "long_5s.wav")
    try:
        wf, sr = preprocessor.process_file(file_long)
        assert wf.shape == (1, 48000), f"Expected cropped shape (1, 48000), got {wf.shape}"
        print(f"[PASS] Test 4: Long 5.0s audio -> Cropped to 3.0s: {wf.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Test 4 failed: {e}")

    # Test 5: Silence Detection Validation
    total_tests += 1
    file_silent = os.path.join(TEST_DIR, "silent.wav")
    try:
        wf, sr = preprocessor.process_file(file_silent)
        print(f"[FAIL] Test 5 failed: Silent audio did not raise AudioValidationError.")
    except AudioValidationError as e:
        print(f"[PASS] Test 5: Silent audio correctly caught & rejected: '{e}'")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Test 5 failed with unexpected exception: {e}")

    # Test 6: Corrupt File Validation
    total_tests += 1
    file_corrupt = os.path.join(TEST_DIR, "corrupt.wav")
    try:
        wf, sr = preprocessor.process_file(file_corrupt)
        print(f"[FAIL] Test 6 failed: Corrupt file did not raise AudioValidationError.")
    except AudioValidationError as e:
        print(f"[PASS] Test 6: Corrupt file correctly caught & rejected: '{e}'")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Test 6 failed with unexpected exception: {e}")

    print("==================================================")
    print(f"   SUMMARY: {tests_passed} / {total_tests} TESTS PASSED")
    print("==================================================")
    return tests_passed == total_tests

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
