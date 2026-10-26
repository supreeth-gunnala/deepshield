"""
DeepShield Phase 3 Test Script
Tests Log-Mel Spectrogram extraction pipeline, verifies feature dimensions, and saves visualization artifact.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
SAMPLE_AUDIO = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments")

def run_spectrogram_test():
    print("==================================================")
    print("  DEEPSHIELD PHASE 3 SPECTROGRAM EXTRACTION TEST   ")
    print("==================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize Preprocessor and Spectrogram Extractor
    preprocessor = AudioPreprocessor(config_path=CONFIG_PATH)
    extractor = LogMelSpectrogramExtractor(config_path=CONFIG_PATH)

    # 2. Preprocess test audio file
    print(f"[STEP 1] Loading and preprocessing sample audio: '{SAMPLE_AUDIO}'")
    waveform, sr = preprocessor.process_file(SAMPLE_AUDIO)

    # 3. Extract Log-Mel Spectrogram
    print(f"[STEP 2] Extracting Log-Mel Spectrogram feature tensor...")
    spec_tensor = extractor(waveform)

    # 4. Print stats and dimensions
    min_val = spec_tensor.min().item()
    max_val = spec_tensor.max().item()
    mean_val = spec_tensor.mean().item()
    std_val = spec_tensor.std().item()

    print("\n--------------------------------------------------")
    print("               FEATURE EXTRACTOR STATS             ")
    print("--------------------------------------------------")
    print(f"Input Waveform Shape : {waveform.shape} (1 channel, {waveform.shape[1]} samples)")
    print(f"Sample Rate          : {sr} Hz")
    print(f"Spectrogram Shape    : {spec_tensor.shape} (Channels, Mels, Time Frames)")
    print(f"Feature Min Value    : {min_val:.4f}")
    print(f"Feature Max Value    : {max_val:.4f}")
    print(f"Feature Mean Value   : {mean_val:.4f}")
    print(f"Feature Std Value    : {std_val:.4f}")
    print("--------------------------------------------------\n")

    # 5. Save Spectrogram Image Visualization
    output_image_path = os.path.join(OUTPUT_DIR, "spectrogram_test.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(
        spec_tensor.squeeze(0).cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("DeepShield Log-Mel Spectrogram Feature (128 Mels x Time Frames)")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    plt.close()

    print(f"[SUCCESS] Saved spectrogram visualization image to: '{output_image_path}'")

    # 6. Verify CNN Compatibility Shape Assertion
    assert spec_tensor.dim() == 3, f"Expected 3D tensor (C, F, T), got {spec_tensor.dim()}D"
    assert spec_tensor.shape[0] == 1, f"Expected 1 channel, got {spec_tensor.shape[0]}"
    assert spec_tensor.shape[1] == 128, f"Expected 128 Mel bins, got {spec_tensor.shape[1]}"

    print("\n==================================================")
    print("   SPECTROGRAM FEATURE EXTRACTION TEST PASSED!    ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_spectrogram_test()
    sys.exit(0 if success else 1)
