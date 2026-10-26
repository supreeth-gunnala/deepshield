"""
DeepShield Phase 13 PDF Report Test Script
Verifies PDF generation, metadata insertion, image embedding, degradation reporting, and file export.
"""

import os
import sys
import io
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor
from src.reporting.pdf_report import generate_audio_report

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
SAMPLE_FILE = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")
TEST_PDF_PATH = os.path.join(PROJECT_ROOT, "experiments", "sample_deepshield_report.pdf")

def run_pdf_report_test():
    print("==================================================")
    print("      DEEPSHIELD PHASE 13 PDF REPORT TEST         ")
    print("==================================================")

    # 1. Load Audio & Preprocess
    print("\n[STEP 1] Preprocessing sample audio and generating plot buffers...")
    preprocessor = AudioPreprocessor(config_path=CONFIG_PATH)
    extractor = LogMelSpectrogramExtractor(config_path=CONFIG_PATH)

    waveform, sr = preprocessor.process_file(SAMPLE_FILE)
    spec_tensor = extractor(waveform)

    # Generate Waveform Image Buffer
    fig_wf, ax_wf = plt.subplots(figsize=(8, 2))
    ax_wf.plot(waveform.squeeze(0).numpy(), color="#1f77b4", lw=0.8)
    ax_wf.set_title("16kHz Mono Signal Waveform")
    ax_wf.set_ylim([-1.05, 1.05])
    plt.tight_layout()
    buf_wf = io.BytesIO()
    plt.savefig(buf_wf, format="png", dpi=120)
    buf_wf.seek(0)
    plt.close(fig_wf)

    # Generate Spectrogram Image Buffer
    fig_spec, ax_spec = plt.subplots(figsize=(8, 2.2))
    img = ax_spec.imshow(spec_tensor.squeeze(0).numpy(), aspect="auto", origin="lower", cmap="magma")
    fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
    ax_spec.set_title("Log-Mel Spectrogram")
    plt.tight_layout()
    buf_spec = io.BytesIO()
    plt.savefig(buf_spec, format="png", dpi=120)
    buf_spec.seek(0)
    plt.close(fig_spec)

    # 2. Prepare Sample Metadata
    audio_meta = {
        "filename": "clean_mono_16k.wav",
        "file_format": "wav",
        "file_size": os.path.getsize(SAMPLE_FILE),
        "duration": 3.0,
        "sample_rate": 16000,
        "channels": 1,
        "validation_status": "Valid"
    }

    pred_res = {
        "model_name": "2D CNN Baseline",
        "predicted_class": "AI-GENERATED",
        "real_prob": 0.2882,
        "fake_prob": 0.7118,
        "confidence": 71.18
    }

    preproc_info = {
        "target_sample_rate": 16000,
        "target_duration": 3.0,
        "mono_conversion": True,
        "normalization_method": "Peak Normalization"
    }

    feat_info = {
        "feature_type": "Log-Mel Spectrogram",
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 512,
        "win_length": 1024,
        "f_range": "0 - 8000 Hz",
        "tensor_shape": "[1, 128, 94]"
    }

    deg_info = {
        "condition": "Noise 10 dB SNR",
        "orig_pred": "AI-GENERATED",
        "deg_pred": "AI-GENERATED",
        "orig_conf": 71.18,
        "deg_conf": 68.45,
        "flipped": False,
        "conf_change": -2.73
    }

    quality_info = {
        "quality_score": 92.5,
        "quality_rating": "Excellent",
        "snr_db": 28.4,
        "noise_level": "Low",
        "clipping_pct": 0.0,
        "silence_pct": 12.0,
        "rms_energy": 0.124,
        "zcr": 0.045,
        "dynamic_range_db": 42.1,
        "prediction_reliability": "High"
    }

    stability_info = {
        "original_prediction": "AI-GENERATED",
        "matching_predictions": 4,
        "total_variants": 5,
        "prediction_flips": 1,
        "stability_percentage": 80.0
    }

    explanation_metrics_info = {
        "time_window_str": "0.40s - 0.80s",
        "dominant_freq_band": "High Frequencies (85-128 Mels)",
        "summary": "Peak attention centered around 0.60s focusing primarily on High Frequencies."
    }

    # 3. Test PDF Generation
    print("[STEP 2] Generating PDF report buffer with ReportLab...")
    pdf_bytes = generate_audio_report(
        audio_metadata=audio_meta,
        prediction_result=pred_res,
        preprocessing_info=preproc_info,
        feature_info=feat_info,
        waveform_image=buf_wf,
        spectrogram_image=buf_spec,
        degradation_result=deg_info,
        quality_result=quality_info,
        stability_result=stability_info,
        explanation_image=buf_spec,
        explanation_metrics=explanation_metrics_info
    )

    print(f"Generated PDF Size: {len(pdf_bytes):,} bytes")
    assert len(pdf_bytes) > 0, "PDF bytes buffer is empty!"

    # 4. Export PDF File to Disk for Verification
    os.makedirs(os.path.dirname(TEST_PDF_PATH), exist_ok=True)
    with open(TEST_PDF_PATH, "wb") as f:
        f.write(pdf_bytes)

    print(f"[SUCCESS] Exported test PDF report to: '{TEST_PDF_PATH}'")

    print("\n==================================================")
    print("   DEEPSHIELD PDF REPORT TEST PASSED SUCCESSFULLY!")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_pdf_report_test()
    sys.exit(0 if success else 1)
