"""
DeepShield Test Script: Explainable AI (Grad-CAM) Heatmap Generation
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
from src.explainability.grad_cam import GradCAMExplainer
from src.explainability.visualizer import plot_explanation_overlay, fig_to_bytes

def main():
    print("=" * 50)
    print("      DEEPSHIELD XAI GRAD-CAM MODULE TEST        ")
    print("=" * 50)

    # 1. Initialize components
    config_path = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    preprocessor = AudioPreprocessor(config_path=config_path)
    extractor = LogMelSpectrogramExtractor(config_path=config_path)
    model = DeepShieldCNN(config_path=config_path)
    model.eval()

    # 2. Process sample audio
    sample_path = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")
    print(f"\n[STEP 1] Preprocessing sample audio: '{sample_path}'")
    waveform, sr = preprocessor.process_file(sample_path)
    spec_tensor = extractor(waveform)
    print(f" -> Spectrogram shape: {spec_tensor.shape}")

    # 3. Initialize GradCAMExplainer
    print("\n[STEP 2] Initializing GradCAMExplainer and hooking target Conv2D block...")
    explainer = GradCAMExplainer(model)
    print(f" -> Target Conv2D Layer: {explainer.target_layer}")

    # 4. Generate Heatmap
    batch_spec = spec_tensor.unsqueeze(0)
    heatmap, info = explainer.generate_heatmap(batch_spec, target_class=1)
    
    print("\n[STEP 3] Evaluating Grad-CAM Heatmap Output...")
    print(f" -> Status: {info['status']}")
    print(f" -> Target Class: {info['target_class_name']}")
    print(f" -> Confidence: {info['confidence']:.2f}%")
    print(f" -> Heatmap shape: {heatmap.shape if heatmap is not None else 'None'}")

    assert heatmap is not None, "Heatmap generation returned None!"
    assert heatmap.shape == (spec_tensor.shape[1], spec_tensor.shape[2]), f"Mismatch in heatmap shape: {heatmap.shape}"
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0, "Heatmap values out of [0, 1] range!"

    # 5. Render Plot Overlay
    print("\n[STEP 4] Rendering Grad-CAM plot overlay and converting to buffer...")
    fig, metrics = plot_explanation_overlay(spec_tensor, heatmap, title="Grad-CAM Test Overlay")
    img_buf = fig_to_bytes(fig)
    print(f" -> Peak Time Window: {metrics['time_window_str']}")
    print(f" -> Dominant Frequency Band: {metrics['dominant_freq_band']}")
    print(f" -> Rendered PNG Buffer Size: {len(img_buf.getvalue()):,} bytes")

    explainer.remove_hooks()

    print("\n" + "=" * 50)
    print("   DEEPSHIELD XAI GRAD-CAM TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
