"""
DeepShield Phase 12 Final System Integration Test Script
Executes end-to-end integration test across all 12 project phases:
Configuration -> Preprocessing -> Spectrogram Extractor -> DataLoaders -> CNN Model -> CRNN Model -> Degradations -> Evaluation -> Streamlit App Import.
"""

import os
import sys
import yaml
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor
from src.dataset.dataloader_factory import create_dataloaders
from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.audio.degradations import apply_combined_degradation
from src.evaluation.evaluator import DeepShieldEvaluator
from src.evaluation.robustness_evaluator import RobustnessEvaluator

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")

def run_final_system_test():
    print("==================================================")
    print("   DEEPSHIELD PHASE 12 FINAL SYSTEM TEST          ")
    print("==================================================")

    device = get_device()

    # 1. Config Test
    print("\n[CHECK 1/10] Loading default configuration...")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg is not None, "Failed to load default_config.yaml"
    print(" -> Config loaded successfully.")

    # 2. Audio Preprocessor & Validator Test
    print("[CHECK 2/10] Testing Audio Preprocessor & Validator...")
    preprocessor = AudioPreprocessor(config_path=CONFIG_PATH)
    sample_file = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")
    wf, sr = preprocessor.process_file(sample_file)
    assert wf.shape == (1, 48000), f"Invalid waveform shape: {wf.shape}"
    print(f" -> Preprocessed waveform shape: {wf.shape} @ {sr} Hz")

    # 3. Log-Mel Spectrogram Extractor Test
    print("[CHECK 3/10] Testing Log-Mel Spectrogram Extractor...")
    extractor = LogMelSpectrogramExtractor(config_path=CONFIG_PATH)
    spec_tensor = extractor(wf)
    assert spec_tensor.shape == (1, 128, 94), f"Invalid spectrogram shape: {spec_tensor.shape}"
    print(f" -> Spectrogram shape: {spec_tensor.shape}")

    # 4. Dataset Loader Test
    print("[CHECK 4/10] Testing Dataset Loaders & Generator-Disjoint Splits...")
    train_loader, val_loader, test_loader = create_dataloaders(config_path=CONFIG_PATH)
    assert len(train_loader) > 0 and len(test_loader) > 0, "DataLoaders are empty!"
    print(f" -> DataLoaders active (Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)})")

    # 5. CNN Baseline Model Test
    print("[CHECK 5/10] Testing CNN Baseline Model & Checkpoint...")
    cnn_model = build_model(model_name="cnn_baseline", config_path=CONFIG_PATH, device=device)
    ckpt_mgr = CheckpointManager(checkpoint_dir="models_saved/checkpoints")
    ckpt_mgr.load_checkpoint("best_model.pt", model=cnn_model, device=device)
    print(" -> CNN Baseline model loaded successfully.")

    # 6. Advanced CRNN Model Test
    print("[CHECK 6/10] Testing Advanced CRNN Model & Checkpoint...")
    crnn_model = build_model(model_name="crnn_advanced", config_path=CONFIG_PATH, device=device)
    crnn_ckpt_mgr = CheckpointManager(checkpoint_dir="models_saved/checkpoints/crnn_advanced")
    crnn_ckpt_mgr.load_checkpoint("best_model.pt", model=crnn_model, device=device)
    print(" -> Advanced CRNN model loaded successfully.")

    # 7. Real-World Audio Degradation Engine Test
    print("[CHECK 7/10] Testing Real-World Audio Degradation Transformations...")
    deg_wf, deg_sr = apply_combined_degradation(wf, sr)
    assert not torch.isnan(deg_wf).any(), "NaN detected in degraded audio!"
    print(f" -> Combined degradation applied: {deg_wf.shape}")

    # 8. Single File & Batch Inference Engine Test
    print("[CHECK 8/10] Testing Inference & Softmax Normalization...")
    evaluator = DeepShieldEvaluator(model=cnn_model, device=device)
    df_preds, _, _, _ = evaluator.predict_loader(test_loader)
    assert len(df_preds) > 0, "Inference prediction table is empty!"
    print(f" -> Generated predictions for {len(df_preds)} test samples.")

    # 9. Robustness Benchmarking Engine Test
    print("[CHECK 9/10] Testing Robustness Evaluator Suite...")
    rob_eval = RobustnessEvaluator(model=cnn_model, device=device, config_path=CONFIG_PATH)
    df_rob = rob_eval.run_full_robustness_suite(test_loader, output_dir=os.path.join(PROJECT_ROOT, "experiments", "test_final_results"))
    assert len(df_rob) > 0, "Robustness summary is empty!"
    print(f" -> Computed robustness metrics across {len(df_rob)} conditions.")

    # 10. Streamlit App Import Test
    print("[CHECK 10/10] Testing Streamlit Application Module Import...")
    import app
    assert app is not None, "Streamlit app.py import failed!"
    print(" -> Streamlit app.py imported cleanly without errors.")

    print("\n==================================================")
    print("   ALL 10 FINAL SYSTEM CHECKS PASSED SUCCESSFULLY! ")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_final_system_test()
    sys.exit(0 if success else 1)
