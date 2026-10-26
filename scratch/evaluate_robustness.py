import os
import sys
import yaml
import json
import torch
import numpy as np
import pandas as pd
import logging
import subprocess
import tempfile
import shutil
from tqdm import tqdm

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports from the DeepShield codebase
from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.evaluation.metrics import compute_classification_metrics, compute_generator_breakdown
from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor

logger = logging.getLogger("DeepShield.RobustnessEval")

# --------------------- Utility functions ---------------------

def add_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add white Gaussian noise to a waveform to achieve the desired SNR (dB)."""
    if snr_db == float('inf'):
        return waveform
    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
    return waveform + noise

def resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform from orig_sr to target_sr using torchaudio."""
    try:
        import torchaudio
    except ImportError as e:
        raise RuntimeError("torchaudio is required for resampling but is not installed.") from e
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)

def telephone_bandpass(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Apply a telephone‑band filter (300‑3400 Hz) using torchaudio functional filters."""
    try:
        import torchaudio.functional as F
    except ImportError as e:
        raise RuntimeError("torchaudio is required for telephone filter but is not installed.") from e
    # high‑pass at 300 Hz then low‑pass at 3400 Hz
    waveform = F.highpass_biquad(waveform, sr, cutoff_freq=300.0)
    waveform = F.lowpass_biquad(waveform, sr, cutoff_freq=3400.0)
    return waveform

def compress_audio(file_path: str, bitrate_kbps: int = 64) -> torch.Tensor:
    """Compress the audio using ffmpeg to a lossy codec (MP3) at the given bitrate and reload it.
    Returns a waveform tensor (1, N). Requires ffmpeg on the system.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg executable not found in PATH – cannot perform compression.")
    # Create a temporary MP3 file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i",
        file_path,
        "-b:a",
        f"{bitrate_kbps}k",
        "-f",
        "mp3",
        tmp_out_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        # Clean up the temporary file if ffmpeg failed
        os.remove(tmp_out_path)
        raise RuntimeError(f"ffmpeg compression failed: {result.stderr.decode(errors='ignore')}")
    # Load the compressed mp3 back using the same preprocessing pipeline
    preproc = AudioPreprocessor(config_path=None)  # config_path not needed for simple load
    waveform, sr = preproc.process_file(tmp_out_path)
    # Ensure we clean up the temp file
    os.remove(tmp_out_path)
    return waveform

def load_test_metadata(cfg: dict) -> pd.DataFrame:
    """Return a DataFrame of test split entries from the metadata CSV."""
    metadata_path = os.path.join(
        PROJECT_ROOT, cfg.get("dataset", {}).get("metadata_csv", "data/metadata/dataset_metadata.csv")
    )
    df = pd.read_csv(metadata_path)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return test_df

def evaluate_condition(
    model: torch.nn.Module,
    device: torch.device,
    preprocessor: AudioPreprocessor,
    feature_extractor: LogMelSpectrogramExtractor,
    test_df: pd.DataFrame,
    degradation_name: str,
    degradation_fn,
):
    """Run inference on the test set with a given degradation function.
    ``degradation_fn`` receives a waveform tensor and sample rate and returns a degraded waveform tensor.
    Returns a dict of metrics and a list of per‑sample records.
    """
    all_labels = []
    all_preds = []
    all_probs = []
    records = []

    for _, row in tqdm(test_df.iterrows(), desc=f"Evaluating [{degradation_name}]", leave=False):
        file_path = row["file_path"]
        true_label = int(row["label_id"])
        # Load original waveform (already normalized/padded)
        waveform, sr = preprocessor.process_file(file_path)
        # Apply degradation if any
        if degradation_fn is not None:
            waveform = degradation_fn(waveform, sr, file_path)
        # Extract spectrogram feature (same as training pipeline)
        spec = feature_extractor(waveform)
        # Ensure 4‑D tensor (B, C, M, T)
        if spec.dim() == 3:
            spec = spec.unsqueeze(0)
        elif spec.dim() != 4:
            raise RuntimeError(f"Unexpected spectrogram shape {spec.shape}, expected 3‑D or 4‑D tensor")
        spec = spec.to(device)
        logits = model(spec)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
        prob_arr = probs.squeeze(0).cpu().numpy()

        all_labels.append(true_label)
        all_preds.append(pred)
        all_probs.append(prob_arr)

        records.append(
            {
                "condition": degradation_name,
                "file_path": file_path,
                "true_label_id": true_label,
                "pred_label_id": pred,
                "prob_real": float(prob_arr[0]),
                "prob_ai_generated": float(prob_arr[1]),
                "correct": true_label == pred,
                "generator": row.get("generator", None),
                "label_id": true_label,
            }
        )

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.vstack(all_probs)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    # Also compute per‑generator breakdown for extra insight
    df_preds = pd.DataFrame(records)
    try:
        generator_breakdown = compute_generator_breakdown(df_preds)
    except Exception as e:
        logger.warning(f"Generator breakdown failed: {e}")
        generator_breakdown = None
    metrics["generator_breakdown"] = generator_breakdown
    return metrics, records

def main():
    # ------------------------------------------------------------
    # 1. Load configuration and device
    # ------------------------------------------------------------
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
    device = get_device()

    # ------------------------------------------------------------
    # 2. Build model and load the best checkpoint (no training)
    # ------------------------------------------------------------
    model = build_model(model_name="crnn_advanced", config_path=CONFIG_PATH, device=device)
    ckpt_dir = os.path.join(PROJECT_ROOT, "models_saved", "checkpoints", "asvspoof_voice")
    ckpt_manager = CheckpointManager(checkpoint_dir=ckpt_dir)
    ckpt_manager.load_checkpoint(filename="best_model.pt", model=model, device=device)
    model.eval()
    torch.set_grad_enabled(False)

    # ------------------------------------------------------------
    # 3. Prepare preprocessing and feature extraction utilities
    # ------------------------------------------------------------
    preprocessor = AudioPreprocessor(config_path=CONFIG_PATH)
    feature_extractor = LogMelSpectrogramExtractor(config_path=CONFIG_PATH)

    # ------------------------------------------------------------
    # 4. Load test metadata (clean reference set)
    # ------------------------------------------------------------
    test_df = load_test_metadata(cfg)
    if test_df.empty:
        logger.error("Test split is empty – cannot evaluate robustness.")
        return

    # ------------------------------------------------------------
    # 5. Define degradation conditions
    # ------------------------------------------------------------
    conditions = {
        "clean": None,  # no degradation – baseline reference
        "noise_20db": lambda w, sr, fp: add_noise(w, 20.0),
        "noise_10db": lambda w, sr, fp: add_noise(w, 10.0),
        "noise_0db": lambda w, sr, fp: add_noise(w, 0.0),
        "resample_8k": lambda w, sr, fp: resample_waveform(w, sr, 8000),
        "telephone_filter": lambda w, sr, fp: telephone_bandpass(w, sr),
        "compression_64kbps": lambda w, sr, fp: compress_audio(fp, 64),
        # Combined: noise 10 dB then down-sample to 8 kHz
        "combined_noise10db_resample8k": lambda w, sr, fp: resample_waveform(
            add_noise(w, 10.0), sr, 8000
        ),
    }

    # ------------------------------------------------------------
    # 6. Run evaluation for each condition
    # ------------------------------------------------------------
    all_records = []
    robustness_results = {}
    for name, fn in conditions.items():
        try:
            metrics, records = evaluate_condition(
                model, device, preprocessor, feature_extractor, test_df, name, fn
            )
            robustness_results[name] = metrics
            all_records.extend(records)
            logger.info(f"[{name}] metrics: {metrics}")
        except Exception as e:
            logger.warning(f"Condition '{name}' could not be evaluated: {e}")
            robustness_results[name] = {"error": str(e)}

    # ------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "asvspoof_voice_robustness_metrics.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(robustness_results, jf, indent=2)
    csv_path = os.path.join(results_dir, "asvspoof_voice_robustness_predictions.csv")
    df_all = pd.DataFrame(all_records)
    df_all.to_csv(csv_path, index=False)
    print(f"\nRobustness evaluation completed. Metrics saved to {json_path}")
    print(f"Per-sample predictions saved to {csv_path}")

if __name__ == "__main__":
    main()
