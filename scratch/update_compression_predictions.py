import os
import json
import pandas as pd
import torch
import yaml
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Project root (cwd is the project root when script is launched)
PROJECT_ROOT = Path.cwd()

# Import project modules
sys_path = str(PROJECT_ROOT)
if sys_path not in __import__('sys').path:
    __import__('sys').path.insert(0, sys_path)

from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.evaluation.metrics import compute_classification_metrics
from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor
from evaluate_robustness import compress_audio

logger = logging.getLogger('DeepShield.CompressionUpdate')

def load_test_metadata(cfg: dict) -> pd.DataFrame:
    metadata_path = os.path.join(
        PROJECT_ROOT, cfg.get('dataset', {}).get('metadata_csv', 'data/metadata/dataset_metadata.csv')
    )
    df = pd.read_csv(metadata_path)
    return df[df['split'] == 'test'].reset_index(drop=True)

def main():
    # Load config
    CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'default_config.yaml')
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    device = get_device()

    # Build model and load checkpoint
    model = build_model(model_name='crnn_advanced', config_path=CONFIG_PATH, device=device)
    ckpt_dir = os.path.join(PROJECT_ROOT, 'models_saved', 'checkpoints', 'asvspoof_voice')
    ckpt_manager = CheckpointManager(checkpoint_dir=ckpt_dir)
    ckpt_manager.load_checkpoint(filename='best_model.pt', model=model, device=device)
    model.eval()
    torch.set_grad_enabled(False)

    # Preprocessor & feature extractor
    preprocessor = AudioPreprocessor(config_path=CONFIG_PATH)
    feature_extractor = LogMelSpectrogramExtractor(config_path=CONFIG_PATH)

    test_df = load_test_metadata(cfg)
    # Ensure deterministic ordering by sorting on file_path
    test_df = test_df.sort_values('file_path').reset_index(drop=True)

    # Run compression condition only
    all_labels = []
    all_preds = []
    all_probs = []
    records = []
    for _, row in tqdm(test_df.iterrows(), desc='Evaluating [compression_64kbps_aligned]', leave=False):
        file_path = row['file_path']
        true_label = int(row['label_id'])
        waveform, sr = preprocessor.process_file(file_path)
        # Apply compression (lossy MP3 at 64 kbps)
        waveform = compress_audio(file_path, bitrate_kbps=64)
        spec = feature_extractor(waveform)
        if spec.dim() == 3:
            spec = spec.unsqueeze(0)
        spec = spec.to(device)
        logits = model(spec)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
        prob_arr = probs.squeeze(0).cpu().numpy()

        all_labels.append(true_label)
        all_preds.append(pred)
        all_probs.append(prob_arr)

        records.append({
            'condition': 'compression_64kbps_aligned',
            'file_path': file_path,
            'true_label_id': true_label,
            'pred_label_id': pred,
            'prob_real': float(prob_arr[0]),
            'prob_ai_generated': float(prob_arr[1]),
            'correct': true_label == pred,
            'generator': row.get('generator', None),
            'label_id': true_label,
        })

    # Save predictions CSV (separate file)
    pred_df = pd.DataFrame(records)
    out_csv = PROJECT_ROOT / 'experiments' / 'results' / 'asvspoof_voice_robustness_predictions_compression_aligned.csv'
    pred_df.to_csv(out_csv, index=False)
    logger.info(f'Compression‑aligned predictions saved to {out_csv}')

    # Also compute and save metrics for this condition (optional)
    y_true = torch.tensor(all_labels).numpy()
    y_pred = torch.tensor(all_preds).numpy()
    y_prob = np.vstack(all_probs)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    out_json = PROJECT_ROOT / 'experiments' / 'results' / 'compression_aligned_metrics.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'Compression‑aligned metrics saved to {out_json}')

if __name__ == '__main__':
    main()
