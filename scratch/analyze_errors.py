import os
import sys
import yaml
import json
import torch
import numpy as np
import hashlib
import csv
from tqdm import tqdm
import pandas as pd

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Project imports
from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.dataset.dataloader_factory import create_dataloaders
from src.evaluation.metrics import compute_classification_metrics

def file_md5(path, block_size=65536):
    """Compute MD5 hash of a file (binary)."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def detect_duplicates(metadata_csv, splits=('train', 'val', 'test')):
    """Detect files that appear in more than one split.
    Returns a dict mapping hash -> list of (split, file_path).
    """
    df = pd.read_csv(metadata_csv)
    # Restrict to requested splits
    df = df[df['split'].isin(splits)]
    hash_map = {}
    for _, row in df.iterrows():
        fp = row['file_path']
        split = row['split']
        if not os.path.isfile(fp):
            continue  # skip missing files – they would have been caught earlier
        h = file_md5(fp)
        hash_map.setdefault(h, []).append((split, fp))
    duplicates = {h: items for h, items in hash_map.items() if len(items) > 1}
    return duplicates

def main():
    # ------------------------------------------------------------
    # 1. Load configuration and device
    # ------------------------------------------------------------
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
    device = get_device()

    # ------------------------------------------------------------
    # 2. Build model and load best checkpoint (no training)
    # ------------------------------------------------------------
    model = build_model(model_name="crnn_advanced", config_path=CONFIG_PATH, device=device)
    ckpt_dir = os.path.join(PROJECT_ROOT, "models_saved", "checkpoints", "asvspoof_voice")
    ckpt_manager = CheckpointManager(checkpoint_dir=ckpt_dir)
    ckpt_manager.load_checkpoint(filename="best_model.pt", model=model, device=device)

    model.eval()
    torch.set_grad_enabled(False)

    # ------------------------------------------------------------
    # 3. Get loaders (only test loader will be used for predictions)
    # ------------------------------------------------------------
    _, _, test_loader = create_dataloaders(config_path=CONFIG_PATH)

    # ------------------------------------------------------------
    # 4. Evaluate each test sample and record detailed info
    # ------------------------------------------------------------
    records = []  # list of dicts for CSV
    all_labels = []
    all_preds = []
    all_probs = []

    for batch in tqdm(test_loader, desc="Analyzing test errors"):
        features, labels, metas = batch
        features = features.to(device)
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        # Convert to CPU numpy for easier handling
        label_arr = labels.cpu().numpy()
        pred_arr = preds.cpu().numpy()
        prob_arr = probs.cpu().numpy()

        all_labels.append(label_arr)
        all_preds.append(pred_arr)
        all_probs.append(prob_arr)

        # Each batch may contain multiple samples; iterate
        for i in range(len(label_arr)):
            meta = metas[i] if isinstance(metas, list) else metas[i]
            file_path = meta.get('file_path', '')
            true_label = int(label_arr[i])
            pred_label = int(pred_arr[i])
            prob_real = float(prob_arr[i][0])
            prob_ai = float(prob_arr[i][1])
            correct = (true_label == pred_label)
            records.append({
                "file_path": file_path,
                "true_label_id": true_label,
                "pred_label_id": pred_label,
                "prob_real": prob_real,
                "prob_ai_generated": prob_ai,
                "correct": correct
            })

    # ------------------------------------------------------------
    # 5. Compute overall metrics (optional sanity check)
    # ------------------------------------------------------------
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    print("Overall test metrics (sanity check):")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ------------------------------------------------------------
    # 6. Detect duplicate audio across splits
    # ------------------------------------------------------------
    metadata_csv = os.path.join(PROJECT_ROOT, cfg.get('dataset', {}).get('metadata_csv', 'data/metadata/dataset_metadata.csv'))
    duplicates = detect_duplicates(metadata_csv)
    if duplicates:
        print("\nDuplicate audio files found across splits:")
        for h, items in duplicates.items():
            print(f"Hash {h} appears in {len(items)} splits:")
            for split, fp in items:
                print(f"  - {split}: {fp}")
    else:
        print("\nNo duplicate audio files across splits detected.")

    # ------------------------------------------------------------
    # 7. Save detailed per‑sample analysis to CSV
    # ------------------------------------------------------------
    results_dir = os.path.join(PROJECT_ROOT, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "asvspoof_voice_error_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file_path", "true_label_id", "pred_label_id", "prob_real", "prob_ai_generated", "correct"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    print(f"\nDetailed error analysis saved to: {csv_path}")

if __name__ == "__main__":
    main()
