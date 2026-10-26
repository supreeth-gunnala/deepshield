import os
import json
import subprocess
import tempfile
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Project root (cwd is the project root when script is launched)
PROJECT_ROOT = Path.cwd()

# Paths
CLEAN_CSV = PROJECT_ROOT / "experiments" / "results" / "asvspoof_voice_robustness_predictions.csv"
COMP_CSV = PROJECT_ROOT / "experiments" / "results" / "asvspoof_voice_robustness_predictions_compression_fixed.csv"
METRICS_JSON = PROJECT_ROOT / "experiments" / "results" / "compression_fixed_metrics.json"
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "results" / "compression_final_verification.json"

logger = logging.getLogger("DeepShield.CompressionVerification")

def load_predictions(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    required = {"file_path", "true_label_id", "pred_label_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")
    return df

def compute_metrics(df):
    y_true = df["true_label_id"].to_numpy()
    y_pred = df["pred_label_id"].to_numpy()
    # probabilities needed for ROC-AUC – use existing columns if present
    if {"prob_real", "prob_ai_generated"}.issubset(df.columns):
        y_prob = df[["prob_real", "prob_ai_generated"]].to_numpy()
        roc = None
        try:
            roc = roc_auc_score(y_true, y_prob[:, 1])
        except Exception:
            roc = None
    else:
        roc = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc, 4) if roc is not None else None,
        "confusion_matrix": cm,
        "false_positive_rate": round(fpr, 4) if fpr is not None else None,
    }

def compare_metrics(stored, recomputed, tol=1e-4):
    diffs = {}
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        s = stored.get(key)
        r = recomputed.get(key)
        if s is None or r is None:
            continue
        if abs(s - r) > tol:
            diffs[key] = {"stored": s, "recomputed": r}
    if stored.get("confusion_matrix") != recomputed.get("confusion_matrix"):
        diffs["confusion_matrix"] = {"stored": stored.get("confusion_matrix"), "recomputed": recomputed.get("confusion_matrix")}
    return diffs

def ffmpeg_check(sample_path, bitrate_kbps=64):
    # Encode to temporary MP3
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    cmd = [
        "ffmpeg", "-y", "-i", str(sample_path),
        "-b:a", f"{bitrate_kbps}k",
        "-f", "mp3",
        tmp_out_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        os.remove(tmp_out_path)
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='ignore')}")
    # Probe the file
    probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=bit_rate", "-of", "default=noprint_wrappers=1:nokey=1", tmp_out_path]
    probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bitrate = None
    if probe_res.returncode == 0:
        try:
            bitrate = int(probe_res.stdout.decode().strip()) // 1000  # convert to kbps
        except Exception:
            bitrate = None
    # Get codec name
    codec_cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", tmp_out_path]
    codec_res = subprocess.run(codec_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    codec = codec_res.stdout.decode().strip() if codec_res.returncode == 0 else None
    os.remove(tmp_out_path)
    return {"codec": codec, "bitrate_kbps": bitrate}

def main():
    clean_df = load_predictions(CLEAN_CSV)
    comp_df = load_predictions(COMP_CSV)

    # 1. Compare predictions by file_path
    merged = clean_df[['file_path', 'pred_label_id']].rename(columns={'pred_label_id': 'clean_pred'}).merge(
        comp_df[['file_path', 'pred_label_id']].rename(columns={'pred_label_id': 'comp_pred'}),
        on='file_path', how='inner')
    changed = (merged['clean_pred'] != merged['comp_pred']).sum()

    # 2. Verify that we have exactly 300 samples and same paths/labels
    same_paths = set(clean_df['file_path']) == set(comp_df['file_path']) and len(clean_df) == 300 and len(comp_df) == 300
    same_labels = (clean_df.sort_values('file_path')['true_label_id'].reset_index(drop=True).equals(
        comp_df.sort_values('file_path')['true_label_id'].reset_index(drop=True)))
    path_label_match = same_paths and same_labels

    # 3. Recompute metrics from compression CSV
    recomputed_metrics = compute_metrics(comp_df)
    with open(METRICS_JSON, 'r', encoding='utf-8') as f:
        stored_metrics = json.load(f)
    metric_diffs = compare_metrics(stored_metrics, recomputed_metrics)
    metrics_match = not metric_diffs

    # 4. Verify ffmpeg encoding on a sample file (first entry)
    sample_path = clean_df['file_path'].iloc[0]
    ffmpeg_info = {}
    ffmpeg_ok = False
    try:
        ffmpeg_info = ffmpeg_check(sample_path, bitrate_kbps=64)
        ffmpeg_ok = (ffmpeg_info.get('codec') == 'mp3' and abs(ffmpeg_info.get('bitrate_kbps', 0) - 64) <= 5)
    except Exception as e:
        ffmpeg_info = {"error": str(e)}
        ffmpeg_ok = False

    # 5. Assemble report
    report = {
        "changed_predictions": int(changed),
        "total_samples": 300,
        "path_label_match": path_label_match,
        "metrics_match": metrics_match,
        "metric_differences": metric_diffs,
        "ffmpeg_check": {
            "codec": ffmpeg_info.get('codec'),
            "bitrate_kbps": ffmpeg_info.get('bitrate_kbps'),
            "pass": ffmpeg_ok,
            "details": ffmpeg_info if not ffmpeg_ok else "Codec is mp3 and bitrate ≈ 64 kbps"
        }
    }
    # Write output JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
