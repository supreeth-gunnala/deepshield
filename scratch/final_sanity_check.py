import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Project root (one level up from this script's directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------- Paths ----------
METRICS_PATH = PROJECT_ROOT / "experiments" / "results" / "asvspoof_voice_robustness_metrics.json"
PRED_PATH = PROJECT_ROOT / "experiments" / "results" / "asvspoof_voice_robustness_predictions.csv"
AUDIT_PATH = PROJECT_ROOT / "experiments" / "results" / "robustness_audit.json"
OUTPUT_PATH = PROJECT_ROOT / "experiments" / "results" / "final_sanity_check.json"

# ---------- Load data ----------
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics_json = json.load(f)

pred_df = pd.read_csv(PRED_PATH)
with open(AUDIT_PATH, "r", encoding="utf-8") as f:
    audit_json = json.load(f)

# ---------- Helper functions ----------
def compute_metrics_from_df(df):
    y_true = df["true_label_id"].to_numpy()
    y_pred = df["pred_label_id"].to_numpy()
    # Probabilities for class 1 (AI‑GENERATED) – column "prob_ai_generated"
    y_prob = df[["prob_real", "prob_ai_generated"]].to_numpy()
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob[:, 1])
    except ValueError:
        roc = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    # False‑positive rate: FP / (FP + TN)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4) if roc is not None else None,
        "confusion_matrix": cm,
        "false_positive_rate": round(fpr, 4) if fpr is not None else None,
    }

# ---------- Re‑compute metrics for each condition ----------
conditions = sorted(pred_df["condition"].unique())
recomputed = {}
for cond in conditions:
    df_cond = pred_df[pred_df["condition"] == cond]
    recomputed[cond] = compute_metrics_from_df(df_cond)

# ---------- Compare with stored JSON ----------
metric_mismatches = {}
for cond, stored in metrics_json.items():
    recomputed_vals = recomputed.get(cond)
    if not recomputed_vals:
        metric_mismatches[cond] = "Condition missing in predictions CSV"
        continue
    diffs = {}
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"]:
        if stored.get(key) != recomputed_vals.get(key):
            diffs[key] = {"stored": stored.get(key), "recomputed": recomputed_vals.get(key)}
    if diffs:
        metric_mismatches[cond] = diffs

# ---------- Verify combined condition consistency ----------
combined_stored = metrics_json.get("combined_noise10db_resample8k", {})
combined_recomputed = recomputed.get("combined_noise10db_resample8k", {})
combined_match = combined_stored == combined_recomputed

# ---------- Investigate telephone_filter & compression ----------
# Import the degradation functions from evaluate_robustness (they are defined there)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "evaluate_robustness",
    PROJECT_ROOT / "scratch" / "evaluate_robustness.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

preproc = mod.AudioPreprocessor(config_path=str(PROJECT_ROOT / "config" / "default_config.yaml"))

# Load a few test files (first 5 unique paths from clean condition)
clean_files = pred_df[pred_df["condition"] == "clean"]["file_path"].unique()[:5]
telephone_changed = []
compression_changed = []
for fp in clean_files:
    # Load original waveform
    wav_orig, sr = preproc.process_file(fp)
    # Apply telephone filter
    wav_tel = mod.telephone_bandpass(wav_orig.clone(), sr)
    telephone_changed.append(torch.allclose(wav_orig, wav_tel, atol=1e-6))
    # Apply compression (the function expects a file path)
    wav_cmp = mod.compress_audio(fp, bitrate_kbps=64)
    compression_changed.append(torch.allclose(wav_orig, wav_cmp, atol=1e-6))

telephone_all_equal = all(telephone_changed)
compression_all_equal = all(compression_changed)

# ---------- Prediction change verification for the two conditions ----------
changes_tel = (pred_df[pred_df["condition"] == "clean"]["pred_label_id"].values !=
               pred_df[pred_df["condition"] == "telephone_filter"]["pred_label_id"].values).sum()
changes_cmp = (pred_df[pred_df["condition"] == "clean"]["pred_label_id"].values !=
               pred_df[pred_df["condition"] == "compression_64kbps"]["pred_label_id"].values).sum()

# ---------- Assemble final report ----------
report = {
    "recomputed_metrics": recomputed,
    "metric_mismatches": metric_mismatches,
    "combined_condition_match": combined_match,
    "telephone_filter_waveform_unchanged": telephone_all_equal,
    "compression_waveform_unchanged": compression_all_equal,
    "prediction_changes": {
        "telephone_filter": int(changes_tel),
        "compression_64kbps": int(changes_cmp),
    },
    "false_positive_rates": {cond: recomputed[cond]["false_positive_rate"] for cond in conditions},
    "audit_inconsistencies": audit_json.get("inconsistencies", []),
}

# Write the report
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"Final sanity-check report written to {OUTPUT_PATH}")
