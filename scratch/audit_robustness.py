import json
import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

METRICS_PATH = os.path.join(PROJECT_ROOT, 'experiments', 'results', 'asvspoof_voice_robustness_metrics.json')
PRED_PATH = os.path.join(PROJECT_ROOT, 'experiments', 'results', 'asvspoof_voice_robustness_predictions.csv')
DATASET_META_PATH = os.path.join(PROJECT_ROOT, 'data', 'metadata', 'dataset_metadata.csv')

# Load data
with open(METRICS_PATH, 'r', encoding='utf-8') as f:
    metrics = json.load(f)

pred_df = pd.read_csv(PRED_PATH)
# Ensure expected columns exist
expected_cols = {'condition', 'file_path', 'true_label_id', 'pred_label_id', 'prob_real', 'prob_ai_generated', 'correct', 'generator', 'label_id'}
missing = expected_cols - set(pred_df.columns)
if missing:
    raise ValueError(f"Missing columns in predictions CSV: {missing}")

# Load dataset metadata (for possible generator column verification)
metadata_df = pd.read_csv(DATASET_META_PATH)
# Verify that true_label_id mapping matches description in metadata if present
label_mapping = {0: 'REAL', 1: 'AI-GENERATED'}

# Helper to compute false positive rate
def false_positive_rate(df):
    # FP: true 0, pred 1
    fp = ((df['true_label_id'] == 0) & (df['pred_label_id'] == 1)).sum()
    tn = ((df['true_label_id'] == 0) & (df['pred_label_id'] == 0)).sum()
    if fp + tn == 0:
        return None
    return fp / (fp + tn)

# Verify each condition has exactly 300 samples and same file_paths/true_labels as clean condition
clean_df = pred_df[pred_df['condition'] == 'clean'].copy()
clean_files = list(clean_df['file_path'])
clean_labels = list(clean_df['true_label_id'])
conditions = sorted(pred_df['condition'].unique())
condition_counts = {}
per_condition_change_counts = {}
fp_rates = {}
inconsistencies = []

for cond in conditions:
    df = pred_df[pred_df['condition'] == cond]
    condition_counts[cond] = len(df)
    if len(df) != 300:
        inconsistencies.append(f"Condition '{cond}' has {len(df)} samples (expected 300).")
    # Verify file paths and true labels match clean
    if list(df['file_path']) != clean_files:
        inconsistencies.append(f"Condition '{cond}' file_path order differs from clean.")
    if list(df['true_label_id']) != clean_labels:
        inconsistencies.append(f"Condition '{cond}' true_label_id differs from clean.")
    # Compute prediction changes vs clean
    # Compute prediction changes vs clean (skip clean condition itself)
    if cond == "clean":
        change_count = 0
    else:
        # Merge clean predictions with current condition predictions on file_path
        merged = clean_df[['file_path', 'pred_label_id']].rename(columns={'pred_label_id': 'clean_pred'})
        merged = merged.merge(df[['file_path', 'pred_label_id']].rename(columns={'pred_label_id': f'{cond}_pred'}), on='file_path')
        change_count = (merged['clean_pred'] != merged[f'{cond}_pred']).sum()
    per_condition_change_counts[cond] = int(change_count)
    # False positive rate
    fp_rates[cond] = false_positive_rate(df)

# Verify label mapping consistency
if not set(pred_df['true_label_id'].unique()).issubset({0, 1}):
    inconsistencies.append("Unexpected true_label_id values found.")

# Verify ROC-AUC positive class handling (by definition in metrics, positive class is label 1)
# We'll note that compute_classification_metrics uses y_prob[:,1] as probability for class 1.
roc_auc_positive_class = 'AI-GENERATED (label 1)'

# Generator audit – count samples per generator if column exists
generator_audit = {}
if 'generator' in pred_df.columns:
    for cond in conditions:
        df = pred_df[pred_df['condition'] == cond]
        gen_counts = df['generator'].value_counts(dropna=False).to_dict()
        generator_audit[cond] = gen_counts
else:
    generator_audit = None

audit_report = {
    "metrics": metrics,
    "condition_counts": condition_counts,
    "per_condition_change_counts": per_condition_change_counts,
    "false_positive_rates": fp_rates,
    "label_mapping": label_mapping,
    "roc_auc_positive_class": roc_auc_positive_class,
    "generator_audit": generator_audit,
    "inconsistencies": inconsistencies,
}

# Save audit report
audit_path = os.path.join(PROJECT_ROOT, 'experiments', 'results', 'robustness_audit.json')
with open(audit_path, 'w', encoding='utf-8') as f:
    json.dump(audit_report, f, indent=2)

print(f"Audit report saved to {audit_path}")
