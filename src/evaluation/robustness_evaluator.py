"""
DeepShield Robustness Evaluator Module
Executes multi-degradation robustness benchmarking and calculates Robustness Drop metrics.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor
from src.audio.degradations import (
    add_noise, apply_resampling_degradation, apply_telephone_degradation,
    apply_compression, apply_pitch_shift, apply_combined_degradation
)
from src.evaluation.evaluator import DeepShieldEvaluator
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.generator_evaluator import evaluate_by_generator
from src.evaluation.stability import PredictionStabilityAnalyzer

logger = logging.getLogger("DeepShield.RobustnessEvaluator")

class RobustnessEvaluator:
    """Executes multi-degradation robustness benchmarking and calculates Robustness Drop metrics."""

    def __init__(self, model: torch.nn.Module, device: torch.device, config_path: str = "config/default_config.yaml"):
        self.model = model
        self.device = device
        self.preprocessor = AudioPreprocessor(config_path=config_path)
        self.extractor = LogMelSpectrogramExtractor(config_path=config_path)
        self.evaluator = DeepShieldEvaluator(model=self.model, device=self.device)

    def evaluate_waveform_batch(self, waveforms: torch.Tensor, sample_rate: int = 16000) -> np.ndarray:
        """Helper method: Converts waveform batch into Log-Mel features and computes model logits/probs."""
        self.model.eval()
        with torch.no_grad():
            spec_list = []
            for i in range(waveforms.size(0)):
                wf = waveforms[i:i+1] # (1, Time)
                wf_proc, sr_proc = self.preprocessor.process_waveform(wf, sample_rate)
                spec = self.extractor(wf_proc) # (1, 128, 94)
                spec_list.append(spec)

            batch_specs = torch.stack(spec_list, dim=0).to(self.device) # (B, 1, 128, 94)
            logits = self.model(batch_specs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

    def run_degraded_dataset_evaluation(self, test_loader, degradation_fn=None, condition_name: str = "clean") -> tuple[dict, pd.DataFrame]:
        """Runs evaluation over a DataLoader, optionally applying a degradation function to audio waveforms."""
        records = []
        all_true = []
        all_pred = []
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            for batch_features, batch_labels, batch_meta in test_loader:
                if degradation_fn is None:
                    # Clean features passed directly
                    batch_features = batch_features.to(self.device)
                    logits = self.model(batch_features)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                else:
                    # Load audio waveforms, apply degradation, re-extract features
                    degraded_specs = []
                    for i in range(len(batch_meta)):
                        file_path = batch_meta[i]["file_path"]
                        wf, sr = self.preprocessor.loader.load_audio(file_path)
                        deg_wf, deg_sr = degradation_fn(wf, sr)
                        proc_wf, p_sr = self.preprocessor.process_waveform(deg_wf, deg_sr)
                        spec = self.extractor(proc_wf)
                        degraded_specs.append(spec)

                    batch_specs = torch.stack(degraded_specs, dim=0).to(self.device)
                    logits = self.model(batch_specs)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

                preds = np.argmax(probs, axis=1)

                for i in range(len(batch_meta)):
                    meta = batch_meta[i]
                    p_real = float(probs[i, 0])
                    p_fake = float(probs[i, 1])
                    conf = float(np.max(probs[i]))

                    rec = {
                        "condition": condition_name,
                        "file_path": meta["file_path"],
                        "filename": meta["filename"],
                        "label": meta["label"],
                        "label_id": int(batch_labels[i].item()),
                        "pred_label_id": int(preds[i]),
                        "pred_label": "fake" if preds[i] == 1 else "real",
                        "prob_real": round(p_real, 4),
                        "prob_fake": round(p_fake, 4),
                        "confidence": round(conf, 4),
                        "generator": meta["generator"],
                        "split": meta["split"]
                    }
                    records.append(rec)
                    all_true.append(batch_labels[i].item())
                    all_pred.append(preds[i])
                    all_probs.append([p_real, p_fake])

        df_preds = pd.DataFrame(records)
        metrics = compute_classification_metrics(np.array(all_true), np.array(all_pred), np.array(all_probs))
        metrics["condition"] = condition_name
        metrics["num_samples"] = len(df_preds)
        return metrics, df_preds

    def run_full_robustness_suite(self, test_loader, output_dir: str = "experiments/results/robustness") -> pd.DataFrame:
        """Runs clean baseline + 9 degradation conditions and computes Robustness Drop metrics."""
        os.makedirs(output_dir, exist_ok=True)
        cm_dir = os.path.join(output_dir, "confusion_matrices")
        os.makedirs(cm_dir, exist_ok=True)

        conditions = [
            ("Clean Audio", None),
            ("Noise 20dB", lambda w, s: add_noise(w, s, snr_db=20.0)),
            ("Noise 10dB", lambda w, s: add_noise(w, s, snr_db=10.0)),
            ("Noise 5dB", lambda w, s: add_noise(w, s, snr_db=5.0)),
            ("Resample 8kHz", lambda w, s: apply_resampling_degradation(w, s, target_sr=8000)),
            ("Resample 12kHz", lambda w, s: apply_resampling_degradation(w, s, target_sr=12000)),
            ("Telephone VoIP", lambda w, s: apply_telephone_degradation(w, s)),
            ("Pitch +2", lambda w, s: apply_pitch_shift(w, s, semitones=2.0)),
            ("Pitch -2", lambda w, s: apply_pitch_shift(w, s, semitones=-2.0)),
            ("Compression 8-bit", lambda w, s: apply_compression(w, s, bits=8)),
            ("Combined Degradation", lambda w, s: apply_combined_degradation(w, s, sequence=["noise", "telephone", "resample"]))
        ]

        summary_records = []
        gen_records = []
        stability_records = []

        df_clean_preds = None
        clean_f1 = 0.0

        for idx, (cond_name, deg_fn) in enumerate(conditions):
            logger.info(f"Running robustness condition [{idx+1}/{len(conditions)}]: '{cond_name}'...")
            metrics, df_preds = self.run_degraded_dataset_evaluation(test_loader, degradation_fn=deg_fn, condition_name=cond_name)

            if cond_name == "Clean Audio":
                df_clean_preds = df_preds
                clean_f1 = metrics["f1_score"]

            # Calculate Robustness Drop & Percentage Drop
            f1 = metrics["f1_score"]
            rob_drop = round(clean_f1 - f1, 4)
            pct_drop = round((rob_drop / clean_f1) * 100.0, 2) if clean_f1 > 1e-6 else 0.0

            # Calculate Stability
            stab_summary, _ = PredictionStabilityAnalyzer.analyze_pair(df_clean_preds, df_preds, condition_name=cond_name)
            stability_records.append(stab_summary)

            rec = {
                "condition": cond_name,
                "num_samples": metrics["num_samples"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "robustness_drop": rob_drop,
                "percentage_drop": pct_drop,
                "flip_rate": stab_summary["flip_rate"],
                "mean_conf_change": stab_summary["mean_conf_change"]
            }
            summary_records.append(rec)

            # Per-Generator breakdown
            df_gen = evaluate_by_generator(df_preds, condition_name=cond_name)
            gen_records.append(df_gen)

        df_summary = pd.DataFrame(summary_records)
        df_gen_summary = pd.concat(gen_records, ignore_index=True)
        df_stability = pd.DataFrame(stability_records)

        # Save Output Files
        df_summary.to_csv(os.path.join(output_dir, "robustness_summary.csv"), index=False)
        df_gen_summary.to_csv(os.path.join(output_dir, "generator_wise_metrics.csv"), index=False)
        df_stability.to_csv(os.path.join(output_dir, "prediction_stability.csv"), index=False)

        with open(os.path.join(output_dir, "robustness_summary.json"), "w") as f:
            json.dump(summary_records, f, indent=4)

        # Plot 1: Clean vs Degraded F1
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_summary, x="condition", y="f1_score", hue="condition", palette="viridis", legend=False)
        plt.axhline(clean_f1, color="red", linestyle="--", label=f"Clean F1 ({clean_f1:.2f})")
        plt.title("DeepShield Clean vs Degraded F1-Score Performance")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("F1-Score")
        plt.ylim([0.0, 1.05])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "clean_vs_degraded_f1.png"), dpi=150)
        plt.close()

        # Plot 2: Robustness Drop
        df_drop = df_summary[df_summary["condition"] != "Clean Audio"]
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_drop, x="condition", y="robustness_drop", hue="condition", palette="Reds", legend=False)
        plt.title("DeepShield Robustness Drop (Clean F1 - Degraded F1)")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Robustness Drop (Δ F1)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "robustness_drop.png"), dpi=150)
        plt.close()

        logger.info(f"Robustness experiment suite finished! Exported to '{output_dir}'.")
        return df_summary
