"""
DeepShield Single-File Prediction Stability Analysis Engine
Evaluates prediction invariance across 5 controlled acoustic variants of a single audio input file.
"""

import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

from src.audio.degradations import (
    add_noise, apply_compression, apply_resampling_degradation
)

class SingleFileStabilityAnalyzer:
    """
    Runs multi-variant inference across 5 controlled acoustic perturbations:
    1. Original Audio
    2. Background Noise (+15 dB SNR)
    3. Mild Compression (8-bit Quantization)
    4. Resampled Audio (8 kHz)
    5. Volume Adjusted (-3 dB Gain)
    """

    def __init__(self, preprocessor, extractor, model, device):
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.model = model
        self.device = device

    def analyze(self, waveform: torch.Tensor, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Runs multi-variant inference and computes prediction stability metrics.

        Args:
            waveform: Tensor of shape (1, Samples) at sample_rate.
            sample_rate: Sample rate in Hz.

        Returns:
            Dictionary containing stability metrics, breakdown table, and summary plot.
        """
        sr = sample_rate

        # 1. Generate 5 Variants
        wf_orig = waveform.clone()
        
        # Noise
        wf_noise, sr_noise = add_noise(wf_orig, sr, snr_db=15.0)
        wf_noise_prep, _ = self.preprocessor.process_waveform(wf_noise, sr_noise)

        # Compression
        wf_comp, sr_comp = apply_compression(wf_orig, sr, bits=8)
        wf_comp_prep, _ = self.preprocessor.process_waveform(wf_comp, sr_comp)

        # Resampling
        wf_resamp, sr_resamp = apply_resampling_degradation(wf_orig, sr, target_sr=8000)
        wf_resamp_prep, _ = self.preprocessor.process_waveform(wf_resamp, sr_resamp)

        # Volume Adjust (-3 dB)
        wf_vol = wf_orig * 0.707

        variants = [
            ("Original Audio", wf_orig),
            ("Noise Added (+15dB)", wf_noise_prep),
            ("Mild Compression (8-bit)", wf_comp_prep),
            ("Resampled (8 kHz)", wf_resamp_prep),
            ("Volume Adjusted (-3dB)", wf_vol)
        ]

        results = []
        self.model.eval()

        # Run inference on all 5 variants
        with torch.no_grad():
            for name, wf in variants:
                spec = self.extractor(wf).unsqueeze(0).to(self.device)
                logits = self.model(spec)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                p_real = float(probs[0])
                p_fake = float(probs[1])
                pred_label = "AI-GENERATED" if p_fake >= 0.5 else "REAL"
                confidence = float(max(p_real, p_fake) * 100.0)

                results.append({
                    "variant_name": name,
                    "pred_label": pred_label,
                    "prob_real": p_real,
                    "prob_fake": p_fake,
                    "confidence": confidence
                })

        orig_pred = results[0]["pred_label"]
        orig_p_fake = results[0]["prob_fake"]

        matching_count = 0
        flip_count = 0

        for r in results:
            flipped = (r["pred_label"] != orig_pred)
            r["is_flipped"] = flipped
            if flipped:
                flip_count += 1
            else:
                matching_count += 1

        total_variants = len(results)
        stability_pct = float((matching_count / total_variants) * 100.0)

        # Generate bar plot figure
        fig, ax = plt.subplots(figsize=(8, 3.5))
        names = [r["variant_name"] for r in results]
        fake_probs = [r["prob_fake"] for r in results]
        colors = ["#d62728" if p >= 0.5 else "#2ca02c" for p in fake_probs]

        bars = ax.bar(names, fake_probs, color=colors, alpha=0.85, width=0.55)
        ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1.2, label="Decision Threshold (0.5)")
        ax.set_ylim([0.0, 1.05])
        ax.set_ylabel("P(AI-GENERATED)")
        ax.set_title(f"Prediction Stability Profile (Stability: {stability_pct:.0f}%, Flips: {flip_count})", fontsize=11, fontweight="bold")
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()

        # Figure buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)

        return {
            "original_prediction": orig_pred,
            "original_p_fake": orig_p_fake,
            "total_variants": total_variants,
            "matching_predictions": matching_count,
            "prediction_flips": flip_count,
            "stability_percentage": stability_pct,
            "results_list": results,
            "figure_buffer": buf
        }
