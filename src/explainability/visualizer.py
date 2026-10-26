"""
DeepShield Explainable AI (XAI) Visualization Engine
Overlays Grad-CAM model attention heatmaps over Log-Mel Spectrograms and computes spectral-temporal focus metrics.
"""

import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Union

def plot_explanation_overlay(
    spec_tensor: Union[torch.Tensor, np.ndarray],
    heatmap: np.ndarray,
    title: str = "Grad-CAM Model Attention Map",
    sample_rate: int = 16000,
    hop_length: int = 512,
    alpha: float = 0.5,
    cmap_spec: str = "magma",
    cmap_heatmap: str = "jet"
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    Overlays a 2D Grad-CAM heatmap over a Log-Mel Spectrogram and computes focus breakdowns.

    Args:
        spec_tensor: Spectrogram array/tensor of shape (1, n_mels, time_frames) or (n_mels, time_frames).
        heatmap: 2D numpy heatmap array normalized to [0, 1] matching spectrogram shape.
        title: Plot title.
        sample_rate: Audio sample rate in Hz (default: 16000).
        hop_length: Spectrogram hop length in samples (default: 512).
        alpha: Heatmap transparency overlay weight [0.0 - 1.0].
        cmap_spec: Base spectrogram colormap.
        cmap_heatmap: Heatmap overlay colormap.

    Returns:
        fig: Matplotlib Figure object.
        metrics: Dictionary containing peak time range (seconds), frequency focus, and summary text.
    """
    if isinstance(spec_tensor, torch.Tensor):
        spec_data = spec_tensor.squeeze(0).cpu().numpy()
    else:
        spec_data = np.squeeze(spec_tensor)

    n_mels, time_frames = spec_data.shape
    total_duration = (time_frames * hop_length) / sample_rate

    # Compute temporal attention profile (column sums)
    temporal_profile = np.sum(heatmap, axis=0)
    peak_frame = int(np.argmax(temporal_profile))
    peak_time_sec = (peak_frame * hop_length) / sample_rate
    time_window_start = max(0.0, peak_time_sec - 0.2)
    time_window_end = min(total_duration, peak_time_sec + 0.2)

    # Compute frequency attention profile (row sums)
    freq_profile = np.sum(heatmap, axis=1)
    low_band_intensity = float(np.mean(freq_profile[: n_mels // 3]))
    mid_band_intensity = float(np.mean(freq_profile[n_mels // 3 : 2 * n_mels // 3]))
    high_band_intensity = float(np.mean(freq_profile[2 * n_mels // 3 :]))

    band_dict = {
        "Low Frequencies (0-40 Mels)": low_band_intensity,
        "Mid Frequencies (40-85 Mels)": mid_band_intensity,
        "High Frequencies (85-128 Mels)": high_band_intensity
    }
    dominant_freq_band = max(band_dict, key=band_dict.get)

    metrics = {
        "peak_time_sec": peak_time_sec,
        "time_window_str": f"{time_window_start:.2f}s - {time_window_end:.2f}s",
        "dominant_freq_band": dominant_freq_band,
        "low_band_intensity": low_band_intensity,
        "mid_band_intensity": mid_band_intensity,
        "high_band_intensity": high_band_intensity,
        "summary": f"Peak attention centered around {peak_time_sec:.2f}s ({time_window_start:.2f}s–{time_window_end:.2f}s) focusing primarily on {dominant_freq_band}."
    }

    # Plot figure with dual axes
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Base Spectrogram
    ax.imshow(
        spec_data,
        aspect="auto",
        origin="lower",
        cmap=cmap_spec,
        extent=[0, total_duration, 0, n_mels]
    )

    # Heatmap Overlay
    img_hm = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        cmap=cmap_heatmap,
        alpha=alpha,
        extent=[0, total_duration, 0, n_mels]
    )

    cbar = fig.colorbar(img_hm, ax=ax, format="%0.2f")
    cbar.set_label("Grad-CAM Attention Intensity", rotation=270, labelpad=15)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel Frequency Bins")
    
    # Highlight peak attention region with vertical dashed lines
    ax.axvline(x=time_window_start, color="cyan", linestyle="--", linewidth=1.2, label="Peak Focus Window")
    ax.axvline(x=time_window_end, color="cyan", linestyle="--", linewidth=1.2)
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig, metrics

def fig_to_bytes(fig: plt.Figure) -> io.BytesIO:
    """Converts a matplotlib figure into an in-memory BytesIO PNG buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf
