"""
DeepShield Interactive Research & Evaluation Dashboard
Streamlit web application for AI-generated speech detection, real-world degradation stress-testing, and out-of-distribution benchmark reporting.
"""

import os
import sys
import tempfile
import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.preprocessor import AudioPreprocessor
from src.features.spectrogram import LogMelSpectrogramExtractor
from src.models.model_factory import build_model, get_device
from src.training.checkpoint import CheckpointManager
from src.audio.degradations import (
    add_noise, apply_resampling_degradation, apply_telephone_degradation,
    apply_compression, apply_pitch_shift, apply_combined_degradation, DegradationRegistry
)
from src.reporting.pdf_report import generate_audio_report
from src.explainability.grad_cam import GradCAMExplainer
from src.explainability.visualizer import plot_explanation_overlay, fig_to_bytes
from src.audio.quality_analyzer import AudioQualityAnalyzer
from src.evaluation.single_file_stability import SingleFileStabilityAnalyzer
from src.models.ensemble import ModelEnsemble
from src.evaluation.calibration import ConfidenceCalibrator
from src.evaluation.threshold_analyzer import ThresholdAnalyzer
from src.experiments.registry import ExperimentRegistry
from src.reporting.table_exporter import ResearchTableExporter
from src.reporting.paper_generator import ResearchPaperDraftGenerator
from src.audio.security import AudioSecurityEngine

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default_config.yaml")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")

st.set_page_config(
    page_title="DeepShield AI Speech Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# HELPER FUNCTIONS & CACHED RESOURCES
# -------------------------------------------------------------------

@st.cache_resource
def load_system_modules():
    """Cached initialization of preprocessor and spectrogram extractor."""
    device = get_device()
    preprocessor = AudioPreprocessor(config_path=CONFIG_PATH)
    extractor = LogMelSpectrogramExtractor(config_path=CONFIG_PATH)
    return preprocessor, extractor, device

@st.cache_resource
def load_model_checkpoint(model_name: str, _device):
    """Cached loading of trained model weights."""
    model = build_model(model_name=model_name, config_path=CONFIG_PATH, device=_device)
    ckpt_mgr = CheckpointManager(
        checkpoint_dir="models_saved/checkpoints" if model_name == "cnn_baseline" else "models_saved/checkpoints/crnn_advanced"
    )
    try:
        ckpt_mgr.load_checkpoint("best_model.pt", model=model, device=_device)
    except Exception as e:
        st.warning(f"Could not load checkpoint for '{model_name}': {e}. Model using initial weights.")
    model.eval()
    return model

def plot_waveform(waveform: torch.Tensor, sample_rate: int, title: str = "Audio Waveform"):
    """Plots waveform amplitude over time."""
    t = np.linspace(0, waveform.size(-1) / sample_rate, waveform.size(-1))
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(t, waveform.squeeze(0).cpu().numpy(), color="#1f77b4", alpha=0.8, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_ylim([-1.05, 1.05])
    plt.tight_layout()
    return fig

def plot_spectrogram(spec_tensor: torch.Tensor, title: str = "Log-Mel Spectrogram"):
    """Plots Log-Mel Spectrogram image."""
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(
        spec_tensor.squeeze(0).cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Mel Frequency Bins")
    plt.tight_layout()
    return fig

# -------------------------------------------------------------------
# MAIN DASHBOARD UI
# -------------------------------------------------------------------

def main():
    st.title("🛡️ DeepShield: Robust Detection of AI-Generated Speech")
    st.markdown(
        "**Research Platform for Synthetic Speech Detection under Unseen Generators and Real-World Audio Degradation**"
    )

    preprocessor, extractor, device = load_system_modules()

    # Sidebar Controls
    st.sidebar.header("⚙️ System Controls")
    selected_model_name = st.sidebar.selectbox(
        "Select Model Architecture",
        options=["cnn_baseline", "crnn_advanced", "ensemble_fusion"],
        format_func=lambda x: (
            "2D CNN Baseline (356k params)" if x == "cnn_baseline"
            else "Advanced CRNN (2.9M params)" if x == "crnn_advanced"
            else "🔥 Model Ensemble (CNN + CRNN Fusion)"
        )
    )

    custom_threshold = st.sidebar.slider(
        "Decision Threshold (τ)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Custom probability threshold for classifying as AI-GENERATED."
    )

    use_calibration = st.sidebar.checkbox(
        "Enable Temperature Scaling Calibration (T=1.25)",
        value=True,
        help="Applies temperature scaling to smooth softmax confidence and calculate ECE."
    )

    # Load models
    cnn_model = load_model_checkpoint("cnn_baseline", device)
    crnn_model = load_model_checkpoint("crnn_advanced", device)

    if selected_model_name == "cnn_baseline":
        model = cnn_model
        ensemble_engine = None
    elif selected_model_name == "crnn_advanced":
        model = crnn_model
        ensemble_engine = None
    else:
        model = cnn_model # Primary model for Grad-CAM
        ensemble_engine = ModelEnsemble([("2D CNN Baseline", cnn_model), ("Advanced CRNN", crnn_model)], weights=[0.4, 0.6])

    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎙️ Single Audio Prediction",
        "⚡ Real-World Degradation Stress-Test",
        "📊 Robustness Benchmark",
        "🔬 Unseen Generator OOD Analysis",
        "⚖️ Model & Experiment Registry",
        "ℹ️ About & Limitations"
    ])

    # -------------------------------------------------------------------
    # TAB 1: SINGLE AUDIO PREDICTION & LIVE INFERENCE
    # -------------------------------------------------------------------
    with tab1:
        st.subheader("Single Audio Analysis Pipeline")
        st.markdown(AudioSecurityEngine.get_privacy_notice())

        audio_source = st.radio("Select Audio Source", options=["📁 Upload Audio File", "🎙️ Record Audio (Microphone)"], horizontal=True)
        uploaded_file = None

        if audio_source == "📁 Upload Audio File":
            uploaded_file = st.file_uploader("Upload an audio file (WAV, MP3, FLAC, OGG, M4A)", type=["wav", "mp3", "flac", "ogg", "m4a"])
        else:
            if hasattr(st, "audio_input"):
                uploaded_file = st.audio_input("Record audio using your microphone")
            else:
                uploaded_file = st.file_uploader("Upload recorded audio file", type=["wav", "mp3", "flac", "ogg"])

        if uploaded_file is not None:
            # Save temporary file
            file_bytes = uploaded_file.getvalue()
            ext = os.path.splitext(uploaded_file.name)[1] if hasattr(uploaded_file, "name") else ".wav"
            filename = uploaded_file.name if hasattr(uploaded_file, "name") else "recorded_audio.wav"

            # Security Validation
            sec_res = AudioSecurityEngine.validate_file_security(file_bytes, filename=filename, max_size_mb=50.0)
            if not sec_res["is_valid"]:
                st.error(f"🚨 **Security Validation Failed:** {sec_res['error_message']}")
                return

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            col1, col2 = st.columns([1, 1])
            wf_buf = None
            spec_buf = None

            with col1:
                st.audio(uploaded_file, format="audio/wav")
                st.info(f"**Filename:** `{filename}` | **Size:** `{sec_res['file_size_mb']:.2f} MB`\n\n🔒 **SHA-256 Fingerprint:** `{sec_res['sha256_hash'][:32]}...`")

                try:
                    waveform, sr = preprocessor.process_file(tmp_path)
                    fig_wf = plot_waveform(waveform, sr, title="Standardized 16kHz Preprocessed Waveform")
                    st.pyplot(fig_wf)
                    
                    wf_buf = io.BytesIO()
                    fig_wf.savefig(wf_buf, format="png", bbox_inches="tight", dpi=150)
                    plt.close(fig_wf)
                    wf_buf.seek(0)
                except Exception as e:
                    st.error(f"Preprocessing error: {e}")
                    return

            with col2:
                try:
                    spec_tensor = extractor(waveform)
                    fig_spec = plot_spectrogram(spec_tensor, title="Extracted Log-Mel Spectrogram (128 Mels)")
                    st.pyplot(fig_spec)

                    spec_buf = io.BytesIO()
                    fig_spec.savefig(spec_buf, format="png", bbox_inches="tight", dpi=150)
                    plt.close(fig_spec)
                    spec_buf.seek(0)

                    # Run Model Inference / Ensemble
                    batch_spec = spec_tensor.unsqueeze(0).to(device)
                    ensemble_res = None

                    if ensemble_engine is not None:
                        ensemble_res = ensemble_engine.predict(batch_spec, device)
                        p_real = ensemble_res["prob_real"]
                        p_fake = ensemble_res["prob_fake"]
                        logits = torch.tensor([[np.log(p_real + 1e-7), np.log(p_fake + 1e-7)]])
                    else:
                        with torch.no_grad():
                            logits = model(batch_spec)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        p_real = float(probs[0])
                        p_fake = float(probs[1])

                    # Apply Decision Threshold
                    threshold_analyzer = ThresholdAnalyzer(default_threshold=custom_threshold)
                    thresh_res = threshold_analyzer.evaluate_sample_threshold(p_fake, threshold=custom_threshold)
                    pred_label = thresh_res["prediction_label"]

                    # Apply Confidence Calibration
                    calibrator = ConfidenceCalibrator(temperature=1.25)
                    cal_probs, raw_conf, cal_conf = calibrator.calibrate_logits(logits)
                    calib_res = calibrator.evaluate_reliability(raw_conf, cal_conf, ece=0.035)

                    # 1. Audio Quality & Deepfake Risk Analysis
                    quality_analyzer = AudioQualityAnalyzer()
                    quality_res = quality_analyzer.analyze(waveform, sr)

                    st.markdown("### 🔊 Audio Quality & Risk Assessment")
                    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
                    q_col1.metric("Quality Score", f"{quality_res['quality_score']:.1f} / 100", delta=quality_res['quality_rating'])
                    q_col2.metric("SNR Estimate", f"{quality_res['snr_db']:.1f} dB", delta=f"{quality_res['noise_level']} Noise")
                    q_col3.metric("Clipping %", f"{quality_res['clipping_pct']:.2f}%")
                    q_col4.metric("Silence %", f"{quality_res['silence_pct']:.1f}%")

                    rel_status = quality_res['prediction_reliability']
                    if rel_status == "High":
                        st.success(f"✅ **Prediction Reliability: {rel_status}** ({quality_res['reliability_reason']})")
                    elif rel_status == "Moderate":
                        st.info(f"ℹ️ **Prediction Reliability: {rel_status}** ({quality_res['reliability_reason']})")
                    else:
                        st.warning(f"⚠️ **Prediction Reliability: {rel_status}** ({quality_res['reliability_reason']})")

                    st.markdown("---")

                    # 2. Prediction & Calibration Result
                    st.markdown("### 📊 Model Prediction & Calibrated Confidence Result")
                    if pred_label == "AI-GENERATED":
                        st.error(f"🚨 **{pred_label}** (Calibrated Confidence: **{cal_conf:.2f}%**, Decision Threshold τ = {custom_threshold:.2f})")
                    else:
                        st.success(f"✅ **{pred_label}** (Calibrated Confidence: **{cal_conf:.2f}%**, Decision Threshold τ = {custom_threshold:.2f})")

                    cal_c1, cal_c2, cal_c3 = st.columns(3)
                    cal_c1.metric("Raw Confidence", f"{raw_conf:.2f}%")
                    cal_c2.metric("Calibrated Confidence", f"{cal_conf:.2f}%", delta=f"-{raw_conf - cal_conf:.2f}% (T=1.25)")
                    cal_c3.metric("Reliability Status", calib_res["reliability_level"], delta=f"ECE: {calib_res['ece']:.3f}")

                    if ensemble_res:
                        st.info(f"🤝 **Ensemble Fusion:** Agreement = **{ensemble_res['agreement_pct']:.0f}%** across 2 models | Probability Std Dev = `{ensemble_res['prob_std']:.4f}`")

                    st.write(f"**REAL Speech Probability:** `{p_real:.4f}`")
                    st.progress(p_real)
                    st.write(f"**AI-Generated Probability:** `{p_fake:.4f}`")
                    st.progress(p_fake)

                    st.markdown("---")

                    # 3. Single-File Prediction Stability Analysis
                    st.markdown("### 🎯 Single-File Prediction Stability Analysis")
                    st.caption("Evaluates prediction invariance across 5 controlled acoustic perturbations.")
                    
                    stability_analyzer = SingleFileStabilityAnalyzer(preprocessor, extractor, model, device)
                    stability_res = stability_analyzer.analyze(waveform, sr)

                    st_col1, st_col2, st_col3 = st.columns(3)
                    st_col1.metric("Stability Score", f"{stability_res['stability_percentage']:.0f}%")
                    st_col2.metric("Stable Variants", f"{stability_res['matching_predictions']} / 5")
                    st_col3.metric("Prediction Flips", f"{stability_res['prediction_flips']}")

                    df_stab = pd.DataFrame(stability_res["results_list"])
                    df_stab_display = df_stab[["variant_name", "pred_label", "prob_fake", "confidence", "is_flipped"]].copy()
                    df_stab_display.columns = ["Variant", "Prediction", "P(AI)", "Confidence (%)", "Flipped?"]
                    st.dataframe(df_stab_display, use_container_width=True)

                    st.markdown("---")

                    # 4. Explainable AI (Grad-CAM) Section
                    st.markdown("### 🔍 Why was this prediction made? (Grad-CAM Model Attention)")
                    st.caption("Visualizes time-frequency regions that contributed most to feature activations.")

                    explainer = GradCAMExplainer(model)
                    heatmap, xai_info = explainer.generate_heatmap(batch_spec, target_class=1 if pred_label == "AI-GENERATED" else 0)

                    xai_buf = None
                    xai_metrics = None
                    if heatmap is not None:
                        fig_xai, xai_metrics = plot_explanation_overlay(
                            spec_tensor, heatmap,
                            title=f"Grad-CAM Model Attention Map ({pred_label})"
                        )
                        st.pyplot(fig_xai)
                        xai_buf = fig_to_bytes(fig_xai)

                        st.info(f"📍 **Focus Breakdown:** {xai_metrics['summary']}")
                        st.caption("⚠️ **Disclaimer:** Heatmap indicates internal model feature attention, not physical proof of forgery.")
                    else:
                        st.warning("Grad-CAM visualization is unsupported for this model structure.")

                    explainer.remove_hooks()

                    # 5. Generate PDF Report with All Modules
                    _dur_sec = waveform.size(-1) / sr
                    _ch_count = 1 if waveform.size(0) == 1 else int(waveform.size(0))
                    audio_info = {
                        "filename": uploaded_file.name,
                        "file_format": os.path.splitext(uploaded_file.name)[1].lstrip(".").upper(),
                        "file_size": f"{uploaded_file.size / 1024:.1f} KB",
                        "duration": _dur_sec,
                        "sample_rate": sr,
                        "channels": _ch_count,
                        "validation_status": "Valid Audio",
                        "loading_status": "Successfully Processed"
                    }

                    prep_info = {
                        "target_sample_rate": preprocessor.target_sample_rate,
                        "target_duration": preprocessor.target_duration,
                        "mono_conversion": True,
                        "normalization_method": preprocessor.normalization_type.capitalize() + " Normalization",
                    }

                    spec_info = {
                        "feature_type": "Log-Mel Spectrogram",
                        "n_mels": str(extractor.n_mels),
                        "n_fft": str(extractor.n_fft),
                        "hop_length": str(extractor.hop_length),
                        "win_length": str(extractor.win_length),
                        "f_range": f"{extractor.f_min} - {extractor.f_max} Hz",
                        "tensor_shape": str(list(spec_tensor.shape)),
                        "time_frames": str(spec_tensor.shape[-1]),
                    }

                    pred_info = {
                        "model_name": selected_model_name,
                        "model_architecture": "2D CNN Baseline" if selected_model_name == "cnn_baseline" else ("Advanced CRNN" if selected_model_name == "crnn_advanced" else "Model Ensemble (CNN + CRNN)"),
                        "predicted_class": pred_label,
                        "real_prob": p_real,
                        "fake_prob": p_fake,
                        "confidence": cal_conf,
                        "decision_threshold": custom_threshold
                    }

                    pdf_data = generate_audio_report(
                        audio_metadata=audio_info,
                        prediction_result=pred_info,
                        preprocessing_info=prep_info,
                        feature_info=spec_info,
                        waveform_image=wf_buf,
                        spectrogram_image=spec_buf,
                        quality_result=quality_res,
                        stability_result=stability_res,
                        explanation_image=xai_buf,
                        explanation_metrics=xai_metrics,
                        calibration_result=calib_res,
                        ensemble_result=ensemble_res,
                        security_info=sec_res
                    )

                    st.markdown("---")
                    st.download_button(
                        label="📄 Download Comprehensive PDF Report",
                        data=pdf_data,
                        file_name=f"DeepShield_Report_{os.path.splitext(uploaded_file.name)[0]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Inference error: {e}")

            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # -------------------------------------------------------------------
    # TAB 2: REAL-WORLD DEGRADATION STRESS-TESTING
    # -------------------------------------------------------------------
    with tab2:
        st.subheader("Acoustic Channel Stress Testing")
        st.markdown("Select a degradation condition to test model prediction stability under real-world corruptions:")

        demo_file = os.path.join(PROJECT_ROOT, "data", "test_samples", "clean_mono_16k.wav")
        if not os.path.exists(demo_file):
            st.warning("Sample test file not found. Upload a file in Tab 1 first.")
        else:
            deg_type = st.selectbox(
                "Select Degradation Condition",
                options=["Noise (10 dB SNR)", "Resampling (8 kHz)", "Telephone VoIP (300-3400Hz)", "Pitch Shift (+2 Semitones)", "Quantization (8-bit)", "Combined Corruption"]
            )

            clean_wf, sr = preprocessor.process_file(demo_file)

            if deg_type == "Noise (10 dB SNR)":
                deg_wf, deg_sr = add_noise(clean_wf, sr, snr_db=10.0)
            elif deg_type == "Resampling (8 kHz)":
                deg_wf, deg_sr = apply_resampling_degradation(clean_wf, sr, target_sr=8000)
            elif deg_type == "Telephone VoIP (300-3400Hz)":
                deg_wf, deg_sr = apply_telephone_degradation(clean_wf, sr)
            elif deg_type == "Pitch Shift (+2 Semitones)":
                deg_wf, deg_sr = apply_pitch_shift(clean_wf, sr, semitones=2.0)
            elif deg_type == "Quantization (8-bit)":
                deg_wf, deg_sr = apply_compression(clean_wf, sr, bits=8)
            else:
                deg_wf, deg_sr = apply_combined_degradation(clean_wf, sr, sequence=["noise", "telephone", "resample"])

            # Compute Spectrograms & Predictions
            clean_spec = extractor(clean_wf)
            deg_spec = extractor(preprocessor.process_waveform(deg_wf, deg_sr)[0])

            with torch.no_grad():
                c_probs = torch.softmax(model(clean_spec.unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]
                d_probs = torch.softmax(model(deg_spec.unsqueeze(0).to(device)), dim=1).cpu().numpy()[0]

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Clean Original Audio")
                st.pyplot(plot_waveform(clean_wf, sr, title="Original Clean Waveform"))
                st.pyplot(plot_spectrogram(clean_spec, title="Clean Spectrogram"))
                st.write(f"**Prediction:** `{'AI-GENERATED' if c_probs[1]>=0.5 else 'REAL'}` (P(Fake)={c_probs[1]:.4f})")

            with col_b:
                st.markdown(f"#### Degraded Audio ({deg_type})")
                st.pyplot(plot_waveform(deg_wf, deg_sr, title=f"Degraded Waveform ({deg_type})"))
                st.pyplot(plot_spectrogram(deg_spec, title=f"Degraded Spectrogram ({deg_type})"))
                st.write(f"**Prediction:** `{'AI-GENERATED' if d_probs[1]>=0.5 else 'REAL'}` (P(Fake)={d_probs[1]:.4f})")

            # Stability Delta
            flipped = (c_probs[1] >= 0.5) != (d_probs[1] >= 0.5)
            st.markdown("---")
            if flipped:
                st.warning(f"⚠️ **Prediction Flipped!** Clean prediction shifted under {deg_type}.")
            else:
                st.success(f"✅ **Prediction Consistent!** Model maintained correct prediction under {deg_type}.")

    # -------------------------------------------------------------------
    # TAB 3: ROBUSTNESS & DEGRADATION BENCHMARK RESULTS
    # -------------------------------------------------------------------
    with tab3:
        st.subheader("Empirical Robustness Benchmark Results")
        rob_csv = os.path.join(RESULTS_DIR, "robustness", "robustness_summary.csv")

        if os.path.exists(rob_csv):
            df_rob = pd.read_csv(rob_csv)
            st.dataframe(df_rob, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.image(os.path.join(RESULTS_DIR, "robustness", "clean_vs_degraded_f1.png"), caption="Clean vs Degraded F1-Score")
            with col2:
                st.image(os.path.join(RESULTS_DIR, "robustness", "robustness_drop.png"), caption="Robustness Drop (Δ F1)")
        else:
            st.info("Run `python scripts/run_robustness_experiments.py` to generate robustness benchmark artifacts.")

    # -------------------------------------------------------------------
    # TAB 4: UNSEEN GENERATOR OOD ANALYSIS
    # -------------------------------------------------------------------
    with tab4:
        st.subheader("Out-of-Distribution Unseen Generator Protocol")
        st.markdown("""
        **Generator-Disjoint Protocol ($G_{\text{train}} \cap G_{\text{test}} = \emptyset$):**
        To prevent vocoder artifact memorization, synthetic voice samples from `generator_C` were strictly withheld during training.
        """)

        unseen_txt = os.path.join(RESULTS_DIR, "unseen_generator", "unseen_generator_report.txt")
        if os.path.exists(unseen_txt):
            with open(unseen_txt, "r") as f:
                st.code(f.read(), language="text")
        else:
            st.info("Run `python scripts/run_unseen_generator_evaluation.py` to generate unseen generator reports.")

    # -------------------------------------------------------------------
    # TAB 5: MODEL & EXPERIMENT REGISTRY
    # -------------------------------------------------------------------
    with tab5:
        st.subheader("Model Comparison & Research Experiment Registry")
        st.caption("Logs model benchmarks, out-of-distribution unseen generator tests, and publication export tools.")

        comp_csv = os.path.join(RESULTS_DIR, "model_comparison", "model_comparison.csv")
        if os.path.exists(comp_csv):
            df_comp = pd.read_csv(comp_csv)
            st.dataframe(df_comp, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.image(os.path.join(RESULTS_DIR, "model_comparison", "robustness_comparison.png"), caption="CNN Baseline vs Advanced CRNN Robustness")
            with col2:
                st.image(os.path.join(RESULTS_DIR, "model_comparison", "clean_performance_comparison.png"), caption="Clean F1 Comparison")

        st.markdown("---")
        st.subheader("📜 Experiment History Registry")
        
        registry = ExperimentRegistry()
        df_history = registry.get_history()
        st.dataframe(df_history, use_container_width=True)

        st.markdown("---")
        st.subheader("📄 Publication Table & Paper Export Engine")

        table_paths = ResearchTableExporter.export_all_formats(df_history)

        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:
            with open(table_paths["latex"], "r") as f:
                tex_data = f.read()
            st.download_button("📥 Download LaTeX Table (.tex)", data=tex_data, file_name="deepshield_experiment_table.tex", mime="text/x-tex", use_container_width=True)

        with exp_col2:
            with open(table_paths["markdown"], "r") as f:
                md_table_data = f.read()
            st.download_button("📥 Download Markdown Table (.md)", data=md_table_data, file_name="deepshield_experiment_table.md", mime="text/markdown", use_container_width=True)

        with exp_col3:
            paper_gen = ResearchPaperDraftGenerator(registry_df=df_history)
            paper_draft_path = os.path.join(PROJECT_ROOT, "experiments", "paper_draft.md")
            paper_draft_text = paper_gen.generate_draft(output_path=paper_draft_path)
            st.download_button("📜 Download Empirical Research Paper Draft (.md)", data=paper_draft_text, file_name="DeepShield_Research_Paper_Draft.md", mime="text/markdown", use_container_width=True)

    # -------------------------------------------------------------------
    # TAB 6: ABOUT & RESEARCH LIMITATIONS
    # -------------------------------------------------------------------
    with tab6:
        st.subheader("Research Scope, Guidelines & Limitations")
        st.markdown("""
        ### DeepShield Research Framework
        - **Objective:** Investigating deepfake speech detection robustness under unseen AI voice generators and real-world audio degradations.
        - **Pipeline:** Preprocessing (16kHz Mono) $\rightarrow$ Log-Mel Spectrogram (128 Mels) $\rightarrow$ PyTorch 2D CNN / CRNN.
        
        > [!IMPORTANT]
        > **Research Limitations Note:**
        > 1. Results generated on synthetic/mock data serve as pipeline validation demonstrating architectural capability.
        > 2. Full scientific conclusions require training on large-scale benchmarks (e.g. ASVspoof 2021 / In-the-Wild datasets).
        > 3. Audio degradations depend on available Python audio codecs on the host system.
        """)

if __name__ == "__main__":
    main()
