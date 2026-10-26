"""
DeepShield PDF Report Generator Module
Generates publication-quality technical PDF analysis reports for audio authenticity predictions.
"""

import os
import io
import uuid
import logging
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

logger = logging.getLogger("DeepShield.PDFReport")

def generate_audio_report(
    audio_metadata: dict,
    prediction_result: dict,
    preprocessing_info: dict,
    feature_info: dict,
    waveform_image=None,
    spectrogram_image=None,
    degradation_result: dict = None,
    quality_result: dict = None,
    stability_result: dict = None,
    explanation_image=None,
    explanation_metrics: dict = None,
    calibration_result: dict = None,
    ensemble_result: dict = None,
    security_info: dict = None
) -> bytes:
    """
    Generates a PDF report as bytes for a given audio analysis.

    Parameters:
    - audio_metadata: dict with keys (filename, file_format, file_size, duration, sample_rate, channels, validation_status)
    - prediction_result: dict with keys (model_name, predicted_class, real_prob, fake_prob, confidence)
    - preprocessing_info: dict with keys (target_sample_rate, target_duration, mono_conversion, normalization_method)
    - feature_info: dict with keys (feature_type, n_mels, n_fft, hop_length, win_length, f_range, tensor_shape)
    - waveform_image: str (path) or BytesIO/bytes object containing PNG image
    - spectrogram_image: str (path) or BytesIO/bytes object containing PNG image
    - degradation_result: optional dict with keys (condition, orig_pred, deg_pred, orig_conf, deg_conf, flipped, conf_change)

    Returns:
    - bytes: The compiled PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1E3A8A")   # Dark Navy
    c_secondary = colors.HexColor("#3B82F6") # Blue
    c_real = colors.HexColor("#16A34A")      # Green
    c_fake = colors.HexColor("#DC2626")      # Red
    c_dark = colors.HexColor("#1F2937")      # Dark Grey
    c_light = colors.HexColor("#F3F4F6")     # Light Grey Background

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563")
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_dark
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6B7280")
    )

    story = []
    report_id = f"DS-{uuid.uuid4().hex[:8].upper()}"
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Header Section
    story.append(Paragraph("🛡️ DeepShield", title_style))
    story.append(Paragraph("Audio Authenticity Analysis & Robustness Report", ParagraphStyle('SubHeader', parent=title_style, fontSize=14, leading=18, textColor=c_secondary)))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Report ID:</b> {report_id} &nbsp;|&nbsp; <b>Generated:</b> {gen_time}", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=0, spaceAfter=10))

    # 2. Executive Prediction Summary Banner
    pred_class = prediction_result.get("predicted_class", "UNKNOWN").upper()
    confidence = prediction_result.get("confidence", 0.0)
    real_p = prediction_result.get("real_prob", 0.0)
    fake_p = prediction_result.get("fake_prob", 0.0)
    model_name = prediction_result.get("model_name", "CNN Baseline")

    banner_color = c_fake if pred_class in ["AI-GENERATED", "FAKE"] else c_real
    banner_text = f"<font size=14 color='{banner_color.hexval()}'><b>PREDICTION: {pred_class}</b></font><br/><font size=10 color='#1F2937'>Confidence: <b>{confidence:.2f}%</b> (REAL: {real_p*100:.1f}% | AI-GENERATED: {fake_p*100:.1f}%)</font>"

    summary_table = Table(
        [[Paragraph(banner_text, body_style)]],
        colWidths=[532]
    )
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_light),
        ('BOX', (0,0), (-1,-1), 1, banner_color),
        ('PADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # Helper for 2-column key-value tables
    def make_kv_table(data_dict, width1=120, width2=140):
        table_data = []
        for k, v in data_dict.items():
            table_data.append([
                Paragraph(f"<b>{k}</b>", body_style),
                Paragraph(str(v), body_style)
            ])
        t = Table(table_data, colWidths=[width1, width2])
        t.setStyle(TableStyle([
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        return t

    # 3. Audio Info & Preprocessing Info (Side-by-side Table)
    story.append(Paragraph("1. Audio Metadata & Standardization", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

    sha256_val = security_info.get("sha256_hash", "N/A") if security_info else audio_metadata.get("sha256_hash", "N/A")
    audio_kv = {
        "Filename": audio_metadata.get("filename", "N/A"),
        "Format": str(audio_metadata.get("file_format", "N/A")).upper(),
        "Duration": f"{audio_metadata.get('duration', 0.0):.2f} sec",
        "Original Sample Rate": f"{audio_metadata.get('sample_rate', 0)} Hz",
        "Channels": "Mono (1)" if audio_metadata.get("channels") == 1 else f"{audio_metadata.get('channels', 1)} Channels",
        "SHA-256 Fingerprint": f"{sha256_val[:16]}..." if sha256_val != "N/A" else "N/A"
    }

    target_sr = preprocessing_info.get('target_sample_rate', 16000)
    target_dur = preprocessing_info.get('target_duration', 5.0)
    target_samples = int(target_sr * target_dur)

    preproc_kv = {
        "Target Sample Rate": f"{target_sr} Hz",
        "Target Duration": f"{target_dur} sec",
        "Mono Conversion": "Executed" if preprocessing_info.get("mono_conversion", True) else "Not Required",
        "Normalization": preprocessing_info.get("normalization_method", "Peak Normalization"),
        "Duration Padding/Crop": f"Fixed {target_samples:,} Samples",
        "Execution Device": "PyTorch CPU/CUDA"
    }

    t_audio = make_kv_table(audio_kv, width1=110, width2=145)
    t_preproc = make_kv_table(preproc_kv, width1=120, width2=145)

    meta_grid = Table(
        [[t_audio, t_preproc]],
        colWidths=[260, 272]
    )
    meta_grid.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(meta_grid)
    story.append(Spacer(1, 10))

    # 4. Feature Extraction Parameters
    story.append(Paragraph("2. Acoustic Feature Extraction (Log-Mel Spectrogram)", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

    feat_kv1 = {
        "Feature Representation": feature_info.get("feature_type", "Log-Mel Spectrogram"),
        "Mel Frequency Bins": f"{feature_info.get('n_mels', 128)} bins",
        "FFT Window Size": f"{feature_info.get('n_fft', 1024)} samples",
        "Hop Length": f"{feature_info.get('hop_length', 512)} samples"
    }
    feat_kv2 = {
        "Frequency Range": f"{feature_info.get('f_range', '0 - 8000 Hz')}",
        "Decibel Scaling": "Power to dB (80 dB Top)",
        "Feature Normalization": "Z-score Instance Normalization",
        "Output Tensor Shape": f"{feature_info.get('tensor_shape', '[1, 128, 94]')}"
    }

    t_feat1 = make_kv_table(feat_kv1, width1=110, width2=145)
    t_feat2 = make_kv_table(feat_kv2, width1=120, width2=145)

    feat_grid = Table(
        [[t_feat1, t_feat2]],
        colWidths=[260, 272]
    )
    feat_grid.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(feat_grid)
    story.append(Spacer(1, 10))

    # 5. Audio Visualizations (Waveform & Spectrogram Images)
    story.append(Paragraph("3. Acoustic Signal & Feature Visualizations", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

    def process_image_input(img_input, target_width=512, target_height=140):
        if img_input is None:
            return None
        if isinstance(img_input, (str, os.PathLike)):
            if os.path.exists(img_input):
                return Image(img_input, width=target_width, height=target_height)
        elif isinstance(img_input, (io.BytesIO, bytes)):
            buf = img_input if isinstance(img_input, io.BytesIO) else io.BytesIO(img_input)
            return Image(buf, width=target_width, height=target_height)
        return None

    img_wf = process_image_input(waveform_image, target_width=520, target_height=110)
    img_spec = process_image_input(spectrogram_image, target_width=520, target_height=130)

    if img_wf:
        story.append(Paragraph("<b>Time-Domain Waveform</b>", body_style))
        story.append(Spacer(1, 2))
        story.append(img_wf)
        story.append(Spacer(1, 6))

    if img_spec:
        story.append(Paragraph("<b>2D Log-Mel Spectrogram</b>", body_style))
        story.append(Spacer(1, 2))
        story.append(img_spec)
        story.append(Spacer(1, 10))

    # 6. Optional Audio Quality & Deepfake Risk Analysis
    if quality_result:
        story.append(Paragraph("4. Pre-Prediction Audio Quality & Deepfake Risk Analysis", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

        q_kv1 = {
            "Audio Quality Score": f"{quality_result.get('quality_score', 0.0):.1f} / 100 ({quality_result.get('quality_rating', 'N/A')})",
            "Signal-to-Noise Ratio (SNR)": f"{quality_result.get('snr_db', 0.0):.1f} dB ({quality_result.get('noise_level', 'N/A')} Noise)",
            "Clipping Percentage": f"{quality_result.get('clipping_pct', 0.0):.2f}%",
            "Silence Ratio": f"{quality_result.get('silence_pct', 0.0):.1f}%"
        }
        q_kv2 = {
            "RMS Energy": f"{quality_result.get('rms_energy', 0.0):.4f}",
            "Zero Crossing Rate (ZCR)": f"{quality_result.get('zcr', 0.0):.4f}",
            "Dynamic Range": f"{quality_result.get('dynamic_range_db', 0.0):.1f} dB",
            "Prediction Reliability": f"<b>{quality_result.get('prediction_reliability', 'N/A')}</b>"
        }

        t_q1 = make_kv_table(q_kv1, width1=120, width2=135)
        t_q2 = make_kv_table(q_kv2, width1=120, width2=145)

        q_grid = Table([[t_q1, t_q2]], colWidths=[260, 272])
        q_grid.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(q_grid)
        story.append(Spacer(1, 10))

    # 7. Optional Prediction Stability Analysis
    if stability_result:
        story.append(Paragraph("5. Single-File Prediction Stability Analysis", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

        stab_kv1 = {
            "Original Prediction": stability_result.get("original_prediction", "N/A"),
            "Stable Variant Count": f"{stability_result.get('matching_predictions', 0)} / {stability_result.get('total_variants', 5)} Variants"
        }
        stab_kv2 = {
            "Prediction Flips": f"{stability_result.get('prediction_flips', 0)} Flips",
            "Stability Percentage": f"<b>{stability_result.get('stability_percentage', 0.0):.1f}%</b>"
        }

        t_s1 = make_kv_table(stab_kv1, width1=120, width2=135)
        t_s2 = make_kv_table(stab_kv2, width1=120, width2=145)

        s_grid = Table([[t_s1, t_s2]], colWidths=[260, 272])
        s_grid.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(s_grid)
        story.append(Spacer(1, 10))

    # 8. Optional Explainable AI (Grad-CAM) Model Attention Section
    if explanation_image:
        story.append(Paragraph("6. Explainable AI (XAI) - Grad-CAM Model Attention Map", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

        img_xai = process_image_input(explanation_image, target_width=520, target_height=150)
        if img_xai:
            story.append(img_xai)
            story.append(Spacer(1, 6))

        if explanation_metrics:
            xai_text = f"<b>Peak Attention Window:</b> {explanation_metrics.get('time_window_str', 'N/A')} &nbsp;|&nbsp; <b>Dominant Frequency Band:</b> {explanation_metrics.get('dominant_freq_band', 'N/A')}<br/>"
            xai_text += f"<font color='#4B5563'><i>{explanation_metrics.get('summary', '')}</i></font>"
            story.append(Paragraph(xai_text, body_style))
            story.append(Spacer(1, 8))

        xai_disc = Paragraph("<b>ATTENTION DISCLAIMER:</b> The Grad-CAM heatmap highlights 2D time-frequency regions that contributed most to feature activations. It reflects internal model attention focus rather than physical proof of forgery.", disclaimer_style)
        xai_box = Table([[xai_disc]], colWidths=[532])
        xai_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#93C5FD")),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(xai_box)
        story.append(Spacer(1, 10))

    # 8.5 Optional Model Calibration & Reliability Section
    if calibration_result:
        story.append(Paragraph("6. Confidence Calibration & Reliability Assessment", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

        cal_kv1 = {
            "Raw Softmax Confidence": f"{calibration_result.get('raw_confidence', 0.0):.2f}%",
            "Calibrated Confidence": f"<b>{calibration_result.get('calibrated_confidence', 0.0):.2f}%</b>",
            "Temperature Scaling (T)": f"{calibration_result.get('temperature', 1.25):.2f}"
        }
        cal_kv2 = {
            "Confidence Adjustment": f"-{calibration_result.get('confidence_adjustment', 0.0):.2f}%",
            "Expected Calibration Error": f"{calibration_result.get('ece', 0.035):.4f}",
            "Reliability Assessment": f"<b>{calibration_result.get('reliability_level', 'High Confidence')}</b>"
        }

        t_cal1 = make_kv_table(cal_kv1, width1=120, width2=135)
        t_cal2 = make_kv_table(cal_kv2, width1=120, width2=145)

        cal_grid = Table([[t_cal1, t_cal2]], colWidths=[260, 272])
        cal_grid.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(cal_grid)
        story.append(Spacer(1, 10))

    # 8.6 Optional Model Ensemble Section
    if ensemble_result:
        story.append(Paragraph("7. Model Ensemble Probability Fusion Breakdown", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

        ens_kv1 = {
            "Ensemble Prediction": ensemble_result.get("ensemble_prediction", "N/A"),
            "Fused P(AI-GENERATED)": f"{ensemble_result.get('prob_fake', 0.0)*100:.2f}%",
            "Fused P(REAL)": f"{ensemble_result.get('prob_real', 0.0)*100:.2f}%"
        }
        ens_kv2 = {
            "Model Agreement Rate": f"<b>{ensemble_result.get('agreement_pct', 100.0):.0f}%</b>",
            "Probability Std Dev": f"{ensemble_result.get('prob_std', 0.0):.4f}",
            "Ensemble Architecture": "2D CNN + Advanced CRNN Fusion"
        }

        t_ens1 = make_kv_table(ens_kv1, width1=120, width2=135)
        t_ens2 = make_kv_table(ens_kv2, width1=120, width2=145)

        ens_grid = Table([[t_ens1, t_ens2]], colWidths=[260, 272])
        ens_grid.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(ens_grid)
        story.append(Spacer(1, 10))

    # 9. Optional Degradation Analysis Section
    if degradation_result:
        story.append(Paragraph("8. Real-World Channel Degradation Analysis", h2_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

        deg_cond = degradation_result.get("condition", "Degraded")
        orig_p = degradation_result.get("orig_pred", "N/A")
        deg_p = degradation_result.get("deg_pred", "N/A")
        orig_c = degradation_result.get("orig_conf", 0.0)
        deg_c = degradation_result.get("deg_conf", 0.0)
        flipped = degradation_result.get("flipped", False)
        c_shift = degradation_result.get("conf_change", 0.0)

        deg_kv1 = {
            "Degradation Condition": deg_cond,
            "Original Prediction": f"{orig_p} ({orig_c:.2f}%)",
            "Degraded Prediction": f"{deg_p} ({deg_c:.2f}%)"
        }
        deg_kv2 = {
            "Prediction Flipped": "YES (Prediction Changed)" if flipped else "NO (Stable Prediction)",
            "Confidence Shift (ΔC)": f"{c_shift:+.2f}%",
            "Channel Channel Impact": "High Frequency Filtered / Phase Shifted"
        }

        t_deg1 = make_kv_table(deg_kv1, width1=110, width2=145)
        t_deg2 = make_kv_table(deg_kv2, width1=120, width2=145)

        deg_grid = Table(
            [[t_deg1, t_deg2]],
            colWidths=[260, 272]
        )
        deg_grid.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(deg_grid)
        story.append(Spacer(1, 10))

    # 10. Technical Interpretation & Research Disclaimer
    story.append(Paragraph("8. Technical Interpretation & Guidance", h2_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=c_secondary, spaceBefore=2, spaceAfter=6))

    interp_text = f"""
    The <b>{model_name}</b> detector evaluated the input audio and assigned a probability of 
    <b>{fake_p*100:.2f}%</b> for AI-generated synthetic speech and <b>{real_p*100:.2f}%</b> for real human speech. 
    Confidence scores reflect statistical model certainty based on extracted 2D Log-Mel acoustic patterns.
    """
    story.append(Paragraph(interp_text, body_style))
    story.append(Spacer(1, 8))

    disclaimer_box = Table(
        [[Paragraph("<b>RESEARCH DISCLAIMER:</b> This report represents the output of an experimental machine-learning detection system. Predictions are probabilistic and should not be treated as definitive proof of authenticity or legal evidence.", disclaimer_style)]],
        colWidths=[532]
    )
    disclaimer_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FEF2F2")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#FCA5A5")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(disclaimer_box)

    # Build PDF
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
