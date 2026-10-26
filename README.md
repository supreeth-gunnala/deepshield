# DeepShield: Robust Detection of AI-Generated Speech Under Unseen Generators and Real-World Audio Degradation

[![Python 3.11](https.img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DeepShield** is a PyTorch-based machine learning research framework and interactive web dashboard engineered to detect **AI-generated (synthetic / deepfake) speech** and evaluate detector robustness against **unseen AI voice generators** and **real-world audio degradations**.

---

## 📋 Table of Contents
1. [Research Problem & Motivation](#1-research-problem--motivation)
2. [Research Novelty](#2-research-novelty)
3. [System Architecture](#3-system-architecture)
4. [Key Features](#4-key-features)
5. [Installation & Windows Environment Setup](#5-installation--windows-environment-setup)
6. [Dataset Organization & Metadata Format](#6-dataset-organization--metadata-format)
7. [Generator-Disjoint Protocol](#7-generator-disjoint-protocol)
8. [Training & Evaluation Commands](#8-training--evaluation-commands)
9. [Robustness & Degradation Experiments](#9-robustness--degradation-experiments)
10. [Streamlit Research Dashboard](#10-streamlit-research-dashboard)
11. [Research Limitations & Future Improvements](#11-research-limitations--future-improvements)

---

## 1. Research Problem & Motivation
Naive deepfake speech detectors often achieve over 95% accuracy under clean laboratory conditions on familiar synthetic voice clones. However, their performance severely degrades when deployed in real-world environments due to:
1. **Unseen AI Voice Generators:** Detectors overfit to specific vocoder artifacts (e.g. ElevenLabs or Bark) and fail on new, unseen synthesis models (e.g. VALL-E or XTTS).
2. **Real-World Audio Degradations:** Ambient noise, VoIP compression, sample rate conversions, and phone channel bandpass filters alter spectral features and degrade detection reliability.

**Key Research Question:**
> *"Can a speech deepfake detector reliably identify AI-generated speech when the audio is degraded and the AI generator is unseen during training?"*

---

## 2. Research Novelty
- **Quantifiable Robustness Drop ($\Delta \text{F1}$):** Measures the exact drop in F1-score ($\text{F1}_{\text{clean}} - \text{F1}_{\text{degraded}}$) under real-world corruptions.
- **Generator-Disjoint Evaluation Protocol:** Enforces $G_{\text{train}} \cap G_{\text{test}} = \emptyset$ to prevent data leakage and measure true out-of-distribution (OOD) generalization.
- **Prediction Stability Analysis:** Evaluates sample-level prediction flip rates and confidence shifts ($\Delta C$) across identical audio clips subjected to incremental degradations.

---

## 3. System Architecture

```text
Audio File Upload (WAV, MP3, FLAC, OGG)
                 │
                 ▼
      [ Input Validation ] (Corrupt, silence, format checks)
                 │
                 ▼
     [ Audio Preprocessing ] (16,000 Hz Mono, Peak Amplitude Normalization)
                 │
                 ├───► [ Real-World Audio Degradation Engine ]
                 │     (Noise, Resampling, VoIP Bandpass, Pitch Shift, Compression)
                 │
                 ▼
   [ Log-Mel Spectrogram Extraction ] (128 Mels, dB Scale, Z-score Normalization)
                 │
                 ▼
   [ PyTorch Deep Learning Model ] (2D CNN Baseline / Advanced CRNN)
                 │
                 ▼
  [ Binary Classification & Confidence ] (0 = REAL, 1 = AI-GENERATED)
                 │
                 ▼
  [ Streamlit Interactive Research UI ] (Playback, Spectrograms, Stress-Testing)
```

---

## 4. Key Features
- **Modular Pipeline:** Independent modules for validation, preprocessing, feature extraction, dataset indexing, model training, evaluation, and degradation.
- **Multi-Model Support:** Includes both a 3-block 2D CNN baseline ($356\text{k}$ params) and a 4-block Conv2D + BiGRU CRNN ($2.9\text{M}$ params).
- **On-Disk Feature Caching:** Caches pre-extracted spectrogram tensors to `data/processed/features/` to accelerate training epochs.
- **Interactive Research Dashboard:** Streamlit UI for single-file live prediction, audio stress-testing, and benchmark reporting.

---

## 5. Installation & Windows Environment Setup

### Prerequisites
- Python 3.10 or 3.11 (64-bit)
- Windows PowerShell or Command Prompt

### Environment Setup Commands
```powershell
# 1. Clone or navigate to the project directory
cd "c:\Users\user\Downloads\Ai Project"

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install PyTorch (GPU or CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Install dependencies
pip install -r requirements.txt
```

---

## 6. Dataset Organization & Metadata Format

### Recommended Directory Structure
```text
data/raw/
├── real/                      # Real human speech (.wav, .mp3, .flac)
└── fake/
    ├── generator_A/           # Known synthetic generator A
    ├── generator_B/           # Known synthetic generator B
    └── generator_C/           # Unseen synthetic generator C
```

---

## 7. Generator-Disjoint Protocol
```yaml
dataset:
  train_generators: ["generator_A", "generator_B"]
  val_generators: ["generator_A", "generator_B"]
  unseen_test_generators: ["generator_C"]
```

---

## 8. Training & Evaluation Commands

### Phase 4: Build Metadata Index
```powershell
python scripts/build_dataset_metadata.py
```

### Phase 7: Train 2D CNN Baseline
```powershell
python scripts/train_baseline.py
```

### Phase 11: Train Advanced CRNN Model
```powershell
python scripts/train_advanced_model.py
```

### Phase 8: Evaluate Baseline Model
```powershell
python scripts/evaluate_baseline.py
```

---

## 9. Robustness & Degradation Experiments

Run the 11-condition degradation benchmark:
```powershell
python scripts/run_robustness_experiments.py
```

Run unseen generator evaluation:
```powershell
python scripts/run_unseen_generator_evaluation.py
```

Run model comparison & ablation suite:
```powershell
python scripts/test_model_comparison.py
```

---

## 10. Streamlit Research Dashboard

Launch the interactive web dashboard:
```powershell
streamlit run app.py
```

---

## 11. Research Limitations & Future Improvements
- **Pipeline Validation:** Current benchmark runs use synthetic mock audio files for pipeline validation.
- **Real Dataset Integration:** Place real-world datasets (e.g. ASVspoof 2021, LibriSeeded, ElevenLabs) in `data/raw/` and re-index.
- **Future Improvements:** Integrating pre-trained self-supervised speech encoders (Wav2Vec 2.0 / HuBERT) and multi-task loss functions.
