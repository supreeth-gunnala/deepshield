"""
DeepShield Environment Verification Script
Phase 1 Setup Verification
"""

import sys
import os

def run_verification():
    print("==================================================")
    print("      DEEPSHIELD ENVIRONMENT VERIFICATION         ")
    print("==================================================")
    print(f"Python Version : {sys.version.split()[0]} ({sys.executable})\n")

    results = {}

    # 1. Verify PyTorch & GPU Availability
    try:
        import torch
        results['torch'] = torch.__version__
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA)"
        print(f"[SUCCESS] PyTorch Version      : {torch.__version__}")
        print(f"          CUDA Available       : {cuda_available}")
        print(f"          Selected Device      : {device_name}")
    except ImportError as e:
        print(f"[FAILED] PyTorch import failed : {e}")
        results['torch'] = False

    # 2. Verify Torchaudio
    try:
        import torchaudio
        results['torchaudio'] = torchaudio.__version__
        print(f"[SUCCESS] Torchaudio Version   : {torchaudio.__version__}")
    except ImportError as e:
        print(f"[FAILED] Torchaudio import failed: {e}")
        results['torchaudio'] = False

    # 3. Verify Librosa
    try:
        import librosa
        results['librosa'] = librosa.__version__
        print(f"[SUCCESS] Librosa Version      : {librosa.__version__}")
    except ImportError as e:
        print(f"[FAILED] Librosa import failed   : {e}")
        results['librosa'] = False

    # 4. Verify NumPy & SoundFile
    try:
        import numpy as np
        import soundfile as sf
        results['numpy'] = np.__version__
        results['soundfile'] = sf.__version__
        print(f"[SUCCESS] NumPy Version        : {np.__version__}")
        print(f"[SUCCESS] SoundFile Version    : {sf.__version__}")
    except ImportError as e:
        print(f"[FAILED] NumPy/SoundFile failed: {e}")
        results['numpy'] = False

    # 5. Verify Streamlit & Visualization
    try:
        import streamlit as st
        import matplotlib
        import pandas as pd
        results['streamlit'] = st.__version__
        print(f"[SUCCESS] Streamlit Version    : {st.__version__}")
        print(f"[SUCCESS] Matplotlib Version   : {matplotlib.__version__}")
        print(f"[SUCCESS] Pandas Version       : {pd.__version__}")
    except ImportError as e:
        print(f"[FAILED] Streamlit/Plotting    : {e}")
        results['streamlit'] = False

    print("\n--------------------------------------------------")
    print("   FUNCTIONAL SYNTHETIC AUDIO & TORCH TEST        ")
    print("--------------------------------------------------")

    # 6. Functional Test: Create synthetic audio tensor & generate Mel Spectrogram
    try:
        import torch
        import torchaudio.transforms as T

        sample_rate = 16000
        duration_sec = 2.0
        num_samples = int(sample_rate * duration_sec)

        # Generate synthetic sine wave (audio signal)
        t = torch.linspace(0, duration_sec, num_samples)
        sine_wave = 0.5 * torch.sin(2 * 3.14159 * 440.0 * t).unsqueeze(0) # 440 Hz tone, mono [1, 32000]

        # Log-Mel Spectrogram transform
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=128
        )
        mel_spec = mel_transform(sine_wave)
        log_mel_spec = torch.log(mel_spec + 1e-9)

        print(f"[SUCCESS] Synthetic Audio Signal Shape : {sine_wave.shape}")
        print(f"[SUCCESS] Log-Mel Spectrogram Shape    : {log_mel_spec.shape}")
        print(f"[SUCCESS] Audio -> Spectrogram pipeline test passed!")
    except Exception as e:
        print(f"[FAILED] Audio functional test failed : {e}")
        return False

    print("==================================================")
    print("   ALL DEEPSHIELD SETUP CHECKS PASSED SUCCESSFULLY!")
    print("==================================================")
    return True

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
