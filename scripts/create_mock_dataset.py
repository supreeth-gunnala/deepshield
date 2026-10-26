"""
DeepShield Mock Dataset Generator Script
Generates a structured synthetic audio dataset to verify metadata indexing and generator-disjoint splits.
"""

import os
import sys
import numpy as np
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_mock_dataset(base_dir: str = None, num_samples_per_category: int = 10):
    if base_dir is None:
        base_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    categories = {
        "real": ["human"],
        "fake": ["generator_A", "generator_B", "generator_C"]
    }

    total_created = 0

    for cat, generators in categories.items():
        for gen in generators:
            if cat == "real":
                dir_path = os.path.join(base_dir, "real")
            else:
                dir_path = os.path.join(base_dir, "fake", gen)

            os.makedirs(dir_path, exist_ok=True)

            for i in range(1, num_samples_per_category + 1):
                filename = f"{gen}_sample_{i:03d}.wav"
                file_path = os.path.join(dir_path, filename)

                # Generate synthetic signal with distinct pitch signature per generator
                freq = 440 + (hash(gen) % 200)
                audio_signal = 0.5 * np.sin(2 * np.pi * freq * t)

                sf.write(file_path, audio_signal, sr)
                total_created += 1

    print(f"[MOCK DATASET] Generated {total_created} mock audio files under: '{base_dir}'")
    return total_created

if __name__ == "__main__":
    generate_mock_dataset()
