"""
DeepShield Real Dataset Inspection Utility
Analyzes C:\\Users\\user\\Downloads\\_sources\\_sources folder structure, audio metrics, SHA-256 hashes, and corruptions.
"""

import os
import glob
import hashlib
import soundfile as sf
import numpy as np
import pandas as pd
import json

DATASET_ROOT = r"C:\Users\user\Downloads\Ai Project\data\raw\asvspoof_voice"
FOLDERS = ["real_voice", "fake_voice", "real_bg", "fake_bg"]

def inspect_dataset():
    print(f"=== INSPECTING REAL DATASET AT: {DATASET_ROOT} ===")
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(f"Dataset root '{DATASET_ROOT}' does not exist.")

    records = []
    hash_map = {}
    corrupted_files = []
    
    for folder in FOLDERS:
        folder_path = os.path.join(DATASET_ROOT, folder)
        if not os.path.exists(folder_path):
            print(f"WARNING: Subdirectory '{folder_path}' missing!")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
        print(f"Folder '{folder}': {len(files)} .wav files found.")

        for f in files:
            file_path = os.path.join(folder_path, f)
            rel_path = os.path.join(folder, f)
            
            # SHA-256 hash
            try:
                with open(file_path, "rb") as fp:
                    content = fp.read()
                    sha256 = hashlib.sha256(content).hexdigest()
                
                file_size = os.path.getsize(file_path)

                # Soundfile inspection
                info = sf.info(file_path)
                data, sr = sf.read(file_path)
                duration = info.duration
                channels = info.channels
                sample_rate = info.samplerate
                format_name = info.format

                peak_amp = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
                is_clipped = peak_amp >= 0.999
                rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0

                is_dup = sha256 in hash_map
                if is_dup:
                    hash_map[sha256].append(rel_path)
                else:
                    hash_map[sha256] = [rel_path]

                records.append({
                    "file_path": file_path,
                    "rel_path": rel_path,
                    "filename": f,
                    "folder": folder,
                    "file_size_bytes": file_size,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "duration": duration,
                    "format": format_name,
                    "peak_amplitude": peak_amp,
                    "is_clipped": is_clipped,
                    "rms_energy": rms,
                    "sha256": sha256,
                    "is_duplicate": is_dup,
                    "valid": True
                })
            except Exception as e:
                print(f"ERROR reading '{file_path}': {e}")
                corrupted_files.append({"file_path": file_path, "error": str(e)})

    df = pd.DataFrame(records)
    print(f"\n--- DATASET SUMMARY ---")
    print(f"Total Valid Files Processed: {len(df)}")
    print(f"Total Corrupted Files: {len(corrupted_files)}")
    print("\nFile counts per folder:")
    print(df["folder"].value_counts())

    print("\nSample rate distribution:")
    print(df["sample_rate"].value_counts())

    print("\nChannel count distribution:")
    print(df["channels"].value_counts())

    print("\nDuration stats (seconds):")
    print(df["duration"].describe())

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    print(f"\nUnique SHA-256 Hashes: {len(hash_map)}")
    print(f"Duplicate Hashes Found: {len(duplicates)}")

    if duplicates:
        print("Sample Duplicate Pairs:")
        for h, paths in list(duplicates.items())[:5]:
            print(f"  Hash {h[:12]}... -> {paths}")

    # Output detailed JSON report for dataset audit
    summary_report = {
        "dataset_root": DATASET_ROOT,
        "total_files": len(df),
        "corrupted_files_count": len(corrupted_files),
        "corrupted_files_list": corrupted_files,
        "folder_counts": df["folder"].value_counts().to_dict(),
        "sample_rates": df["sample_rate"].value_counts().to_dict(),
        "channels": df["channels"].value_counts().to_dict(),
        "duration_min": float(df["duration"].min()),
        "duration_max": float(df["duration"].max()),
        "duration_mean": float(df["duration"].mean()),
        "duplicate_hash_groups_count": len(duplicates),
        "duplicate_pairs_sample": list(duplicates.values())[:10]
    }

    with open("reports/real_dataset_inspection.json", "w") as f:
        json.dump(summary_report, f, indent=2)

    df.to_csv("reports/real_dataset_inspection_full.csv", index=False)
    print("\nInspection complete! Reports saved to reports/real_dataset_inspection.json & reports/real_dataset_inspection_full.csv")

if __name__ == "__main__":
    inspect_dataset()
