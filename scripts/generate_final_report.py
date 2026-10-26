"""
DeepShield Final Report Generator Script
Consolidates evaluation results, formats final summary tables, and exports executive research reports.
"""

import os
import sys
import shutil
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

def generate_final_report():
    print("==================================================")
    print("     DEEPSHIELD FINAL RESEARCH REPORT GENERATOR   ")
    print("==================================================")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Copy Figures to final figures directory
    fig_sources = [
        os.path.join(RESULTS_DIR, "robustness", "clean_vs_degraded_f1.png"),
        os.path.join(RESULTS_DIR, "robustness", "robustness_drop.png"),
        os.path.join(RESULTS_DIR, "model_comparison", "robustness_comparison.png"),
        os.path.join(RESULTS_DIR, "model_comparison", "clean_performance_comparison.png")
    ]

    for src in fig_sources:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(FIGURES_DIR, os.path.basename(src)))

    # 2. Compile Summary Reports
    rob_csv = os.path.join(RESULTS_DIR, "robustness", "robustness_summary.csv")
    comp_csv = os.path.join(RESULTS_DIR, "model_comparison", "model_comparison.csv")

    df_rob = pd.read_csv(rob_csv) if os.path.exists(rob_csv) else pd.DataFrame()
    df_comp = pd.read_csv(comp_csv) if os.path.exists(comp_csv) else pd.DataFrame()

    report_path = os.path.join(RESULTS_DIR, "final_report.txt")

    with open(report_path, "w") as f:
        f.write("========================================================================\n")
        f.write("      DEEPSHIELD: ROBUST DETECTION OF AI-GENERATED SPEECH               \n")
        f.write("            FINAL RESEARCH & EVALUATION REPORT                           \n")
        f.write("========================================================================\n\n")

        f.write("1. EXECUTIVE SUMMARY & OBJECTIVE\n")
        f.write("------------------------------------------------------------------------\n")
        f.write("DeepShield evaluates deepfake speech detection robustness under unseen AI\n")
        f.write("generators and real-world audio degradations (noise, compression, resampling,\n")
        f.write("and telephone bandpass filtering).\n\n")

        f.write("2. EXPERIMENTAL PROTOCOL\n")
        f.write("------------------------------------------------------------------------\n")
        f.write("- Audio Standardization : 16,000 Hz Mono, Peak Amplitude Normalized\n")
        f.write("- Feature Extractor    : 128-bin Log-Mel Spectrogram (dB scaled, Z-score normalized)\n")
        f.write("- Evaluated Models     : 2D CNN Baseline (356k params) vs Advanced CRNN (2.9M params)\n")
        f.write("- Protocol             : Generator-Disjoint Splitting (Unseen Generator C in Test Only)\n\n")

        f.write("3. ROBUSTNESS PERFORMANCE BENCHMARK\n")
        f.write("------------------------------------------------------------------------\n")
        if not df_rob.empty:
            f.write(df_rob[["condition", "accuracy", "precision", "recall", "f1_score", "robustness_drop", "percentage_drop"]].to_string(index=False))
        else:
            f.write("No robustness data found.\n")
        f.write("\n\n")

        f.write("4. RESEARCH LIMITATIONS & NOTES\n")
        f.write("------------------------------------------------------------------------\n")
        f.write("- Pipeline Validation: Current runs utilize synthetic mock benchmark files.\n")
        f.write("- Real Data Integration: Place real datasets in data/raw/ and re-index.\n")
        f.write("========================================================================\n")

    print(f"[SUCCESS] Final research report generated at: '{report_path}'")
    print(f"[SUCCESS] Figures copied to: '{FIGURES_DIR}'")
    return True

if __name__ == "__main__":
    generate_final_report()
