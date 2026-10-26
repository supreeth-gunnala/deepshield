"""
DeepShield Test Script: Research Table Exporter Test
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.reporting.table_exporter import ResearchTableExporter

def main():
    print("=" * 50)
    print("    DEEPSHIELD RESEARCH TABLE EXPORTER TEST      ")
    print("=" * 50)

    # Sample dataframe
    df = pd.DataFrame([
        {
            "model_architecture": "2D CNN Baseline",
            "clean_f1": 0.9125,
            "noise_f1": 0.7840,
            "compression_f1": 0.8120,
            "unseen_generator_f1": 0.6940,
            "robustness_drop": 0.1695
        },
        {
            "model_architecture": "Advanced CRNN",
            "clean_f1": 0.9420,
            "noise_f1": 0.8350,
            "compression_f1": 0.8610,
            "unseen_generator_f1": 0.7480,
            "robustness_drop": 0.1440
        }
    ])

    print("\n[STEP 1] Generating LaTeX table code...")
    latex_code = ResearchTableExporter.export_latex(df)
    print(latex_code[:300] + "\n...")

    print("\n[STEP 2] Exporting to LaTeX, Markdown, CSV, and Excel...")
    paths = ResearchTableExporter.export_all_formats(df, output_dir=os.path.join(PROJECT_ROOT, "experiments", "results", "tables"))

    for k, p in paths.items():
        print(f" -> Exported {k.upper():<8}: '{p}' (Size: {os.path.getsize(p):,} bytes)")

    assert os.path.exists(paths["latex"]), "LaTeX export file missing!"
    assert os.path.exists(paths["markdown"]), "Markdown export file missing!"

    print("\n" + "=" * 50)
    print("  RESEARCH TABLE EXPORTER TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
