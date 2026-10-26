"""
DeepShield Test Script: Research Paper Draft Generator Test
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.experiments.registry import ExperimentRegistry
from src.reporting.paper_generator import ResearchPaperDraftGenerator

def main():
    print("=" * 50)
    print("  DEEPSHIELD RESEARCH PAPER DRAFT GENERATOR TEST ")
    print("=" * 50)

    registry = ExperimentRegistry(registry_dir=os.path.join(PROJECT_ROOT, "experiments", "registry"))
    df_history = registry.get_history()

    generator = ResearchPaperDraftGenerator(registry_df=df_history)
    output_path = os.path.join(PROJECT_ROOT, "experiments", "paper_draft.md")

    print(f"\n[STEP 1] Compiling empirical paper draft from registry ({len(df_history)} entries)...")
    draft_text = generator.generate_draft(output_path=output_path)

    print(f" -> Generated Paper Draft Length: {len(draft_text):,} characters")
    print(f" -> Saved Paper Draft to: '{output_path}'")

    assert os.path.exists(output_path), "Paper draft file was not created!"
    assert "Abstract" in draft_text, "Missing Abstract section!"
    assert "Methodology" in draft_text, "Missing Methodology section!"

    print("\n" + "=" * 50)
    print(" PAPER DRAFT GENERATOR TEST PASSED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()
