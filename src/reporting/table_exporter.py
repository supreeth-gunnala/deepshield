"""
DeepShield Publication Table Exporter Module
Exports empirical model comparison and experiment evaluation metrics into LaTeX, Markdown, CSV, and Excel tables.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional

class ResearchTableExporter:
    """
    Exports experiment evaluation dataframes into publication-ready research table formats.
    """

    @staticmethod
    def export_latex(df: pd.DataFrame, caption: str = "Empirical Detection Performance under Clean & Degraded Audio", label: str = "tab:model_comparison") -> str:
        """
        Formats dataframe as a publication-ready LaTeX tabular environment.

        Args:
            df: Experiment dataframe containing model metrics.
            caption: LaTeX table caption.
            label: LaTeX table reference label.

        Returns:
            LaTeX code string.
        """
        cols = [c for c in df.columns if c not in ["timestamp", "status"]]
        df_clean = df[cols].copy()

        latex_lines = []
        latex_lines.append("\\begin{table}[ht]")
        latex_lines.append("\\centering")
        latex_lines.append(f"\\caption{{{caption}}}")
        latex_lines.append(f"\\label{{{label}}}")
        
        col_spec = "l" + "c" * (len(cols) - 1)
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\hline\\hline")

        # Header Row
        headers = [c.replace("_", " ").title() for c in cols]
        latex_lines.append(" & ".join(headers) + " \\\\")
        latex_lines.append("\\hline")

        # Data Rows
        for _, row in df_clean.iterrows():
            formatted_vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    formatted_vals.append(f"{val:.4f}")
                else:
                    formatted_vals.append(str(val))
            latex_lines.append(" & ".join(formatted_vals) + " \\\\")

        latex_lines.append("\\hline\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        return "\n".join(latex_lines)

    @staticmethod
    def export_markdown(df: pd.DataFrame) -> str:
        """Exports dataframe as a GitHub Flavored Markdown table string."""
        cols = [c for c in df.columns if c not in ["timestamp"]]
        lines = []
        lines.append("| " + " | ".join([c.replace("_", " ").title() for c in cols]) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        
        for _, row in df[cols].iterrows():
            row_vals = []
            for c in cols:
                val = row[c]
                if isinstance(val, float):
                    row_vals.append(f"{val:.4f}")
                else:
                    row_vals.append(str(val))
            lines.append("| " + " | ".join(row_vals) + " |")

        return "\n".join(lines)

    @classmethod
    def export_all_formats(
        cls,
        df: pd.DataFrame,
        output_dir: str = "experiments/results/tables",
        base_name: str = "research_metrics_comparison"
    ) -> Dict[str, str]:
        """
        Exports dataframe into LaTeX (.tex), Markdown (.md), CSV (.csv), and Excel (.xlsx).

        Returns:
            Dictionary mapping format key to output file path.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # 1. LaTeX
        tex_path = os.path.join(output_dir, f"{base_name}.tex")
        with open(tex_path, "w") as f:
            f.write(cls.export_latex(df))
        paths["latex"] = tex_path

        # 2. Markdown
        md_path = os.path.join(output_dir, f"{base_name}.md")
        with open(md_path, "w") as f:
            f.write(cls.export_markdown(df))
        paths["markdown"] = md_path

        # 3. CSV
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        df.to_csv(csv_path, index=False)
        paths["csv"] = csv_path

        # 4. Excel
        xlsx_path = os.path.join(output_dir, f"{base_name}.xlsx")
        try:
            df.to_excel(xlsx_path, index=False)
            paths["excel"] = xlsx_path
        except Exception:
            paths["excel"] = csv_path # Fallback to CSV if openpyxl unavailable

        return paths
