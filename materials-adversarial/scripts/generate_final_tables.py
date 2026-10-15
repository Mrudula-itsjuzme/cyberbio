#!/usr/bin/env python3
"""Generate canonical final evaluation tables for the polyVERSE Bandgap lineage.

Extracts verified metrics from canonical artifacts and outputs cleanly formatted
Markdown, CSV, JSON, and LaTeX tables. Complies strictly with project guidelines:
- Marks missing values clearly instead of replacing with zeros.
- Enforces strict separation of historical Tg (K) lineage vs Bandgap (eV).
- Uses explicit terminology (representation-preserving Rand-SMILES drift,
  chemistry-changing stress-test drift).
"""

import json
import pandas as pd
from pathlib import Path

def generate_markdown_table(df):
    if df.empty:
        return ""
    headers = df.columns.tolist()
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "|---" * len(headers) + "|"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(val) for val in row.values) + " |")
    return "\n".join([header_row, separator_row] + rows)

def main():
    canonical_benchmark = Path("results/canonical_benchmark_no_leakage.json")
    out_dir = Path("results/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not canonical_benchmark.exists():
        print(f"Error: Canonical benchmark not found at {canonical_benchmark}")
        return
        
    with open(canonical_benchmark) as f:
        data = json.load(f)
        
    if "experiments_matrix" not in data:
        print("Error: Invalid benchmark format")
        return
        
    print("Generating tables for polyVERSE Bandgap (eV) lineage...")
    
    records = []
    for row in data["experiments_matrix"]:
        experiment = row.get("experiment", "UNKNOWN")
        split = "Random" if "RandomSplit" in experiment else "Scaffold" if "ScaffoldSplit" in experiment else "Unknown"
        model = experiment.replace("RandomSplit_", "").replace("ScaffoldSplit_", "")
        
        records.append({
            "Split": split,
            "Model Configuration": model,
            "Clean RMSE (eV)": row.get("clean_rmse_eV"),
            "Clean MAE (eV)": row.get("clean_mae_eV"),
            "R^2": row.get("clean_r2"),
            "Rand-SMILES Drift (eV) [Repr. Preserving]": row.get("rand_smiles_drift_mean_eV"),
            "MCMC Drift (eV) [Chem. Changing Stress-Test]": row.get("mcmc_drift_mean_eV")
        })

    df = pd.DataFrame(records)
    df_display = df.copy()
    for col in df_display.columns:
        if df_display[col].dtype == float:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "MISSING")
            
    df.to_json(out_dir / "canonical_tables.json", orient="records", indent=2)
    df.to_csv(out_dir / "canonical_tables.csv", index=False)
    
    md_content = [
        "# Canonical Evaluation Tables: polyVERSE Bandgap Lineage",
        "",
        "Generated from `results/canonical_benchmark_no_leakage.json`.",
        "Target property: Bandgap (eV)",
        "",
        "**NOTE**: Drift under randomized SMILES represents representation sensitivity "
        "(molecular graph is preserved). MCMC drift represents stress-test drift from true chemistry changes.",
        "Chemistry-changing drift is a measure of robustness to distribution shift, but is NOT proven physical model error without oracle validation.",
        "",
        "## Table 1: Random Split",
        ""
    ]
    md_content.append(generate_markdown_table(df_display[df_display["Split"] == "Random"].drop(columns=["Split"])))
    md_content.extend([
        "",
        "## Table 2: Scaffold Split",
        ""
    ])
    md_content.append(generate_markdown_table(df_display[df_display["Split"] == "Scaffold"].drop(columns=["Split"])))
    
    with open(out_dir / "canonical_tables.md", "w") as f:
        f.write("\n".join(md_content) + "\n")

    latex_content = [
        "% Canonical Evaluation Tables: polyVERSE Bandgap Lineage",
        "% Target property: Bandgap (eV)",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Random Split Performance (Bandgap eV)}",
        df_display[df_display["Split"] == "Random"].drop(columns=["Split"]).to_latex(index=False),
        "\\end{table}",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Scaffold Split Performance (Bandgap eV)}",
        df_display[df_display["Split"] == "Scaffold"].drop(columns=["Split"]).to_latex(index=False),
        "\\end{table}"
    ]
    with open(out_dir / "canonical_tables.tex", "w") as f:
        f.write("\n".join(latex_content) + "\n")

    print(f"Successfully generated Markdown, CSV, JSON, and LaTeX tables in {out_dir}")

if __name__ == "__main__":
    main()
