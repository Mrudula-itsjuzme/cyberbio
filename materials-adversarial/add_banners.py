import os

banner = """> [!WARNING]
> **Historical Metrics / Superseded Baseline**
> The metrics and narratives in this document (e.g., 20.63% drift reduction, clean RMSE 1.1439, 83.33% uncertainty drop) belong to an earlier evaluation phase. 
> Please refer to `results/experimental_summary.md` and `results/canonical_benchmark_no_leakage.json` for the final, leakage-free canonical benchmark matrix, which explicitly separates representation-preserving augmentation from chemistry-changing MCMC stress tests.

"""

files_to_banner = [
    "docs/VIVA_GUIDE.md",
    "docs/FINAL_REPORT_DRAFT.md",
    "docs/RESEARCH_PAPER_DRAFT.md",
    "docs/ABLATIONS.md",
    "docs/ABLATION_STUDIES.md"
]

for filepath in files_to_banner:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
        if "Historical Metrics" not in content:
            with open(filepath, "w") as f:
                f.write(banner + content)

