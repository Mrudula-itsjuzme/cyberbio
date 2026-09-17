# Master Experiment Index and Artifact Mapping

This index maps every reported experimental result to its underlying script, configuration file, input dataset, output log, and publication figure:

| Experiment Name | Primary Script | Config File | Input Dataset | Output Results Artifact | Publication Figure |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multi-Seed Robustness Validation (5 Seeds)** | `scripts/run_comprehensive_benchmark_suite.py` | `configs/closed_loop.yaml` | `data/processed/processed.csv` | `results/comprehensive_benchmark_summary.json` | `outputs/baseline_vs_defended_multiseed.png` |
| **Before/After Defense Pilot** | `scripts/run_before_after_defense_experiment.py` | `configs/model.yaml` | `data/processed/processed.csv` | `results/before_after_defense_experiment_results.json` | — |
| **Attack Paradigm Comparison** | `scripts/run_comprehensive_benchmark_suite.py` | `configs/attack.yaml` | `data/processed/processed.csv` | `results/comprehensive_benchmark_summary.json` | — |
| **Loss Weight Sweep ($\lambda$)** | `scripts/run_comprehensive_benchmark_suite.py` | `configs/closed_loop.yaml` | `data/processed/processed.csv` | `results/comprehensive_benchmark_summary.json` | `outputs/ablation_study_chart.png` |
| **Tanimoto Cutoff Sweep ($S_{\text{min}}$)** | `scripts/run_comprehensive_benchmark_suite.py` | `configs/attack.yaml` | `data/processed/processed.csv` | `results/comprehensive_benchmark_summary.json` | `outputs/ablation_study_chart.png` |
| **MCMC Step Budget Sensitivity** | `scripts/run_comprehensive_benchmark_suite.py` | `configs/attack.yaml` | `data/processed/processed.csv` | `results/comprehensive_benchmark_summary.json` | `outputs/mcmc_steps_drift_curve.png` |
| **Epistemic Uncertainty Quantification** | `scripts/run_comprehensive_benchmark_suite.py` | `configs/specialized_transformer.yaml` | `data/processed/processed.csv` | `results/comprehensive_benchmark_summary.json` | — |
| **Reproducibility & Environment Audit** | `scripts/reproducibility_audit.py` | `RELEASE_MANIFEST.json` | — | Audit terminal log output | — |
| **Publication Figure Generation** | `scripts/generate_publication_plots.py` | — | `results/comprehensive_benchmark_summary.json` | Saved PNG files in `outputs/` | Figures 1, 2, 3 |
