"""Publication-Quality Plot Generator for Materials Adversarial Framework.

Generates 3 publication-ready matplotlib figures saved to outputs/:
1. outputs/baseline_vs_defended_multiseed.png: Bar chart with error bars showing Baseline vs Defended performance across 5 seeds.
2. outputs/mcmc_steps_drift_curve.png: Sensitivity curve showing Prediction Drift vs MCMC Attack Steps.
3. outputs/ablation_study_chart.png: Multi-panel chart illustrating loss weight lambda and Tanimoto threshold ablations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Use clean publication style settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
})


def generate_plots(results_path: str = "results/comprehensive_benchmark_summary.json", output_dir: str = "outputs"):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file = Path(results_path)

    if not res_file.exists():
        logger.error(f"Results file {res_file} not found! Run scripts/run_comprehensive_benchmark_suite.py first.")
        return

    data = json.loads(res_file.read_text())
    b_stats = data["baseline_model_stats"]
    d_stats = data["defended_model_stats"]

    # --------------------------------------------------------------------------
    # Figure 1: Baseline vs Defended Multi-Seed Performance
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_keys = ["clean_rmse", "adv_rmse", "mean_absolute_drift"]
    metric_labels = ["Clean RMSE\n(eV)", "Adv RMSE\n(eV)", "Mean Drift\n(eV)"]

    b_means = [b_stats[f"{k}_mean"] for k in metrics_keys]
    b_stds = [b_stats[f"{k}_std"] for k in metrics_keys]

    d_means = [d_stats[f"{k}_mean"] for k in metrics_keys]
    d_stds = [d_stats[f"{k}_std"] for k in metrics_keys]

    x = np.arange(len(metric_labels))
    width = 0.35

    ax.bar(x - width/2, b_means, width, yerr=b_stds, label='Baseline Model', color='#d95f02', capsize=5, alpha=0.85)
    ax.bar(x + width/2, d_means, width, yerr=d_stds, label='Defended Model', color='#1b9e77', capsize=5, alpha=0.85)

    ax.set_ylabel('Energy (eV)')
    ax.set_title('Baseline vs. Defended Model Robustness Across 5 Seeds')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig1_path = out_dir / "baseline_vs_defended_multiseed.png"
    plt.savefig(fig1_path)
    plt.close()
    logger.info(f"Saved Figure 1 to {fig1_path}")

    # --------------------------------------------------------------------------
    # Figure 2: MCMC Attack Steps Sensitivity Curve
    # --------------------------------------------------------------------------
    mcmc_steps_data = data.get("mcmc_steps_sensitivity", {})
    if mcmc_steps_data:
        steps_list = [int(k.replace("steps_", "")) for k in mcmc_steps_data.keys()]
        drifts_list = [v["mean_absolute_drift"] for v in mcmc_steps_data.values()]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(steps_list, drifts_list, marker='o', linewidth=2.5, color='#7570b3', markersize=8)
        ax.set_xlabel('MCMC Search Budget (Steps)')
        ax.set_ylabel('Mean Absolute Prediction Drift (eV)')
        ax.set_title('Adversarial Prediction Drift vs. MCMC Attack Budget')
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig2_path = out_dir / "mcmc_steps_drift_curve.png"
        plt.savefig(fig2_path)
        plt.close()
        logger.info(f"Saved Figure 2 to {fig2_path}")

    # --------------------------------------------------------------------------
    # Figure 3: Ablation Study (Lambda Sweeps)
    # --------------------------------------------------------------------------
    lambda_data = data.get("lambda_ablations", {})
    if lambda_data:
        lambdas = [float(k.replace("lambda_", "")) for k in lambda_data.keys()]
        clean_rmses = [v["clean_rmse"] for v in lambda_data.values()]
        adv_drifts = [v["mean_absolute_drift"] for v in lambda_data.values()]

        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        color = '#e7298a'
        ax1.set_xlabel('Adversarial Loss Weight (λ)')
        ax1.set_ylabel('Mean Prediction Drift (eV)', color=color)
        line1 = ax1.plot(lambdas, adv_drifts, marker='s', color=color, linewidth=2.5, label='Mean Drift (Robustness)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2 = ax1.twinx()
        color = '#66a61e'
        ax2.set_ylabel('Clean RMSE (eV)', color=color)
        line2 = ax2.plot(lambdas, clean_rmses, marker='^', color=color, linewidth=2.5, linestyle='--', label='Clean RMSE (Accuracy)')
        ax2.tick_params(axis='y', labelcolor=color)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        plt.title('Tradeoff Between Clean Accuracy and Robustness across Loss Weight λ')

        plt.tight_layout()
        fig3_path = out_dir / "ablation_study_chart.png"
        plt.savefig(fig3_path)
        plt.close()
        logger.info(f"Saved Figure 3 to {fig3_path}")


if __name__ == "__main__":
    generate_plots()
