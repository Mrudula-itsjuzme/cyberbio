"""Generate multi-seed aggregate plots from real pilot data."""
import os, csv, json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp"

FINAL = Path("results/final")
ARENA = Path("results/arena")
SEEDS = ["20260920", "20260921", "20260922"]
SEED_COLORS = {"20260920": "#4C72B0", "20260921": "#DD8452", "20260922": "#55A868"}


def load_seed_rounds(seed: str) -> pd.DataFrame:
    p = ARENA / f"arena_transformer_seed{seed}" / "final" / "arena_round_summary.csv"
    df = pd.read_csv(p)
    df["seed"] = seed
    return df


def load_seed_ca(seed: str) -> pd.DataFrame:
    p = ARENA / f"arena_transformer_seed{seed}" / "final" / "arena_cross_attack_matrix.csv"
    df = pd.read_csv(p)
    df["seed"] = seed
    return df


def plot_with_error(ax, df, x_col, y_mean, y_std, title, ylabel="", xlabel="Round"):
    if df.empty:
        ax.set_title(title)
        return
    ax.errorbar(df[x_col], df[y_mean], yerr=df[y_std], marker="o", capsize=5, linewidth=2)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FINAL / name, dpi=120)
    plt.close(fig)


def main():
    agg_rounds = pd.read_csv(FINAL / "arena_multiseed_rounds.csv")
    agg_transfer = pd.read_csv(FINAL / "arena_multiseed_transfer.csv")
    agg_gain = pd.read_csv(FINAL / "arena_multiseed_selection_gain.csv")
    agg_thompson = pd.read_csv(FINAL / "arena_multiseed_thompson.csv")
    redists = pd.read_csv(FINAL / "arena_robustness_redistribution.csv")
    tradeoff = pd.read_csv(FINAL / "arena_clean_robust_tradeoff.csv")
    all_rounds = pd.concat([load_seed_rounds(s) for s in SEEDS], ignore_index=True)
    all_ca = pd.concat([load_seed_ca(s) for s in SEEDS], ignore_index=True)

    # 1. multiseed_attacker_drift_vs_round.png
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed in SEEDS:
        sr = all_rounds[all_rounds["seed"] == seed]
        ax.plot(sr["round"], sr["mean_drift"], alpha=0.55, color=SEED_COLORS[seed],
                linestyle="--", linewidth=1, label=f"seed {seed}")
    ax.errorbar(agg_rounds["round"], agg_rounds["mean_drift_mean"],
                yerr=agg_rounds["mean_drift_std"], marker="o", linewidth=2,
                color="black", capsize=5, label="mean ± std")
    ax.set_title("Attacker Mean Drift vs Round (3 seeds)", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Robustness Drift")
    ax.legend(fontsize=7)
    save(fig, "multiseed_attacker_drift_vs_round.png")

    # 2. multiseed_clean_rmse_vs_round.png
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed in SEEDS:
        sr = all_rounds[all_rounds["seed"] == seed]
        ax.plot(sr["round"], sr["defender_clean_rmse"], alpha=0.55, color=SEED_COLORS[seed],
                linestyle="--", linewidth=1, label=f"seed {seed}")
    ax.errorbar(agg_rounds["round"], agg_rounds["defender_clean_rmse_mean"],
                yerr=agg_rounds["defender_clean_rmse_std"], marker="s", linewidth=2,
                color="black", capsize=5, label="mean ± std")
    ax.set_title("Defender Clean RMSE vs Round (3 seeds)", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Clean RMSE")
    ax.legend(fontsize=7)
    save(fig, "multiseed_clean_rmse_vs_round.png")

    # 3. multiseed_defender_gain.png
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed in SEEDS:
        sr = all_rounds[all_rounds["seed"] == seed]
        ax.plot(sr["round"], sr["defender_gain"], alpha=0.55, color=SEED_COLORS[seed],
                linestyle="--", linewidth=1, label=f"seed {seed}")
    ax.errorbar(agg_rounds["round"], agg_rounds["defender_gain_mean"],
                yerr=agg_rounds["defender_gain_std"], marker="^", linewidth=2,
                color="black", capsize=5, label="mean ± std")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="no gain baseline")
    ax.set_title("Defender Gain vs Round (3 seeds)", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Defender Gain")
    ax.legend(fontsize=7)
    save(fig, "multiseed_defender_gain.png")

    # 4. multiseed_evolutionary_transfer.png
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(agg_transfer["round"], agg_transfer["transfer_gain_mean"],
                yerr=agg_transfer["transfer_gain_std"], marker="D", linewidth=2,
                color="green", capsize=5, label="mean ± std")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Held-out Evolutionary Transfer Gain (3 seeds)", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("T_Evo(t) = R(Evo,D_t) - R(Evo,D_{t+1})")
    ax.legend(fontsize=7)
    save(fig, "multiseed_evolutionary_transfer.png")

    # 5. multiseed_adaptive_selection_gain.png
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(agg_gain["round"], agg_gain["selection_gain_mean"],
                yerr=agg_gain["selection_gain_std"], marker="o", linewidth=2,
                color="purple", capsize=5, label="mean ± std")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_title("Adaptive Selection Gain G_select(t) (3 seeds)", fontweight="bold")
    ax.set_xlabel("Round t")
    ax.set_ylabel("G_select = R(A_{t+1}, D_t) - R(A_t, D_t)")
    ax.legend(fontsize=7)
    save(fig, "multiseed_adaptive_selection_gain.png")

    # 6. multiseed_thompson_probabilities.png
    fig, ax = plt.subplots(figsize=(6, 4))
    for att in agg_thompson["attacker"].unique():
        th_a = agg_thompson[agg_thompson["attacker"] == att]
        ax.errorbar(th_a["round"], th_a["posterior_mean_mean"],
                    yerr=th_a["posterior_mean_std"], marker="o", capsize=4,
                    linewidth=2, label=att)
    ax.set_title("Thompson Posterior Mean (3 seeds)", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("E[θ] (posterior mean reward)")
    ax.legend(fontsize=7)
    save(fig, "multiseed_thompson_probabilities.png")

    # 7. multiseed_cross_attack_heatmap.png – mean drift across seeds per (round, attacker)
    pivot = all_ca.groupby(["round", "attacker"])["mean_robustness_effect"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values.T, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels([f"D_{r+1}" for r in pivot.index], fontsize=8)
    ax.set_yticks(range(len(pivot.columns)))
    yticklabels = [f"{a} (HELD OUT)" if a == "Evolutionary" else a for a in pivot.columns]
    ax.set_yticklabels(yticklabels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean Drift")
    ax.set_title("Cross-Attack Mean Drift (mean across seeds)", fontweight="bold")
    save(fig, "multiseed_cross_attack_heatmap.png")

    # 8. robustness_redistribution_events.png
    fig, ax = plt.subplots(figsize=(6, 4))
    redists_true = redists[redists["redistribution_observed"].astype(str) == "True"]
    if not redists_true.empty:
        counts = redists_true.groupby(["round", "active_attacker"]).size().reset_index(name="count")
        rounds = counts["round"].unique()
        x = np.arange(len(rounds))
        for i, att in enumerate(counts["active_attacker"].unique()):
            sub = counts[counts["active_attacker"] == att]
            sub = sub.set_index("round").reindex(rounds, fill_value=0)
            ax.bar(x + i * 0.35, sub["count"], width=0.35, label=att)
        ax.set_xticks(x + 0.175)
        ax.set_xticklabels([f"Round {r}" for r in rounds])
        ax.legend(fontsize=8)
    ax.set_title("Robustness Redistribution Events\n(active attacker improves, other worsens)", fontweight="bold")
    ax.set_ylabel("Count across seeds")
    save(fig, "robustness_redistribution_events.png")

    # 9. clean_vs_robust_tradeoff_multiseed.png
    fig, ax = plt.subplots(figsize=(6, 4))
    colors_cls = {"ROBUST_WIN_CLEAN_WIN": "green", "ROBUST_WIN_CLEAN_COST": "orange",
                  "ROBUST_LOSS_CLEAN_WIN": "blue", "ROBUST_LOSS_CLEAN_COST": "red"}
    for cls, grp in tradeoff.groupby("classification"):
        ax.scatter(grp["delta_robust"], grp["delta_clean"],
                   color=colors_cls.get(cls, "gray"), label=cls, s=60, alpha=0.8)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("ΔRobust (positive = less vulnerable)")
    ax.set_ylabel("ΔClean RMSE (positive = worse)")
    ax.set_title("Clean vs Robust Tradeoff (3 seeds)", fontweight="bold")
    ax.legend(fontsize=6)
    save(fig, "clean_vs_robust_tradeoff_multiseed.png")

    # 10. per_seed_trajectory_comparison.png
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        sr = all_rounds[all_rounds["seed"] == seed]
        ax.plot(sr["round"], sr["mean_drift"], marker="o", label="Mean Drift", linewidth=2)
        ax.plot(sr["round"], sr["defender_clean_rmse"], marker="s", linestyle="--",
                label="Clean RMSE", linewidth=2)
        ax.set_title(f"Seed {seed}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Round", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("Value")
    fig.suptitle("Per-Seed Trajectory Comparison", fontsize=10, fontweight="bold")
    save(fig, "per_seed_trajectory_comparison.png")

    print("All 10 multiseed plots saved.")


if __name__ == "__main__":
    main()
