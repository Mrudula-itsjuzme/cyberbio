import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
os.environ["MPLCONFIGDIR"] = "/tmp"

final = Path("results/final")

def make_plots():
    rounds = pd.read_csv(final/"arena_pilot_rounds.csv")
    ca = pd.read_csv(final/"arena_pilot_cross_attack.csv")
    att = pd.read_csv(final/"arena_pilot_attackers.csv")
    defenders = pd.read_csv(final/"arena_pilot_defenders.csv")
    thompson = pd.read_csv(final/"arena_pilot_thompson.csv")
    
    plt.figure()
    plt.plot(rounds["round"], rounds["mean_drift"], marker='o')
    plt.title("Attacker Drift vs Round")
    plt.savefig(final/"attacker_drift_vs_round.png")
    
    plt.figure()
    plt.plot(rounds["round"], rounds["defender_robustness"], marker='o')
    plt.title("Defender Robustness vs Round")
    plt.savefig(final/"defender_robustness_vs_round.png")
    
    plt.figure()
    plt.plot(rounds["round"], rounds["defender_clean_rmse"], marker='o')
    plt.title("Clean RMSE vs Round")
    plt.savefig(final/"clean_rmse_vs_round.png")
    
    plt.figure()
    ca_pivot = ca.pivot(index="defender_checkpoint", columns="attacker", values="mean_robustness_effect")
    plt.imshow(ca_pivot, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Cross Attack Transfer Heatmap")
    plt.savefig(final/"cross_attack_transfer_heatmap.png")
    
    plt.figure()
    plt.plot(rounds["round"], rounds["defender_gain"], marker='o')
    plt.title("Defender Gain by Round")
    plt.savefig(final/"defender_gain_by_round.png")
    
    
    adapt = pd.read_csv(final/"arena_pilot_adaptation.csv")
    plt.figure()
    plt.plot(adapt["round"], adapt["adaptation_gain"], marker='o')
    plt.title("Attacker Adaptation Gain")

    plt.savefig(final/"attacker_adaptation_gain.png")
    
    plt.figure()
    for attacker in thompson["attacker"].unique():
        t = thompson[thompson["attacker"] == attacker]
        plt.plot(t["round"], t["posterior_mean"], marker='o', label=attacker)
    plt.legend()
    plt.title("Thompson Arm Probabilities")
    plt.savefig(final/"thompson_arm_probabilities.png")
    
    plt.figure()
    plt.plot(rounds["round"], rounds["candidate_diversity"], marker='o')
    plt.title("Hard Example Diversity")
    plt.savefig(final/"hard_example_diversity.png")
    
    plt.figure()
    plt.plot(rounds["round"], rounds["validity_rate"], marker='o')
    plt.title("Validity vs Round")
    plt.savefig(final/"validity_vs_round.png")
    
    plt.figure()
    plt.scatter(rounds["defender_robustness"], rounds["defender_clean_rmse"])
    plt.title("Clean vs Robust Tradeoff")
    plt.xlabel("Robustness")
    plt.ylabel("Clean RMSE")
    plt.savefig(final/"clean_vs_robust_tradeoff.png")

make_plots()
