#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    out_dir = Path("results/thompson")
    trace_path = out_dir / "posterior_trace.csv"
    summary_path = out_dir / "thompson_sanity_summary.json"
    regret_path = out_dir / "regret.csv"
    
    if not trace_path.exists() or not summary_path.exists():
        print("Data files not found.")
        return
        
    df = pd.read_csv(trace_path)
    reg_df = pd.read_csv(regret_path)
    with open(summary_path) as f:
        summary = json.load(f)
    sum_df = pd.DataFrame(summary)
    
    units = sum_df["target_units"].iloc[0] if not sum_df.empty else "eV"
    
    # 1. Arm frequency
    plt.figure(figsize=(8, 5))
    thomp_df = df[df["controller"] == "thompson"].copy()
    counts = thomp_df["arm_name"].value_counts()
    counts.plot(kind="bar")
    plt.title("Thompson Sampling: Arm Selection Frequency")
    plt.ylabel("Selections")
    plt.tight_layout()
    plt.savefig(out_dir / "thompson_arm_frequency.png")
    plt.close()
    
    # 2. Posterior Means
    thomp_df["posterior_mean"] = thomp_df["alpha_after"] / (thomp_df["alpha_after"] + thomp_df["beta_after"])
    plt.figure(figsize=(10, 6))
    for arm_name in thomp_df["arm_name"].unique():
        arm_data = thomp_df[thomp_df["arm_name"] == arm_name]
        example_data = arm_data[(arm_data["sample_id"] == "row_3577") & (arm_data["seed"] == 42)]
        if not example_data.empty:
            plt.plot(example_data["iteration"], example_data["posterior_mean"], label=arm_name)
    
    plt.title("Posterior Means (E[θ]) Over Iterations (Example Trajectory)")
    plt.ylabel("E[θ]")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "thompson_posterior_means.png")
    plt.close()
    
    # 3. Reward by step
    plt.figure(figsize=(10, 6))
    thomp_df["reward_smooth"] = thomp_df.groupby(["sample_id", "seed", "budget"])["reward"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    avg_reward = thomp_df.groupby("iteration")["reward_smooth"].mean()
    plt.plot(avg_reward.index, avg_reward.values)
    plt.title("Average Reward Over Time (Smoothed)")
    plt.ylabel("Reward (Rolling Mean)")
    plt.tight_layout()
    plt.savefig(out_dir / "thompson_reward_by_step.png")
    plt.close()
    
    # 4. Thompson vs Uniform (Drift comparison)
    plt.figure(figsize=(8, 5))
    data_to_plot = [sum_df["thompson_drift"].values, sum_df["uniform_drift"].values]
    plt.boxplot(data_to_plot, tick_labels=["Thompson", "Uniform"])
    plt.title("Best Adversarial Drift: Thompson vs Uniform")
    plt.ylabel(f"Drift ({units})")
    plt.tight_layout()
    plt.savefig(out_dir / "thompson_vs_uniform.png")
    plt.close()
    
    # 5. Regret Analysis
    plt.figure(figsize=(10, 6))
    reg_avg = reg_df.groupby(["iteration", "controller"])["empirical_pseudo_regret"].mean().unstack()
    if "thompson" in reg_avg.columns and "uniform" in reg_avg.columns:
        plt.plot(reg_avg.index, reg_avg["thompson"], label="Thompson", color="blue")
        plt.plot(reg_avg.index, reg_avg["uniform"], label="Uniform", color="orange")
    plt.title("Empirical Pseudo-Regret Over Time")
    plt.ylabel("Pseudo-Regret")
    plt.xlabel("Proposals Generated (Iteration)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "thompson_regret.png")
    plt.close()
    
    print("Plots generated in results/thompson/")

if __name__ == "__main__":
    main()
