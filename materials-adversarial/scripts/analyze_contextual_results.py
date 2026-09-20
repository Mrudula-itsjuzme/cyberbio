#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    out_dir = Path("results/contextual_thompson")
    test_results_path = out_dir / "test_results.csv"
    if not test_results_path.exists():
        print("No test results found.")
        return
        
    df = pd.read_csv(test_results_path)
    
    # 1. Regret calculation
    # Regret = Hindsight Oracle - Controller Reward
    df["regret_unif"] = df["hindsight_best_reward"] - df["uniform_reward"]
    df["regret_thomp"] = df["hindsight_best_reward"] - df["thompson_reward"]
    df["regret_ctx"] = df["hindsight_best_reward"] - df["contextual_reward"]
    df["regret_global"] = df["hindsight_best_reward"] - df["global_best_reward"]
    df["regret_shuf"] = df["hindsight_best_reward"] - df["shuffled_context_reward"]
    
    df.to_csv(out_dir / "regret.csv", index=False)
    
    budgets = sorted(df["budget"].unique())
    
    print("\n--- Mean Empirical Pseudo-Regret ---")
    for b in budgets:
        b_df = df[df["budget"] == b]
        print(f"Budget {b}:")
        print(f"  Uniform:      {b_df['regret_unif'].mean():.2f}")
        print(f"  Thompson:     {b_df['regret_thomp'].mean():.2f}")
        print(f"  Contextual:   {b_df['regret_ctx'].mean():.2f}")
        print(f"  Global Best:  {b_df['regret_global'].mean():.2f}")
        print(f"  Shuffled Ctx: {b_df['regret_shuf'].mean():.2f}")
        
    # Plot Mean Reward
    plt.figure(figsize=(10, 6))
    means = df.groupby("budget")[["uniform_reward", "thompson_reward", "contextual_reward", "global_best_reward", "shuffled_context_reward"]].mean()
    plt.plot(means.index, means["contextual_reward"], label="Contextual Thompson", marker='o')
    plt.plot(means.index, means["thompson_reward"], label="Non-Contextual Thompson", marker='s')
    plt.plot(means.index, means["uniform_reward"], label="Uniform", marker='^')
    plt.plot(means.index, means["global_best_reward"], label="Global Best", marker='d')
    plt.plot(means.index, means["shuffled_context_reward"], label="Shuffled Context (Control)", marker='x', linestyle='--')
    plt.title("Cumulative Reward on Held-Out Test Sources")
    plt.xlabel("Query Budget")
    plt.ylabel("Mean Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "contextual_vs_noncontextual_reward.png")
    plt.close()
    
    print("\nStatistical design requires paired comparisons at Budget 10:")
    b10 = df[df["budget"] == 10]
    diff = b10["contextual_reward"] - b10["thompson_reward"]
    print(f"Contextual vs Thompson Mean Diff: {diff.mean():.2f} (Std: {diff.std():.2f})")
    
    diff_unif = b10["contextual_reward"] - b10["uniform_reward"]
    print(f"Contextual vs Uniform Mean Diff: {diff_unif.mean():.2f}")
    
    print("\nResults and plots generated.")

if __name__ == "__main__":
    main()
