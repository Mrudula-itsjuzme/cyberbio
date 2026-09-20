import pandas as pd
import json

df = pd.read_csv("results/thompson/fixed_arm_rewards.csv")

# We want total cumulative reward per arm per source per budget per seed.
# The maximum iteration row gives the total cumulative reward.
idx = df.groupby(["sample_id", "budget", "seed", "arm"])["iteration"].idxmax()
totals = df.loc[idx]

results = []
for (sample, budget, seed), group in totals.groupby(["sample_id", "budget", "seed"]):
    sorted_group = group.sort_values(by="cumulative_reward", ascending=False)
    arms = sorted_group["arm"].values
    rewards = sorted_group["cumulative_reward"].values
    
    best_arm = arms[0]
    best_reward = rewards[0]
    second_best_arm = arms[1] if len(arms) > 1 else None
    second_best_reward = rewards[1] if len(rewards) > 1 else 0.0
    margin = best_reward - second_best_reward
    
    results.append({
        "sample_id": sample,
        "budget": budget,
        "seed": seed,
        "best_arm": best_arm,
        "second_best_arm": second_best_arm,
        "best_reward": best_reward,
        "second_best_reward": second_best_reward,
        "margin": margin
    })

res_df = pd.DataFrame(results)
res_df.to_csv("results/thompson/step8_summary.csv", index=False)
print(res_df.groupby(["sample_id", "best_arm"]).size())
