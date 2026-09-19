import os
import pandas as pd
os.makedirs("../results/cross_domain", exist_ok=True)
agg = pd.read_csv("../results/bio_cyber/aggregated_results.csv")
agg.to_csv("../results/cross_domain/attack_strategy_comparison.csv", index=False)
