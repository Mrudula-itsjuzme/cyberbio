import pandas as pd
import os

df = pd.read_csv("results/comparison/experiments/summaries/budget_sweep_results.csv")
counts = df.groupby(['attack_family', 'query_budget', 'seed']).size().reset_index(name='count')
print("--- Sweep Counts ---")
print(counts)
print(f"Total rows: {len(df)}")
