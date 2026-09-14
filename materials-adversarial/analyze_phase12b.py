import pandas as pd
import json

df = pd.read_csv("results/phase12b_adaptive_search/1789331655/search_summary.csv")

print("=== MAX DRIFT ===")
max_drift = df.groupby(["TargetModel", "Method", "Budget"])["TargetDrift"].max().reset_index()
print(max_drift)

print("\n=== MEAN DRIFT ===")
mean_drift = df.groupby(["TargetModel", "Method", "Budget"])["TargetDrift"].mean().reset_index()
print(mean_drift)

print("\n=== TRANSFER DRIFT (Optimized on Target, eval on other) ===")
transfer = df.groupby(["TargetModel", "Method", "Budget"])[["TargetDrift", "TransferDrift"]].mean().reset_index()
print(transfer)

print("\n=== VULNERABILITY AGREEMENT ===")
# Correlation of target drift vs transfer drift might be an proxy, but we can also just compute it
df_gnn = df[df["TargetModel"] == "GraphMPNN"].set_index(["SourceID", "Method", "Budget", "Seed"])["TargetDrift"]
df_tx = df[df["TargetModel"] == "Transformer"].set_index(["SourceID", "Method", "Budget", "Seed"])["TargetDrift"]

combined = pd.DataFrame({"GraphMPNN": df_gnn, "Transformer": df_tx}).dropna()
print("Pearson:", combined["GraphMPNN"].corr(combined["Transformer"], method='pearson'))
print("Spearman:", combined["GraphMPNN"].corr(combined["Transformer"], method='spearman'))

print("\n=== TOP 5 CANDIDATES ===")
top = pd.read_csv("results/phase12b_adaptive_search/1789331655/top_candidate_audit.csv").head(5)
print(top[["source", "candidate", "drift", "transfer_drift", "method", "edit_count"]])
