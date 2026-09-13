import pandas as pd
import json
from pathlib import Path

out_dir = Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/results/phase12_graph_adversarial/1789329207")

corr_df = pd.read_csv(out_dir / "architecture_drift_correlation.csv")
print("CORRELATION")
print(corr_df)

adapt_df = pd.read_csv(out_dir / "adaptive_search_summary.csv")
print("\nADAPTIVE SEARCH SUMMARY")
for q in [50]:
    df_q = adapt_df[adapt_df["Budget"] == q]
    gnn_target = df_q[df_q["TargetModel"] == "GraphMPNN"]
    print(f"GraphMPNN Vulnerability Budget {q}: Max Drift = {gnn_target['TargetDrift'].max():.3f}, Mean = {gnn_target['TargetDrift'].mean():.3f}")
    
    print(f"Cross-Model Transfer (Target=GraphMPNN, evaluated on TX): Mean = {gnn_target['TransferDrift'].mean():.3f}")
    
    tx_target = df_q[df_q["TargetModel"] == "Transformer"]
    print(f"Cross-Model Transfer (Target=TX, evaluated on GraphMPNN): Mean = {tx_target['TransferDrift'].mean():.3f}")

