import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

out_gen = ROOT / "results/generative"
out_gen.mkdir(parents=True, exist_ok=True)
out_cd = ROOT / "results/cross_domain"
out_cd.mkdir(parents=True, exist_ok=True)

df_cd = pd.read_csv(out_cd / "attack_strategy_comparison.csv")
bc_df = pd.read_csv(ROOT / "results/generative/bio_cyber/aggregated_results.csv")
mat_df = pd.read_csv(ROOT / "results/generative/materials/aggregated_results.csv")

# 1. results/cross_domain/cross_domain_efficiency.png
# Compare Generative vs Random, MCMC, Evolutionary, Attribution-High across both domains
# Y-axis: drift / queries (efficiency).
def safe_val(df, cond, b, domain, metric):
    sub = df[(df["attack_condition"]==cond) & (df["search_budget"]==b) & (df["domain"]==domain)]
    if sub.empty: return 0.0
    val = sub.iloc[0][metric]
    return val if pd.notna(val) else 0.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
conds = ["Generative", "Random", "MCMC", "Evolutionary", "Attribution-High"]
budgets = [5, 20, 50]
x = np.arange(len(conds)); width = 0.25
for ax, domain, drift_col in zip([ax1, ax2], ["bio_cyber", "materials"], ["bio_cyber_probability_drift", "materials_Tg_absolute_drift"]):
    for i, b in enumerate(budgets):
        vals = []
        for c in conds:
            d = safe_val(df_cd, c, b, domain, drift_col)
            q = safe_val(df_cd, c, b, domain, "mean_total_model_queries")
            vals.append(d / q if q > 0 else 0)
        ax.bar(x + (i - 1)*width, vals, width, label=f"Budget {b}")
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=30, ha="right")
    ax.set_title(f"{domain} Efficiency (Drift / Total Queries)")
    ax.legend()
plt.tight_layout(); fig.savefig(out_cd / "cross_domain_efficiency.png", dpi=120); plt.close(fig)

# 2. generative/bio_cyber_generative_trajectories.png
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sub = bc_df[bc_df["attack_condition"] == "Generative"]
axes[0].bar(sub["search_budget"].astype(str), sub["edit_distance"])
axes[0].set_title("Edit Distance")
axes[1].bar(sub["search_budget"].astype(str), sub["total_model_queries"])
axes[1].set_title("Query Count")
axes[2].bar(sub["search_budget"].astype(str), sub["success"])
axes[2].set_title("Success Rate")
fig.suptitle("Bio-Cyber Generative Trajectories")
plt.tight_layout(); fig.savefig(out_gen / "bio_cyber_generative_trajectories.png", dpi=120); plt.close(fig)

# 3. generative/materials_generative_trajectories.png
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sub = mat_df[mat_df["attack_condition"] == "Generative"]
axes[0].bar(sub["search_budget"].astype(str), sub["edit_distance"])
axes[0].set_title("Edit Distance")
axes[1].bar(sub["search_budget"].astype(str), sub["total_model_queries"])
axes[1].set_title("Query Count")
axes[2].bar(sub["search_budget"].astype(str), sub["absolute_drift"])
axes[2].set_title("Absolute Drift")
fig.suptitle("Materials Generative Trajectories")
plt.tight_layout(); fig.savefig(out_gen / "materials_generative_trajectories.png", dpi=120); plt.close(fig)

# 4. generative/bio_cyber_ablation.png
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sub = bc_df[bc_df["search_budget"] == 20]
gen = sub[sub["attack_condition"] == "Generative"]
un = sub[sub["attack_condition"] == "Generative-Untrained"]
g_succ = gen["success"].iloc[0] if not gen.empty else 0
u_succ = un["success"].iloc[0] if not un.empty else 0
axes[0].bar(["Generative", "Untrained"], [g_succ, u_succ])
axes[0].set_title("Success Rate")
g_q = gen["total_model_queries"].iloc[0] if not gen.empty else 0
u_q = un["total_model_queries"].iloc[0] if not un.empty else 0
axes[1].bar(["Generative", "Untrained"], [g_q, u_q])
axes[1].set_title("Total Model Queries")
fig.suptitle("Bio-Cyber Ablation (Budget 20)")
plt.tight_layout(); fig.savefig(out_gen / "bio_cyber_ablation.png", dpi=120); plt.close(fig)

# 5. generative/materials_ablation.png
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sub = mat_df[mat_df["search_budget"] == 20]
gen = sub[sub["attack_condition"] == "Generative"]
un = sub[sub["attack_condition"] == "Generative-Untrained"]
g_drift = gen["absolute_drift"].iloc[0] if not gen.empty else 0
u_drift = un["absolute_drift"].iloc[0] if not un.empty else 0
axes[0].bar(["Generative", "Untrained"], [g_drift, u_drift])
axes[0].set_title("Absolute Drift")
g_val = (gen["valid_candidates"]/gen["proposal_attempts"]).iloc[0] if not gen.empty and gen["proposal_attempts"].iloc[0]>0 else 0
u_val = (un["valid_candidates"]/un["proposal_attempts"]).iloc[0] if not un.empty and un["proposal_attempts"].iloc[0]>0 else 0
axes[1].bar(["Generative", "Untrained"], [g_val, u_val])
axes[1].set_title("Validity Rate")
fig.suptitle("Materials Ablation (Budget 20)")
plt.tight_layout(); fig.savefig(out_gen / "materials_ablation.png", dpi=120); plt.close(fig)

# 6. generative/generator_training_loss.png
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
try:
    bc_tr = pd.read_csv(ROOT / "results/generative/bio_cyber/training_history.csv")
    ax1.plot(bc_tr["epoch"], bc_tr["train_loss"], label="Train")
    ax1.plot(bc_tr["epoch"], bc_tr["val_loss"], label="Val")
    ax1.set_title("Bio-Cyber Training Loss")
    ax1.legend()
except Exception as e:
    ax1.set_title(f"Bio-Cyber Loss (Not found)")
    
try:
    mat_tr = pd.read_csv(ROOT / "results/generative/materials/training_history.csv")
    ax2.plot(mat_tr["epoch"], mat_tr["train_loss"], label="Train")
    ax2.plot(mat_tr["epoch"], mat_tr["val_loss"], label="Val")
    ax2.set_title("Materials Training Loss")
    ax2.legend()
except Exception as e:
    ax2.set_title(f"Materials Loss (Not found)")

fig.suptitle("Generator Training Loss")
plt.tight_layout(); fig.savefig(out_gen / "generator_training_loss.png", dpi=120); plt.close(fig)

print("Generated all figures.")
