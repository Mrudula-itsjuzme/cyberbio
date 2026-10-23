import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bc_agg = pd.read_csv("../results/bio_cyber/aggregated_results.csv")
mat_agg = pd.read_csv("../results/materials/aggregated_results.csv")
os.makedirs("../figures", exist_ok=True)

# ── rebuild cross_domain comparison CSV ───────────────────────────────────────
bc_agg["domain"] = "bio_cyber"
mat_agg["domain"] = "materials"

SHARED = ["domain","attack_condition","search_budget",
          "mean_edit_distance","mean_attack_model_queries",
          "mean_attribution_queries","mean_total_model_queries","mean_runtime","mean_proposal_attempts"]

def safe_col(df, col):
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)

bc_shared = bc_agg[list(set(SHARED) & set(bc_agg.columns))].copy()
mat_shared = mat_agg[list(set(SHARED) & set(mat_agg.columns))].copy()

# domain-specific columns
bc_shared["bio_cyber_label_flip_rate"] = safe_col(bc_agg, "success_rate").values if "success_rate" in bc_agg.columns else np.nan
bc_shared["bio_cyber_probability_drift"] = safe_col(bc_agg, "mean_absolute_drift").values if "mean_absolute_drift" in bc_agg.columns else np.nan
bc_shared["materials_Tg_absolute_drift"] = np.nan
bc_shared["materials_rdkit_proposal_validity_rate"] = np.nan

mat_shared["bio_cyber_label_flip_rate"] = np.nan
mat_shared["bio_cyber_probability_drift"] = np.nan
mat_shared["materials_Tg_absolute_drift"] = mat_agg["mean_absolute_drift"].values if "mean_absolute_drift" in mat_agg.columns else np.nan
mat_shared["materials_rdkit_proposal_validity_rate"] = mat_agg["rdkit_proposal_validity_rate"].values if "rdkit_proposal_validity_rate" in mat_agg.columns else np.nan

cross = pd.concat([bc_shared, mat_shared], ignore_index=True)
os.makedirs("../results/cross_domain", exist_ok=True)
cross.to_csv("../results/cross_domain/attack_strategy_comparison.csv", index=False)

# ── helper: grouped bar ────────────────────────────────────────────────────────
ATTACKS = sorted(cross["attack_condition"].unique())
BUDGETS = [5, 20, 50]
x = np.arange(len(ATTACKS)); width = 0.15
DOMAIN_COLORS = {"bio_cyber": "#2196F3", "materials": "#FF5722"}

def domain_grouped_bar(ax, col, title, ylabel):
    domains = ["bio_cyber", "materials"]
    for i, (b, domain) in enumerate([(b, d) for b in BUDGETS for d in domains]):
        sub = cross[(cross["search_budget"]==b) & (cross["domain"]==domain)]
        vals = [sub[sub["attack_condition"]==a][col].values[0]
                if (not sub[sub["attack_condition"]==a].empty and col in sub.columns) else np.nan
                for a in ATTACKS]
        offset = (i - 3)*width
        bars = ax.bar(x + offset, vals, width,
                      label=f"{domain} b={b}",
                      color=DOMAIN_COLORS[domain], alpha=0.4+0.2*(BUDGETS.index(b)))
    ax.set_xticks(x); ax.set_xticklabels(ATTACKS, rotation=30, ha="right")
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2)

# 1. cross_domain_model_query_cost.png
fig, ax = plt.subplots(figsize=(11, 5))
domain_grouped_bar(ax, "mean_attack_model_queries",
                   "Cross-domain: attack model query cost", "Mean attack model queries")
plt.tight_layout(); fig.savefig("../figures/cross_domain_model_query_cost.png", dpi=120); plt.close(fig)

# 2. cross_domain_total_query_cost.png
fig, ax = plt.subplots(figsize=(11, 5))
domain_grouped_bar(ax, "mean_total_model_queries",
                   "Cross-domain: total model query cost (attack + attribution)", "Mean total model queries")
plt.tight_layout(); fig.savefig("../figures/cross_domain_total_query_cost.png", dpi=120); plt.close(fig)

print("Cross-domain figures written.")
