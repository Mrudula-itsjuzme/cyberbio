import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

agg = pd.read_csv("../results/materials/aggregated_results.csv")
per = pd.read_csv("../results/materials/per_example_results.csv")
pvs = pd.read_csv("../results/materials/proposal_validity_summary.csv")
os.makedirs("../figures", exist_ok=True)

ATTACKS = sorted(agg["attack_condition"].unique())
BUDGETS = [5, 20, 50]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
CMAP = {a: COLORS[i % len(COLORS)] for i, a in enumerate(ATTACKS)}

# ── 1. materials_drift_vs_budget.png ──────────────────────────────────────────
fig, ax = plt.subplots()
for atk in ATTACKS:
    sub = agg[agg["attack_condition"] == atk]
    ax.plot(sub["search_budget"], sub["mean_absolute_drift"], label=atk,
            marker="o", color=CMAP[atk])
ax.set_xlabel("Model query budget"); ax.set_ylabel("Mean |ΔTg| (absolute drift)")
ax.set_title("Materials: mean absolute drift vs model query budget")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout(); fig.savefig("../figures/materials_drift_vs_budget.png", dpi=120); plt.close(fig)

# ── 2. materials_model_queries_vs_attack.png ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(ATTACKS)); width = 0.25
for i, b in enumerate(BUDGETS):
    vals = [agg[(agg["attack_condition"]==a)&(agg["search_budget"]==b)]["mean_attack_model_queries"].values[0]
            if not agg[(agg["attack_condition"]==a)&(agg["search_budget"]==b)].empty else 0
            for a in ATTACKS]
    ax.bar(x + i*width, vals, width, label=f"budget={b}")
ax.set_xticks(x + width); ax.set_xticklabels(ATTACKS, rotation=30, ha="right")
ax.set_ylabel("Mean model queries (attack only)"); ax.set_title("Materials: attack model queries")
ax.legend(); plt.tight_layout()
fig.savefig("../figures/materials_model_queries_vs_attack.png", dpi=120); plt.close(fig)

# ── 3. materials_total_queries_vs_attack.png ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for i, b in enumerate(BUDGETS):
    vals = [agg[(agg["attack_condition"]==a)&(agg["search_budget"]==b)]["mean_total_model_queries"].values[0]
            if not agg[(agg["attack_condition"]==a)&(agg["search_budget"]==b)].empty else 0
            for a in ATTACKS]
    ax.bar(x + i*width, vals, width, label=f"budget={b}")
ax.set_xticks(x + width); ax.set_xticklabels(ATTACKS, rotation=30, ha="right")
ax.set_ylabel("Mean total model queries (attack + attribution)")
ax.set_title("Materials: total model queries (end-to-end)")
ax.legend(); plt.tight_layout()
fig.savefig("../figures/materials_total_queries_vs_attack.png", dpi=120); plt.close(fig)

# ── 4. materials_proposal_overhead.png ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for i, b in enumerate(BUDGETS):
    vals = [agg[(agg["attack_condition"]==a)&(agg["search_budget"]==b)]["mean_proposal_attempts"].values[0]
            if not agg[(agg["attack_condition"]==a)&(agg["search_budget"]==b)].empty else 0
            for a in ATTACKS]
    ax.bar(x + i*width, vals, width, label=f"budget={b}")
ax.set_xticks(x + width); ax.set_xticklabels(ATTACKS, rotation=30, ha="right")
ax.set_ylabel("Mean proposal attempts"); ax.set_title("Materials: proposal overhead per attack condition")
ax.legend(); plt.tight_layout()
fig.savefig("../figures/materials_proposal_overhead.png", dpi=120); plt.close(fig)

# ── 5. materials_rdkit_proposal_validity.png ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for i, b in enumerate(BUDGETS):
    vals = [pvs[(pvs["attack_condition"]==a)&(pvs["search_budget"]==b)]["rdkit_proposal_validity_rate"].values[0]
            if not pvs[(pvs["attack_condition"]==a)&(pvs["search_budget"]==b)].empty else np.nan
            for a in ATTACKS]
    ax.bar(x + i*width, vals, width, label=f"budget={b}")
ax.set_xticks(x + width); ax.set_xticklabels(ATTACKS, rotation=30, ha="right")
ax.set_ylabel("RDKit proposal validity rate"); ax.set_ylim(0, 1)
ax.set_title("Materials: RDKit SMILES parse validity rate (proposals)")
ax.legend(); plt.tight_layout()
fig.savefig("../figures/materials_rdkit_proposal_validity.png", dpi=120); plt.close(fig)

# ── 6. materials_runtime_vs_attack.png ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for i, b in enumerate(BUDGETS):
    vals = [agg[(agg["attack_condition"]==a)&(agg["search_budget"]==b)]["mean_runtime"].values[0]
            if not agg[(agg["attack_condition"]==a)&(agg["search_budget"]==b)].empty else 0
            for a in ATTACKS]
    ax.bar(x + i*width, vals, width, label=f"budget={b}")
ax.set_xticks(x + width); ax.set_xticklabels(ATTACKS, rotation=30, ha="right")
ax.set_ylabel("Mean runtime (s)"); ax.set_title("Materials: runtime per attack condition")
ax.legend(); plt.tight_layout()
fig.savefig("../figures/materials_runtime_vs_attack.png", dpi=120); plt.close(fig)

# ── 7. materials_attribution_comparison.png ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
attr_attacks = ["Attribution-High", "Attribution-Low", "Attribution-Random", "Random"]
for atk in attr_attacks:
    sub = agg[agg["attack_condition"] == atk]
    if sub.empty: continue
    # search-only efficiency
    axes[0].plot(sub["search_budget"], sub["mean_absolute_drift"],
                 label=atk, marker="o", color=CMAP.get(atk))
    axes[1].plot(sub["search_budget"], sub["mean_total_model_queries"],
                 label=atk, marker="o", color=CMAP.get(atk))
axes[0].set_title("Attribution comparison: drift (search-only budget)")
axes[0].set_xlabel("Attack model query budget"); axes[0].set_ylabel("Mean |ΔTg|")
axes[0].legend(fontsize=8)
axes[1].set_title("Attribution comparison: total model queries")
axes[1].set_xlabel("Attack model query budget"); axes[1].set_ylabel("Mean total queries")
axes[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig("../figures/materials_attribution_comparison.png", dpi=120); plt.close(fig)

print("Materials figures written.")
