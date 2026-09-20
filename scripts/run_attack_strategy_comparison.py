#!/usr/bin/env python3
"""Rebuild provenance-qualified attack-comparison outputs.

This entrypoint never executes credential- or compute-blocked methods implicitly.
Classical/generative execution remains in the domain-specific scripts; --analysis
rebuilds the final table, statistics, quality summary, and figures from recorded runs.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "comparison-experiments/attack-strategy-benchmark"
OUT = ROOT / "results/final"
FIG = OUT / "figures/attack_strategy_comparison"


def q90(s: pd.Series) -> float:
    return float(s.quantile(0.9))


def row_from_group(domain: str, family: str, variant: str, status: str,
                   budget: int | float, g: pd.DataFrame, source: str,
                   drift_col: str, query_col: str, proposal_col: str,
                   attr_col: str, runtime_col: str, success_col: str | None,
                   valid_col: str | None) -> dict:
    drift = pd.to_numeric(g[drift_col], errors="coerce")
    queries = pd.to_numeric(g[query_col], errors="coerce")
    proposals = pd.to_numeric(g[proposal_col], errors="coerce")
    attrs = pd.to_numeric(g[attr_col], errors="coerce")
    success = pd.to_numeric(g[success_col], errors="coerce") if success_col else pd.Series(dtype=float)
    if valid_col:
        valid = pd.to_numeric(g[valid_col], errors="coerce")
        validity = float(valid.sum() / proposals.sum()) if proposals.sum() else np.nan
    else:
        validity = 1.0
    return {
        "domain": domain, "attack_family": family, "attack_variant": variant,
        "status": status, "budget": budget, "n_sources": int(g["source_id"].nunique()),
        "mean_drift": drift.mean(), "median_drift": drift.median(), "p90_drift": q90(drift),
        "max_drift": drift.max(), "success_rate": success.mean() if len(success) else np.nan,
        "validity_rate": validity, "target_queries": queries.sum(),
        "proposal_count": proposals.sum(), "attribution_queries": attrs.sum(),
        "llm_calls": np.nan, "llm_tokens": np.nan,
        "runtime": pd.to_numeric(g[runtime_col], errors="coerce").sum(),
        "seed": 42, "source_artifact": source,
    }


def build_table() -> pd.DataFrame:
    rows: list[dict] = []
    for domain in ("bio_cyber", "materials"):
        rel = f"comparison-experiments/attack-strategy-benchmark/results/{domain}/per_example_results.csv"
        d = pd.read_csv(ROOT / rel)
        for (attack, budget), g in d.groupby(["attack_condition", "search_budget"], sort=True):
            family = "Attribution-guided" if attack.startswith("Attribution-") else attack
            rows.append(row_from_group(
                domain, family, attack, "EXECUTED", budget, g, rel,
                "absolute_prediction_change",
                "search_queries" if domain == "bio_cyber" else "attack_model_queries",
                "search_queries" if domain == "bio_cyber" else "proposal_attempts",
                "attribution_queries", "runtime_seconds",
                "success" if domain == "bio_cyber" else None,
                None if domain == "bio_cyber" else "valid_candidates"))

    for domain in ("bio_cyber", "materials"):
        folder = BENCH / f"results/generative/{domain}"
        for variant, filename in (("Generative-Trained", "per_example_results_trained.csv"),
                                  ("Generative-Untrained", "per_example_results_untrained.csv")):
            p = folder / filename
            if not p.exists():
                continue
            rel = str(p.relative_to(ROOT))
            d = pd.read_csv(p)
            for budget, g in d.groupby("search_budget"):
                rows.append(row_from_group(
                    domain, "Generative", variant, "EXECUTED", budget, g, rel,
                    "absolute_drift", "attack_model_queries", "proposal_attempts",
                    "attribution_queries", "runtime_seconds",
                    "success" if domain == "bio_cyber" else None, "valid_candidates"))

    # Explicit non-measurements make absence machine-readable without fake numbers.
    blocked = [
        ("materials", "Generative", "Generative-Untrained", "INCOMPLETE_BLOCKED_TIMEOUT"),
        ("bio_cyber", "LLM", "LLM-Proposal", "BLOCKED_EXTERNAL_CREDENTIALS"),
        ("materials", "LLM", "LLM-Proposal", "BLOCKED_EXTERNAL_CREDENTIALS"),
        ("bio_cyber", "RL", "RL-Policy", "BLOCKED_COMPUTE"),
        ("materials", "RL", "RL-Policy", "BLOCKED_COMPUTE"),
        ("bio_cyber", "GAN", "GAN-like", "REJECTED_METHOD"),
        ("materials", "GAN", "GAN-like", "REJECTED_METHOD"),
    ]
    for domain, family, variant, status in blocked:
        rows.append({k: v for k, v in zip(
            ["domain", "attack_family", "attack_variant", "status", "budget", "n_sources",
             "mean_drift", "median_drift", "p90_drift", "max_drift", "success_rate",
             "validity_rate", "target_queries", "proposal_count", "attribution_queries",
             "llm_calls", "llm_tokens", "runtime", "seed", "source_artifact"],
            [domain, family, variant, status] + [np.nan] * 16)})
    return pd.DataFrame(rows)


def statistics() -> dict:
    from scipy.stats import wilcoxon
    out = {"method": "two-sided Wilcoxon signed-rank on matched source rows", "comparisons": []}
    for domain in ("bio_cyber", "materials"):
        p = BENCH / f"results/{domain}/per_example_results.csv"
        d = pd.read_csv(p)
        drift = "absolute_prediction_change"
        for budget in sorted(d.search_budget.unique()):
            z = d[d.search_budget == budget].pivot(index="source_id", columns="attack_condition", values=drift)
            if "Random" not in z:
                continue
            for attack in [c for c in z if c != "Random"]:
                pair = z[["Random", attack]].dropna()
                diff = pair[attack] - pair["Random"]
                try:
                    stat, pval = wilcoxon(pair[attack], pair["Random"], alternative="two-sided")
                except ValueError:
                    stat, pval = np.nan, 1.0
                out["comparisons"].append({
                    "domain": domain, "budget": int(budget), "attack": attack,
                    "reference": "Random", "n": len(pair), "statistic": stat,
                    "p_value": pval, "median_paired_difference": diff.median(),
                    "effect_size": "median paired drift difference",
                })
    return out


def quality_summary(table: pd.DataFrame) -> dict:
    out = {"policy": "real recorded runs only; fabricated fast-control rows excluded"}
    for domain in ("bio_cyber", "materials"):
        out[domain] = {}
        for variant in ("Generative-Trained", "Generative-Untrained"):
            z = table[(table.domain == domain) & (table.attack_variant == variant) &
                      (table.status == "EXECUTED")]
            if z.empty:
                out[domain][variant] = {"status": "incomplete", "validity_rate": None,
                    "accepted_proposals": None, "proposal_total": None, "attack_success": None}
            else:
                out[domain][variant] = {"status": "executed", "validity_rate": float(z.validity_rate.mean()),
                    "accepted_proposals": int(z.target_queries.sum()), "proposal_total": int(z.proposal_count.sum()),
                    "attack_success": None if z.success_rate.isna().all() else float(z.success_rate.mean())}
    return out


def figures(table: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    executed = table[table.status == "EXECUTED"].copy()
    views = [
        ("effectiveness", "mean_drift", "Mean absolute drift"),
        ("drift_per_query", None, "Mean drift per target query"),
        ("validity", "validity_rate", "Valid candidate rate"),
        ("success", "success_rate", "Attack success rate"),
        ("runtime", "runtime", "Total runtime (s)"),
    ]
    for name, col, ylabel in views:
        z = executed[executed.budget == 50].copy()
        if name == "drift_per_query":
            z["metric"] = z.mean_drift / (z.target_queries / z.n_sources)
        else:
            z["metric"] = z[col]
        z = z.dropna(subset=["metric"])
        fig, ax = plt.subplots(figsize=(11, 5))
        labels = z.domain.str.replace("_", "-") + "\n" + z.attack_variant
        ax.bar(range(len(z)), z.metric)
        ax.set_xticks(range(len(z)), labels, rotation=55, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(f"Attack comparison: {ylabel} (budget 50)")
        fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=180); plt.close(fig)
    status = table.drop_duplicates(["domain", "attack_variant", "status"])
    fig, ax = plt.subplots(figsize=(11, 5))
    counts = status.groupby(["status", "domain"]).size().unstack(fill_value=0)
    counts.plot.bar(ax=ax); ax.set_ylabel("Domain-method cells"); ax.set_title("Attack family evidence status")
    fig.tight_layout(); fig.savefig(FIG / "status_overview.png", dpi=180); plt.close(fig)

    # Budget trajectory from executed data.
    fig, ax = plt.subplots(figsize=(9, 5))
    for (domain, variant), g in executed.groupby(["domain", "attack_variant"]):
        g = g.sort_values("budget"); ax.plot(g.budget, g.mean_drift, marker="o", label=f"{domain}:{variant}")
    ax.set_xlabel("Target-query budget"); ax.set_ylabel("Mean absolute drift"); ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "budget_trajectories.png", dpi=180); plt.close(fig)

    hist = BENCH / "results/generative/checkpoints"
    fig, ax = plt.subplots(figsize=(8, 5))
    for domain in ("bio_cyber", "materials"):
        p = hist / domain / "training_history.csv"
        if p.exists():
            h = pd.read_csv(p); ax.plot(h.epoch, h.val_loss, label=f"{domain} validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG / "generator_training_loss.png", dpi=180); plt.close(fig)

    gen = executed[(executed.attack_family == "Generative") & (executed.budget == 20)].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not gen.empty:
        labels = gen.domain.str.replace("_", "-") + "\n" + gen.attack_variant
        ax.bar(range(len(gen)), gen.validity_rate)
        ax.set_xticks(range(len(gen)), labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Valid candidate rate")
    ax.set_title("Recorded generator controls (missing Materials control omitted)")
    fig.tight_layout(); fig.savefig(FIG / "generator_control.png", dpi=180); plt.close(fig)

    stats = statistics()["comparisons"]
    s = pd.DataFrame(stats)
    fig, ax = plt.subplots(figsize=(10, 5))
    s = s[s.budget == 50].copy()
    labels = s.domain.str.replace("_", "-") + "\n" + s.attack
    ax.bar(range(len(s)), s.median_paired_difference)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(s)), labels, rotation=50, ha="right")
    ax.set_ylabel("Median paired drift difference vs Random")
    ax.set_title("Paired effect summary (budget 50; see JSON for p-values)")
    fig.tight_layout(); fig.savefig(FIG / "statistical_comparison.png", dpi=180); plt.close(fig)


def run_analysis() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_table()
    table.to_csv(OUT / "attack_strategy_comparison.csv", index=False, na_rep="NA")
    (OUT / "statistical_results.json").write_text(json.dumps(statistics(), indent=2, default=float) + "\n")
    quality = quality_summary(table)
    (OUT / "generator_quality_summary.json").write_text(json.dumps(quality, indent=2) + "\n")
    figures(table)

    # Compatibility artifacts for the original benchmark tests/readers. They contain
    # only provenance-qualified rows and point conceptually to results/final.
    legacy_cross = BENCH / "results/cross_domain"
    legacy_cross.mkdir(parents=True, exist_ok=True)
    legacy = table.rename(columns={"attack_variant": "attack_condition"}).copy()
    legacy.loc[legacy.attack_condition == "Generative-Trained", "attack_condition"] = "Generative"
    legacy["materials_rdkit_proposal_validity_rate"] = np.where(
        legacy.domain == "materials", legacy.validity_rate, np.nan)
    legacy["bio_cyber_label_flip_rate"] = np.where(
        legacy.domain == "bio_cyber", legacy.success_rate, np.nan)
    legacy.to_csv(legacy_cross / "attack_strategy_comparison.csv", index=False, na_rep="NA")
    shutil.copyfile(FIG / "drift_per_query.png", legacy_cross / "cross_domain_efficiency.png")
    legacy_gen = BENCH / "results/generative"
    (legacy_gen / "generator_quality_summary.json").write_text(json.dumps(quality, indent=2) + "\n")
    mat = table[(table.domain == "materials") & (table.attack_variant == "Generative-Trained")]
    mat.to_csv(legacy_gen / "materials/aggregated_results.csv", index=False, na_rep="NA")
    for domain in ("bio_cyber", "materials"):
        src = legacy_gen / f"checkpoints/{domain}/training_manifest.json"
        dst = legacy_gen / f"{domain}/training_manifest.json"
        if src.exists(): shutil.copyfile(src, dst)
    print(f"Wrote {len(table)} comparison rows to {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", action="store_true", help="rebuild final analysis from recorded real runs")
    ap.add_argument("--classical", action="store_true", help="print the explicit classical execution command")
    ap.add_argument("--generative", action="store_true", help="print the explicit generative execution commands")
    ap.add_argument("--llm", action="store_true", help="fail clearly: real provider credentials are absent")
    ap.add_argument("--rl", action="store_true", help="fail clearly: no trained policy/run exists")
    ap.add_argument("--all", action="store_true", help="analysis only; blocked methods are never silently run")
    a = ap.parse_args()
    if a.llm: ap.error("LLM execution blocked: configure a real provider and credentials; mock providers are test-only")
    if a.rl: ap.error("RL execution blocked: no implemented/trained policy evidence exists")
    if a.classical: print("python3 comparison-experiments/attack-strategy-benchmark/scripts/run_benchmark.py")
    if a.generative: print("Run domain-specific training/benchmark scripts explicitly; Materials untrained remains timeout-blocked")
    if a.analysis or a.all or not any(vars(a).values()): run_analysis()


if __name__ == "__main__":
    main()
