#!/usr/bin/env python3
"""Run and analyze the first matched-target-query probabilistic-search pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baselines import EvolutionarySearch, MCMCSearch, RandomSearch
from cem import CEMSearch
from probabilistic_search_lab.canonical import CanonicalValidator, load_model, load_sources
from probabilistic_search_lab.proposals import canonical_families
from smc import SMCSearch
from thompson import ThompsonSearch
from validity_smc import EmpiricalValidityModel

METHODS = ("random", "mcmc", "evolutionary", "thompson", "cem", "smc", "validity_smc")
DEFAULT_BUDGETS = (25, 50, 100)
DEFAULT_SEEDS = (20260920, 20260921, 20260922)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sources", type=int, default=10)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pilot")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_search(method, families):
    constructors = {
        "random": lambda: RandomSearch(families),
        "mcmc": MCMCSearch,
        "evolutionary": lambda: EvolutionarySearch(families),
        "thompson": lambda: ThompsonSearch(families),
        "cem": lambda: CEMSearch(families),
        "smc": lambda: SMCSearch(families),
        "validity_smc": lambda: SMCSearch(families, validity_model=EmpiricalValidityModel()),
    }
    return constructors[method]()


def token_length(value: str) -> int:
    from materials_adv.data.tokenizer import tokenize
    try:
        return len(tokenize(value))
    except Exception:
        return len(value)


def candidate_metadata(source, candidate, validator):
    if not candidate:
        return {"length_change": None, "tanimoto_similarity": None,
                "representation_valid": None, "domain_constraint_pass": None,
                "validity_reasons": None}
    from materials_adv.domain.chemistry.plausibility import compute_tanimoto_similarity
    checked = validator.validate(source, candidate)
    return {
        "length_change": token_length(candidate) - token_length(source),
        "tanimoto_similarity": compute_tanimoto_similarity(source, candidate),
        "representation_valid": checked.representation_valid,
        "domain_constraint_pass": checked.domain_constraint_pass,
        "validity_reasons": list(checked.reasons),
    }


def trajectory_rows(result):
    rows = []
    best = 0.0
    if result.method.startswith("thompson"):
        queries = 1
        for step, item in enumerate(result.trajectories.get("posterior", [])):
            effect = item.get("effect")
            if effect is not None:
                queries += 1; best = max(best, effect)
            rows.append((step, queries, best, item.get("family"), item.get("alpha"), item.get("beta"), None, None))
    elif result.method == "cem":
        for item in result.trajectories.get("cem", []):
            elite = item.get("elite_effects") or []
            best = max([best, *elite])
            rows.append((item["iteration"], item["queries"], best, None, None, None, item.get("entropy"), None))
    elif result.method in {"smc", "validity_smc"}:
        for item in result.trajectories.get("smc", []):
            rows.append((item["step"], item["queries"], None, None, None, None,
                         item.get("weight_entropy"), item.get("ess")))
    elif result.method == "evolutionary":
        for item in result.trajectories.get("generations", []):
            rows.append((item["generation"], item["queries"], item.get("best_effect"), None, None, None, None, None))
    if not rows or rows[-1][1] != result.metrics.target_model_queries:
        rows.append((len(rows), result.metrics.target_model_queries, result.metrics.best_effect,
                     None, None, None, None, None))
    return rows


def execute(args):
    if not 10 <= args.num_sources <= 20:
        raise ValueError("pilot requires 10-20 fixed test sources")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "pilot_progress.jsonl"
    completed = {}
    if args.resume and progress_path.exists():
        for line in progress_path.read_text().splitlines():
            record = json.loads(line)
            completed[tuple(record["run_key"])] = record

    model = load_model(args.device)
    validator = CanonicalValidator()
    families = canonical_families()
    sources = load_sources("test", args.num_sources)
    manifest = {
        "methods": list(METHODS), "budgets": list(args.budgets), "seeds": list(args.seeds),
        "source_ids": [source_id for source_id, _ in sources], "split": "test",
        "checkpoint": model.config(), "validator": "CanonicalValidator(min_tanimoto=0.5)",
        "primary_matching": "nominal target-model query budget; actual queries also reported",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total = len(METHODS) * len(args.budgets) * len(args.seeds) * len(sources)
    count = len(completed)
    for method in METHODS:
        for budget in args.budgets:
            for seed in args.seeds:
                for source_id, source in sources:
                    key = (method, budget, seed, source_id)
                    if key in completed:
                        continue
                    search = make_search(method, families)
                    started = time.time()
                    result = search.run(source_id=source_id, source=source, model=model,
                                        validator=validator, budget=budget, seed=seed)
                    metadata = candidate_metadata(source, result.best_candidate, validator)
                    record = {"run_key": list(key), "source": source, "result": result.to_dict(),
                              "candidate_metadata": metadata, "wall_finished_at": time.time()}
                    with progress_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                    completed[key] = record; count += 1
                    print(f"[{count}/{total}] {method} budget={budget} seed={seed} source={source_id} "
                          f"queries={result.metrics.target_model_queries}", flush=True)
    return manifest, list(completed.values())


def write_artifacts(output: Path, manifest, records):
    per_source, candidates, trajectories = [], [], []
    for record in records:
        result = record["result"]; metrics = result["metrics"]; meta = record["candidate_metadata"]
        method = result["method"].removesuffix("_binary")
        row = {
            "source_id": result["source_id"], "method": method, "seed": result["seed"],
            "query_budget": result["query_budget"], "proposal_count": metrics["proposal_count"],
            "valid_count": metrics["valid_count"], "invalid_count": metrics["invalid_count"],
            "target_model_queries": metrics["target_model_queries"], "best_candidate": result["best_candidate"],
            "best_drift": metrics["best_effect"], "mean_drift": metrics["mean_effect"],
            "median_drift": metrics["median_effect"], "p90_drift": metrics["p90_effect"],
            "edit_type": result["metadata"].get("best_edit_type"), "length_change": meta["length_change"],
            "tanimoto_similarity": meta["tanimoto_similarity"],
            "representation_valid": meta["representation_valid"],
            "domain_constraint_pass": meta["domain_constraint_pass"],
            "validity_reasons": json.dumps(meta["validity_reasons"]),
            "runtime_seconds": metrics["runtime_seconds"], "failure_status": result["failure_status"],
        }
        row["validity_rate"] = row["valid_count"] / max(1, row["valid_count"] + row["invalid_count"])
        row["drift_per_query"] = row["best_drift"] / row["target_model_queries"] if row["best_drift"] is not None and row["target_model_queries"] else None
        per_source.append(row)
        if result["best_candidate"]:
            candidates.append({**row, "source": record["source"], "trajectories": result["trajectories"],
                               "method_metadata": result["metadata"]})
        for step, queries, best, family, alpha, beta, entropy, ess in trajectory_rows_object(result):
            trajectories.append({"source_id": result["source_id"], "method": method,
                                 "seed": result["seed"], "query_budget": result["query_budget"],
                                 "step": step, "target_model_queries": queries, "best_drift": best,
                                 "arm": family, "alpha": alpha, "beta": beta,
                                 "entropy": entropy, "ess": ess})

    frame = pd.DataFrame(per_source).sort_values(["query_budget", "method", "seed", "source_id"])
    frame.to_csv(output / "per_source_results.csv", index=False)
    with (output / "candidate_bank.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in candidates: handle.write(json.dumps(candidate, allow_nan=False) + "\n")
    pd.DataFrame(trajectories).to_csv(output / "query_trajectories.csv", index=False)

    summary = frame.groupby(["method", "query_budget"], dropna=False).agg(
        n_runs=("source_id", "size"), n_sources=("source_id", "nunique"),
        seeds=("seed", "nunique"), mean_best_drift=("best_drift", "mean"),
        median_best_drift=("best_drift", "median"), std_best_drift=("best_drift", "std"),
        mean_drift_per_query=("drift_per_query", "mean"), mean_validity_rate=("validity_rate", "mean"),
        mean_target_queries=("target_model_queries", "mean"), mean_proposals=("proposal_count", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"), failure_count=("failure_status", lambda x: x.notna().sum()),
    ).reset_index()
    summary.to_csv(output / "method_summary.csv", index=False)
    make_plots(output, frame, pd.DataFrame(trajectories), candidates)
    write_reports(output, frame, summary, candidates, pd.DataFrame(trajectories), manifest)


def trajectory_rows_object(result):
    class Object:
        pass
    obj = Object(); obj.method = result["method"]; obj.trajectories = result["trajectories"]
    obj.metrics = Object(); obj.metrics.target_model_queries = result["metrics"]["target_model_queries"]
    obj.metrics.best_effect = result["metrics"]["best_effect"]
    return trajectory_rows(obj)


def make_plots(output, frame, trajectories, candidates):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(name):
        plt.tight_layout(); plt.savefig(output / name, dpi=180); plt.close()

    for column, ylabel, name in [
        ("best_drift", "Best drift (eV)", "effect_vs_queries.png"),
        ("validity_rate", "Validity rate", "validity_rate.png"),
        ("runtime_seconds", "Runtime (s)", "runtime_comparison.png")]:
        grouped = frame.groupby(["query_budget", "method"])[column].mean().unstack()
        grouped.plot(marker="o"); plt.xlabel("Target-model query budget"); plt.ylabel(ylabel); save(name)

    sets = {method: set(group.best_candidate.dropna()) for method, group in frame.groupby("method")}
    methods = list(sets); matrix = np.zeros((len(methods), len(methods)))
    for i, left in enumerate(methods):
        for j, right in enumerate(methods):
            union = sets[left] | sets[right]; matrix[i, j] = len(sets[left] & sets[right]) / len(union) if union else np.nan
    plt.imshow(matrix, vmin=0, vmax=1, cmap="viridis"); plt.colorbar(label="Jaccard overlap")
    plt.xticks(range(len(methods)), methods, rotation=45, ha="right"); plt.yticks(range(len(methods)), methods); save("method_overlap.png")

    high = frame[frame.best_drift >= frame.best_drift.quantile(0.75)]
    high.edit_type.fillna("missing").value_counts().plot.bar(); plt.ylabel("Top-quartile best candidates"); save("edit_motif_distribution.png")
    for method, group in frame.groupby("method"):
        plt.scatter(group.length_change, group.best_drift, s=12, alpha=.45, label=method)
    plt.xlabel("Token length change"); plt.ylabel("Best drift (eV)"); plt.legend(fontsize=7); save("length_change_vs_drift.png")

    th = trajectories[trajectories.method == "thompson"].dropna(subset=["alpha", "beta"])
    if not th.empty:
        means = th.assign(posterior_mean=th.alpha / (th.alpha + th.beta)).groupby(["step", "arm"]).posterior_mean.mean().unstack()
        means.plot(); plt.ylabel("Mean Beta posterior mean"); save("thompson_posterior_evolution.png")
    ce = trajectories[trajectories.method == "cem"].dropna(subset=["entropy"])
    if not ce.empty:
        ce.groupby("step").entropy.mean().plot(); plt.ylabel("Mean categorical entropy"); save("cem_entropy.png")
    sm = trajectories[trajectories.method.isin(["smc", "validity_smc"])].dropna(subset=["ess"])
    if not sm.empty:
        for method, group in sm.groupby("method"): group.groupby("step").ess.mean().plot(label=method)
        plt.legend(); plt.ylabel("Mean ESS"); save("smc_ess.png")


def write_reports(output, frame, summary, candidates, trajectories, manifest):
    best_effect = summary.loc[summary.groupby("query_budget").mean_best_drift.idxmax(), ["query_budget", "method", "mean_best_drift"]]
    best_efficiency = summary.loc[summary.groupby("query_budget").mean_drift_per_query.idxmax(), ["query_budget", "method", "mean_drift_per_query"]]
    best_validity = summary.loc[summary.groupby("query_budget").mean_validity_rate.idxmax(), ["query_budget", "method", "mean_validity_rate"]]
    classifications = {}
    overall = summary.groupby("method").agg(drift=("mean_best_drift", "mean"), validity=("mean_validity_rate", "mean"), runtime=("mean_runtime_seconds", "mean"))
    median_drift = overall.drift.median(); median_runtime = overall.runtime.median()
    for method, row in overall.iterrows():
        if row.validity < 0.2: label = "INVALIDITY_BOTTLENECK"
        elif row.runtime > 5 * median_runtime: label = "TOO_EXPENSIVE"
        elif row.drift >= median_drift: label = "PROMISING" if method not in {"random", "mcmc", "evolutionary"} else "WORKING_BASELINE"
        else: label = "UNDERPERFORMS" if method not in {"random", "mcmc", "evolutionary"} else "WORKING_BASELINE"
        classifications[method] = label

    pilot = ["# Pilot Benchmark", "", "This is a matched-budget pilot, not evidence of method superiority.", "",
             f"Grid: {len(manifest['source_ids'])} fixed test sources, budgets {manifest['budgets']}, seeds {manifest['seeds']}.",
             "The nominal target-query budget is the primary match. Actual queries remain explicit; MCMC is not forced to consume rejected-candidate budget.",
             "", "## Highest mean best drift by nominal budget", "", best_effect.to_markdown(index=False),
             "", "## Highest mean drift per actual query", "", best_efficiency.to_markdown(index=False),
             "", "## Highest mean validity rate", "", best_validity.to_markdown(index=False),
             "", "## Pilot classifications", "", "| Method | Classification |", "|---|---|"]
    pilot.extend(f"| {method} | {label} |" for method, label in sorted(classifications.items()))
    pilot += ["", "Classifications are triage labels for the next experiment, not promotion decisions."]
    (ROOT / "docs/PILOT_BENCHMARK.md").write_text("\n".join(pilot) + "\n")

    sets = {method: set(group.best_candidate.dropna()) for method, group in frame.groupby("method")}
    overlap_lines = []
    for i, left in enumerate(sets):
        for right in list(sets)[i + 1:]:
            union = sets[left] | sets[right]
            overlap_lines.append((left, right, len(sets[left] & sets[right]), len(union)))
    length_corr = frame[["length_change", "best_drift"]].dropna().corr().iloc[0, 1]
    validity = summary.pivot(index="query_budget", columns="method", values="mean_validity_rate")
    failure = ["# Pilot Failure Analysis", "", "All findings are descriptive pilot observations.", "",
               "## Failure-case overlap", "", "| Method A | Method B | Shared best candidates | Union |", "|---|---|---:|---:|"]
    failure.extend(f"| {a} | {b} | {shared} | {union} |" for a, b, shared, union in overlap_lines)
    failure += ["", "## Length change", "", f"Pearson correlation between token length change and per-run best drift: `{length_corr:.4f}`. This is an association, not a causal result.",
                "", "## Validity-SMC check", "", validity[[column for column in ["smc", "validity_smc"] if column in validity]].to_markdown(),
                "", "## Adaptive diagnostics", "",
                "Thompson posterior, CEM entropy, and SMC ESS trajectories are reported in the generated diagnostic figures. Interpret collapse only relative to the finite two-family proposal space.",
                "", "## Limitations", "", "- Best-candidate banks support overlap analysis but are not exhaustive candidate banks.",
                "- The pilot uses one frozen checkpoint and three seeds.", "- The same finite operator families can cause candidate overlap.",
                "- Validity filters are necessary checks, not proof of synthesizability or property preservation."]
    (ROOT / "docs/PILOT_FAILURE_ANALYSIS.md").write_text("\n".join(failure) + "\n")


def main():
    args = arguments(); manifest, records = execute(args); write_artifacts(args.output_dir, manifest, records)
    print(f"COMPLETE: {len(records)} runs written to {args.output_dir}")


if __name__ == "__main__": main()
