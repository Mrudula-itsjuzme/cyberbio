"""Phase 6 equal-query-budget attack efficiency benchmark."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..attacks.search import (
    BeamSearch, EvolutionarySearch, GreedySearch, MetropolisSearch, RandomSearch,
)
from ..evaluation.representation_attribution import paired_bootstrap_ci


@dataclass(frozen=True)
class AttackEfficiencySettings:
    seed: int = 20260910
    query_budget: int = 30
    max_changes: int = 3
    require_plausible: bool = True
    beam_width: int = 4
    branching_factor: int = 3
    temperature: float = 0.1
    population_size: int = 8
    elite_size: int = 3
    max_proposal_multiplier: int = 200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AttackEfficiencyBenchmark:
    STRATEGIES = (RandomSearch, GreedySearch, BeamSearch, MetropolisSearch, EvolutionarySearch)

    def __init__(self, settings: AttackEfficiencySettings):
        self.settings = settings

    def run(self, *, samples: Sequence[tuple[str, str]], predictor,
            proposal_factory: Callable[[np.random.Generator], object], output_dir: Path,
            checkpoint_path: Path, metadata: dict, config_snapshot: dict) -> dict:
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite Phase 6 output: {output_dir}")
        output_dir.mkdir(parents=True)
        summaries, traces, curves = [], [], []
        for sample_number, (sample_id, representation) in enumerate(samples):
            for strategy_number, strategy_class in enumerate(self.STRATEGIES):
                # Proposal randomness is paired by sample. Policy randomness is separate.
                proposal_seed = self.settings.seed + sample_number * 1009
                policy_seed = self.settings.seed + sample_number * 1009 + strategy_number + 1
                common = dict(query_budget=self.settings.query_budget,
                              require_plausible=self.settings.require_plausible,
                              max_changes=self.settings.max_changes,
                              max_proposal_multiplier=self.settings.max_proposal_multiplier)
                extra = {}
                if strategy_class is BeamSearch:
                    extra = {"beam_width": self.settings.beam_width,
                             "branching_factor": self.settings.branching_factor}
                elif strategy_class is MetropolisSearch:
                    extra = {"temperature": self.settings.temperature}
                elif strategy_class is EvolutionarySearch:
                    extra = {"population_size": self.settings.population_size,
                             "elite_size": self.settings.elite_size}
                search = strategy_class(predictor,
                    proposal_factory(np.random.default_rng(proposal_seed)),
                    np.random.default_rng(policy_seed), **common, **extra)
                result = search.search(representation)
                summaries.append({"sample_id": sample_id, "proposal_seed": proposal_seed,
                                  "policy_seed": policy_seed, **result.summary()})
                best_so_far, query_number = 0.0, 0
                curves.append({"sample_id": sample_id, "strategy": result.strategy,
                               "query_number": 0, "best_drift": 0.0})
                for entry in result.trace:
                    traces.append({"sample_id": sample_id, "strategy": result.strategy, **entry})
                    if entry["queried"]:
                        query_number += 1
                        best_so_far = max(best_so_far, float(entry["absolute_drift"]))
                        curves.append({"sample_id": sample_id, "strategy": result.strategy,
                                       "query_number": query_number, "best_drift": best_so_far})

        per_example = pd.DataFrame(summaries)
        per_example.to_csv(output_dir / "per_example_results.csv", index=False)
        pd.DataFrame(curves).to_csv(output_dir / "query_efficiency_curves.csv", index=False)
        with (output_dir / "proposal_trace.jsonl").open("w") as handle:
            for trace in traces: handle.write(json.dumps(trace) + "\n")

        rows = []
        for strategy, group in per_example.groupby("strategy", sort=False):
            acceptance = group.acceptance_rate.dropna()
            rows.append({
                "strategy": strategy, "n_examples": len(group),
                "query_budget": self.settings.query_budget,
                "mean_candidate_queries": group.candidate_model_queries.mean(),
                "budget_exhaustion_rate": group.query_budget_exhausted.mean(),
                "mean_candidate_proposals": group.candidate_proposals.mean(),
                "representation_validity_rate": (group.representation_valid_candidates.sum() /
                                                 group.candidate_proposals.sum()),
                "eligible_validity_rate": group.valid_candidates.sum() / group.candidate_proposals.sum(),
                "mean_acceptance_rate": acceptance.mean() if len(acceptance) else np.nan,
                "mean_best_drift": group.best_drift.mean(),
                "median_best_drift": group.best_drift.median(),
                "mean_drift_per_query": group.drift_per_model_query.mean(),
                "mean_elapsed_seconds": group.elapsed_seconds.mean(),
                "mean_perturbation_size": group.perturbation_size.mean(),
            })
        matrix = pd.DataFrame(rows)
        matrix.to_csv(output_dir / "attack_efficiency.csv", index=False)
        paired = per_example.pivot(index="sample_id", columns="strategy", values="best_drift")
        paired_vs_random = {}
        for strategy in paired.columns:
            if strategy == "random":
                continue
            delta = (paired[strategy] - paired["random"]).dropna().to_numpy()
            paired_vs_random[strategy] = paired_bootstrap_ci(
                delta, rng=np.random.default_rng(self.settings.seed + 99991),
                n_resamples=2000, minimum_n=20)
        equal_budget = bool((per_example.candidate_query_budget == self.settings.query_budget).all())
        summary = {
            "status": "observed", "n_examples": len(samples),
            "configured_equal_candidate_query_budget": equal_budget,
            "query_budget": self.settings.query_budget,
            "all_runs_exhausted_budget": bool(per_example.query_budget_exhausted.all()),
            "equal_realized_budget_comparison_valid": bool(per_example.query_budget_exhausted.all()),
            "search_strategies": matrix.replace({np.nan: None}).to_dict("records"),
            "paired_best_drift_delta_vs_random": paired_vs_random,
            "scientific_guardrails": {
                "predictor_access": "black_box_PredictorProtocol_only",
                "query_definition": "one candidate prediction; original prediction recorded separately",
                "proposal_semantics": "local chemistry-changing stress-test edits; labels not preserved",
                "metropolis": "Metropolis-style acceptance heuristic, not formal MH because proposal correction is absent",
                "ranking": "compare best drift and drift/query only when realized budgets and validity constraints align",
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        import rdkit, torch
        repro = {**metadata, "settings": self.settings.__dict__,
                 "checkpoint_sha256": _sha256(checkpoint_path),
                 "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "rdkit": rdkit.__version__,
                             "torch": torch.__version__}}
        (output_dir / "reproducibility.json").write_text(json.dumps(repro, indent=2) + "\n")
        (output_dir / "config_snapshot.json").write_text(json.dumps(config_snapshot, indent=2) + "\n")
        return summary
