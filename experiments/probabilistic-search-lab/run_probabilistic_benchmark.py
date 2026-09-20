#!/usr/bin/env python3
"""Shared entry point for the non-canonical probabilistic search lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB_ROOT))

from probabilistic_search_lab.contracts import MethodStatus, SearchResult

METHODS = (
    "random", "mcmc", "evolutionary", "attribution_prior", "attribution_smc",
    "generative", "smc", "validity_smc", "cem", "thompson", "bayesopt",
    "validity_bo", "uncertainty", "gflownet",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-sources", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--split", default="test")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path)
    return p


def unavailable(args, status: MethodStatus, reason: str) -> SearchResult:
    return SearchResult(
        method=args.method,
        status=status,
        seed=args.seed,
        source_id="__benchmark__",
        query_budget=args.budget,
        failure_status=reason,
        metadata={
            "num_sources_requested": args.num_sources,
            "device": args.device,
            "split": args.split,
            "smoke": args.smoke,
        },
    )


def run(args) -> list[SearchResult]:
    if args.budget < 1 or args.num_sources < 1:
        raise ValueError("--budget and --num-sources must be positive")
    if args.method in {"bayesopt", "validity_bo", "uncertainty"}:
        return [unavailable(args, MethodStatus.BLOCKED_DEPENDENCY, "dependency and representation feasibility audit pending")]
    if args.method == "gflownet":
        return [unavailable(args, MethodStatus.BLOCKED_COMPUTE, "feasibility spike not yet performed")]
    if args.method in {"attribution_prior", "attribution_smc", "generative"}:
        return [unavailable(args, MethodStatus.NOT_IMPLEMENTED, "canonical adapter not implemented in this milestone")]

    from baselines import EvolutionarySearch, MCMCSearch, RandomSearch
    from cem import CEMSearch
    from probabilistic_search_lab.canonical import CanonicalValidator, load_model, load_sources
    from probabilistic_search_lab.proposals import canonical_families
    from smc import SMCSearch
    from thompson import ThompsonSearch
    from validity_smc import EmpiricalValidityModel

    families = canonical_families()
    constructors = {
        "random": lambda: RandomSearch(families),
        "mcmc": MCMCSearch,
        "evolutionary": lambda: EvolutionarySearch(families),
        "thompson": lambda: ThompsonSearch(families),
        "cem": lambda: CEMSearch(families),
        "smc": lambda: SMCSearch(families),
        "validity_smc": lambda: SMCSearch(families, validity_model=EmpiricalValidityModel()),
    }
    model = load_model(args.device)
    validator = CanonicalValidator()
    sources = load_sources(args.split, args.num_sources)
    results = []
    for offset, (source_id, source) in enumerate(sources):
        search = constructors[args.method]()
        results.append(search.run(
            source_id=source_id, source=source, model=model, validator=validator,
            budget=args.budget, seed=args.seed + offset,
        ))
    return results


def main() -> int:
    args = parser().parse_args()
    results = run(args)
    payload = json.dumps([result.to_dict() for result in results], indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if all(result.status is MethodStatus.WORKS for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
