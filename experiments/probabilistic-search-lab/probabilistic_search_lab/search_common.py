from __future__ import annotations

import time

from .accounting import CountedTargetModel, QueryLedger
from .contracts import MethodStatus, SearchMetrics, SearchResult
from .metrics import summarize_effects


def finalize(*, method, seed, source_id, budget, ledger, started, effects,
             valid_candidates, invalid_count, best_candidate, trajectories=None,
             status=MethodStatus.WORKS, failure_status=None, metadata=None):
    summary = summarize_effects(effects)
    valid_count = len(effects)
    return SearchResult(
        method=method, status=status, seed=seed, source_id=str(source_id),
        query_budget=budget,
        metrics=SearchMetrics(
            proposal_count=ledger.proposal_count,
            validity_checks=ledger.validity_checks,
            valid_count=valid_count,
            invalid_count=invalid_count,
            target_model_queries=ledger.target_model_queries,
            attribution_queries=ledger.attribution_queries,
            surrogate_queries=ledger.surrogate_queries,
            runtime_seconds=time.perf_counter() - started,
            mean_effect=summary["mean"], median_effect=summary["median"],
            p90_effect=summary["p90"], best_effect=summary["best"],
            diversity_score=(len(valid_candidates) / valid_count) if valid_count else None,
        ),
        best_candidate=best_candidate, failure_status=failure_status,
        trajectories=trajectories or {}, metadata=metadata or {},
    )


def begin(model, budget):
    ledger = QueryLedger(budget=budget)
    return ledger, CountedTargetModel(model, ledger), time.perf_counter()
