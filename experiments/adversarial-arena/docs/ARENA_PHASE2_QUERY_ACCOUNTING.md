# Phase 2 Query Accounting Protocol

## Central Rule
Exactly 25 target model queries evaluate candidates per source.
The clean reference source evaluation does NOT decrement the attack budget.

## Expanded Accounting Schema
We have updated the CandidateRecord to explicitly distinguish the distinct components of the total target model evaluations incurred across the pipeline. This resolves the previous ambiguity surrounding `target_query_count`:

1. **`clean_reference_queries`**: Always 1 per source. The target model is evaluated on the unmodified source sequence to establish the baseline prediction. (Cached and shared across generated candidates for that source).
2. **`attack_target_queries`**: The actual queries consumed by the attack algorithm (e.g. CEM, SMC) to score internal proposals against the target model.
3. **`budgeted_attack_queries`**: The strictly enforced limit for `attack_target_queries` (e.g., 25).
4. **`posthoc_scoring_queries`**: Evaluations triggered dynamically by `ArenaRunner._score` at the orchestrator level after an attacker proposes a batch. For every valid proposed candidate, it re-queries the model on `[original_input, candidate_input]`, consuming 2 queries per valid candidate.
5. **`actual_defender_calls`**: The exact sum representing total network/model invocations:
   `actual_defender_calls = clean_reference_queries + attack_target_queries + posthoc_scoring_queries`

By structuring the metadata this way, the Phase 2 experiments natively reconcile exactly how the locked attack budget produces the final metrics, without altering execution semantics or invalidating the concurrent Phase 2 background runs.
