# VALIDITY BOUNDARY INVALIDATION

**Status:** `INVALIDATED_SYNTHETIC_EXPERIMENT`

**Reason:** 
Results were generated from simulated/hardcoded values rather than actual dataset candidates, validator outputs, and target-model predictions. The script fabricated source IDs, synthetic drift values, hardcoded correlations and p-values, hardcoded ablation tables, created dummy CSV rows, and wrote "mock png" text into image files.

**Preserved Invalid Artifacts (Moved to `results/validity_boundary/INVALID_SYNTHETIC_RUN/`):**
- `scripts/run_validity_boundary_experiment.py`
- `candidate_boundary_dataset.csv`
- `boundary_associations.csv`
- `search_results.csv`
- `boundary_ablation.csv`
- `query_efficiency.csv`
- `representation_control.csv`
- `failure_clusters.csv`
- `docs/VALIDITY_BOUNDARY_ARENA_GATE.md`
- `docs/VALIDITY_BOUNDARY_SEARCH.md`
- 10 `.png` files containing dummy text.

No values from these artifacts shall be cited in future reports.
