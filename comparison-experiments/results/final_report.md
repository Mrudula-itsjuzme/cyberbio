# Comparison Experiments - End Report

## Environment Used
- **Project Root**: `/home/mrudula/Downloads/DL_cyberbio/comparison-experiments`
- **Canonical Model Project**: `/home/mrudula/Downloads/DL_cyberbio/materials-adversarial`
- **Python Environment**: `/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/.venv/bin/python` (Python 3.12, PyTorch CPU, RDKit). Reused the existing venv to ensure exact chemical parsing compatibility and dependency alignment with the canonical research project.

## Files Modified/Created
- `src/adapters/canonical_model.py`: Updated to properly utilize `materials_adv.framework.predictors.GraphMPNNPredictor` to predict natively unscaled target values.
- `src/adapters/canonical_dataset.py`: Created to pull directly from the original `processed.csv` mapping via `splits.json`.
- `src/adapters/canonical_validator.py`: Fixed `is_valid_rdkit` logic to correctly parse RDKit Mol objects rather than throwing strings at the checker.
- `src/adapters/canonical_mcmc.py`: Fixed the signature and parameters passed to `ProbabilisticMCMCAttack.generate` (wrapped inputs correctly using `Candidate` and `Vocabulary` loaded from `vocab.json`).
- `src/attackers/random_search.py` & `src/attackers/evolutionary.py`: Fixed operator application logic by wrapping incoming strings into `Candidate` objects before executing attacks. 
- `src/metrics/statistics.py`: Created helper standardizing `Mean/Median/P90` drift and computing runtimes & query budgets.
- `scripts/build_source_bank.py`: Deterministically extracted 100 sealed test IDs (`frozen_source_ids.json`).
- `scripts/run_comparison.py`: Connected budget components, models, validators, and output serialization into an end-to-end framework. 
- `scripts/generate_plots.py`: Reads the JSONL logs to compute plots comparing drift, runtime, and validity.
- `tests/*`: Wrote tests (`test_canonical_model.py`, `test_budgets.py`, `test_adapters.py`, `test_schemas.py`, `test_random_search.py`, `test_llm_mock.py`).

## Tests Run
- Successfully executed 7 Pytest tests validating adapter sanity, random search generation, mock LLM generation, budget enforcement limits (queries, generations, runtime), and data serialization.

## Verified Inference & Validation
- Integrated the original checkpoints (`model.pt`, `scaler.pkl`) smoothly. The models natively predict values without manual inverse transformation when using `GraphMPNNPredictor(strict=True)`.
- RDKit correctly flags constraints based on standard parsing routines via `materials_adv.domain.chemistry.rdkit_adapter.RDKitValidityChecker`.

## Real Attack Runs
- Executed `random_search`, `evolutionary_search`, and `canonical_mcmc` over 20 source sequences with a query budget of 20, seeded on `42`, `123`, and `2026`.
- **Results Snapshot**:
  - `random_search` / `evolutionary_search`: Produced 0 valid constraints passed in early generations because the simple combinatorial operators generated invalid chemical structures (failing RDKit parsing limits like "Explicit valence greater than permitted"). The budget successfully stopped them from infinite loops.
  - `canonical_mcmc`: Completed an average of ~10 model queries per run, yielding `Mean Drift ~0.47` for seed 123. The MCMC candidates successfully bypassed the RDKit checker failures observed in random/evolutionary searches, though success rates against full plausibility checks required more budget to trigger definitively.

## Result Paths
- **Logs**: `/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/raw/raw_*.jsonl`
- **Plots**: `/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/plots/`
  - `drift_distribution.png`
  - `validity_rate.png`
  - `drift_vs_similarity.png`
  - `runtime_vs_drift.png`
- **Benchmark Summaries**: `/home/mrudula/Downloads/DL_cyberbio/comparison-experiments/results/summaries/benchmark_*.txt`

## Scientific Limitations
1. **Plausibility Penalties**: Because the MCMC loop explicitly tracks chemical validity (including Tanimoto similarities), it yields candidates that survive basic structural validation. Random and Evolutionary currently apply operators "blindly," heavily hitting explicit valence errors, which terminates them early under a tight budget.
2. **Budget Thresholds**: A 20-query budget limit restricts large population evaluations (e.g., Evolutionary requires larger populations across dozens of generations to optimize meaningfully).
3. **MCMC Constraint Alignment**: MCMC results reported 0% final constraint pass despite scoring positive in structural validity inside the MCMC logic, indicating that either `canonical_validator` applies a stricter final post-process plausibility check, or the `min_tanimoto` bound (0.5) forced them to fail the similarity check downstream. 
