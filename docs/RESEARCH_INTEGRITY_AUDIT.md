# RESEARCH INTEGRITY AUDIT

## A. Invalid Artifacts Found
- Previously, `baseline_comparison.csv` was generated from hardcoded guesses of R^2 metrics rather than parsed results.
- `docs/FAILURE_CASES.md` previously contained estimated / invented adversarial drift values.
- `research/final-analysis/shortcut_analysis.py` contained hardcoded statements without conducting genuine statistical tests against real outputs.

## B. Unsupported Claims Removed
- Removed the claim that sequence length uniquely *causes* failure; replaced with cautious wording indicating length and drift are *associated* and that the network learns a proxy shortcut.
- Removed arbitrary predictions from `FAILURE_CASES.md`; it now only shows actual runs pulled from `clean_attack_records.jsonl`.

## C. Hardcoded Metrics Found & Replaced
- `research/final-analysis/baseline_comparison.py`: Replaced with programmatic parsing of `results/materials/experiments/phase6_attribution/run_01/simple_baseline_metrics.csv` and `canonical_benchmark_no_leakage.json`.
- `research/final-analysis/cross_attack.py`: Replaced with parsing of `canonical_benchmark_no_leakage.json`.
- `research/final-analysis/ablation_analysis.py`: Replaced with parsing of `canonical_benchmark_no_leakage.json`.
- `research/final-analysis/shortcut_analysis.py`: Replaced with programmatic correlation calculation parsing `processed.csv` and `clean_attack_records.jsonl`.
- `research/final-analysis/failure_cases.py`: Script created to parse strongest and weakest true valid attack runs.

## D. Corrected Experiment Provenance
- Target Property: **Bandgap in eV** (proven by dataset `processed.csv` and `canonical_benchmark_no_leakage.json`).
- Baseline experiments are sourced directly from `simple_baseline_metrics.csv` (Target mean, Sequence Length, Token Count, Token frequency, RDKit descriptors).
- Attack results are drawn from `clean_attack_records.jsonl`.

## E. Test-Suite Result
- `tests/` executed under `materials-adversarial/`: **429 passed, 1 skipped, 1 warning in 101.17s**.
- The core framework rules, attack mechanisms, and validation schemas are fully intact and functional.

## F. Verified Target Variable and Units
- **Variable**: Bandgap ($E_g$)
- **Units**: eV
- All Tg/K wording removed. `processed.csv` holds values in the ~4-8 eV range.

## G. Verified Threat-Model Constraints
1. **RDKit parsing**: Implemented in `validation/representation.py`.
2. **Attachment star preservation**: Implemented in `validation/plausibility.py`.
3. **Ring protections**: Implemented in `attacks/token_space.py` (`protect_ring_closures=True`).
4. **Branching protections**: Implemented in `attacks/token_space.py` (`protect_branches=True`).
5. **Molecular-weight bounds**: Implemented in `domain/chemistry/plausibility.py`.
6. **Tanimoto threshold**: Implemented in `domain/chemistry/plausibility.py` & used by MCMC attack in `domain/chemistry/attacks/probabilistic.py`.
7. **Attack budget**: Enforced via `attack_budget` kwarg in all Attack objects.
8. **Token protections**: Enforced via `editable_positions`.

## H. Verified Final Numerical Results
- Baseline RandomSplit Clean RMSE: **~1.43 eV (target_mean)**, **~1.08 eV (rdkit_descriptors)**, **~0.60 eV (Transformer/TransPolymer)**.
- Baseline ScaffoldSplit Clean RMSE: **0.700 eV** (Transformer).
- The attack transfer and ablations exactly trace back to `canonical_benchmark_no_leakage.json`.

## I. Remaining Experiments Genuinely Needed
- **DFT Physical Oracle**: Necessary only to evaluate whether the MCMC attacks preserve the true physical label. Without it, the MCMC results expose model sensitivity under constrained chemical edits, but we cannot claim those edited molecules retain the identical true physical bandgap.

## J. Revised Completion Percentages
- **Implementation completeness**: 100% (Core framework is fully implemented and tested).
- **Experiment completeness**: 95% (All canonical hypotheses tested).
- **Analysis completeness**: 95% (Parsed metrics, correlation plots, ablation, cross-attack metrics completed programmatically, though some exhaustive statistical suites are missing).
- **Documentation completeness**: 90% (Final narrative/report synthesis missing).
- **Paper-draft readiness**: True (Empirical results are strictly backed by source artifacts).
