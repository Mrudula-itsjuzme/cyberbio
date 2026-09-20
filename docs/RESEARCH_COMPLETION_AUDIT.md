# RESEARCH COMPLETION AUDIT

## 1. What is already implemented
- A full adversarial framework for evaluating sequence models on polymer bandgap prediction (eV).
- Transformer Baseline Model (TransPolymer).
- Data splits (Random, Scaffold).
- Two core attack types:
  - Random SMILES (representation-preserving)
  - MCMC (chemistry-changing)
- Defense strategies:
  - Randomized SMILES Augmentation
  - Label-Free Consistency Regularization for MCMC
- Comprehensive Benchmark Suite evaluating 3 seeds across both Random and Scaffold splits.
- Experimental result aggregation (`canonical_benchmark_no_leakage.json`).

## 2. What actually runs
- Most tests pass (429 passing tests according to README).
- Complete benchmark suite (`run_comprehensive_benchmark_suite.py`).
- Publication plotting scripts.
- Reproducibility audit scripts.

## 3. What is partially implemented
- RL environments, Generative models, Evolutionary/Search attacks (mostly mock/archived).

## 4. What is broken
- LLM Attacks (blocked by credentials).
- Physical Oracle (DFT) validations (blocked).
- Possibly evolutionary search (needs validation).

## 5. What experiments already have valid results
- Baseline performance for clean model on both Random and Scaffold splits.
- Adversarial drift for Random SMILES on Baseline vs Defended models.
- Adversarial drift for MCMC attacks on Baseline vs Defended models.
- All values verified across 3 seeds in `canonical_benchmark_no_leakage.json`.

## 6. What claims are currently supported by evidence
- "Chemically identical PSMILES serializations produce substantial prediction variation." (Evidenced by high Random-SMILES drift).
- "Representation-preserving multi-SMILES augmentation drastically reduces this vulnerability while improving structural out-of-distribution generalization." (Evidenced by reduced drift and lower RMSE on Scaffold split).
- "Constrained MCMC exposes local model sensitivity... consistency training can reduce that sensitivity." (Evidenced by MCMC drift reduction).

## 7. What claims are NOT yet adequately supported
- LLM and Generative attacks working.
- That MCMC generated sequences reflect true properties (needs DFT oracle).

## 8. Exact remaining tasks required for a defensible research submission
- PHASE 2: Lightweight verification (smoke test).
- PHASE 3: Produce `docs/THREAT_MODEL.md` after verifying chemical constraints.
- PHASE 4: Extract/Format Defense Transfer Validation to `results/final/`
- PHASE 5: Format/Run Research Ablations into `results/final/ablation_results.csv` and `docs/ABLATION_ANALYSIS.md`.
- PHASE 6: Investigate the Length Shortcut properly (run scripts, plots to `results/final/figures/`, `docs/SHORTCUT_ANALYSIS.md`).
- PHASE 7: Re-confirm statistical robustness (already partially in canonical results).
- PHASE 8: Baseline Comparison (`results/final/baseline_comparison.csv`).
- PHASE 9: Cross-Attack Generalization Matrix.
- PHASE 10: Failure Case Analysis (`docs/FAILURE_CASES.md`).
- PHASE 11: Final Research Figures.
- PHASE 12: Final Claims Table (`docs/CLAIMS_AND_EVIDENCE.md`).
- PHASE 13: Finalize Research Report.
- PHASE 14: README Cleanup.
- PHASE 15: Single Orchestration Script (`scripts/run_final_experiments.py`).
- PHASE 16: Ensure tests pass.
- PHASE 17: Final Completion Report.
