# Phase 3 Contextual Thompson Audit

Based on the inspection of the existing materials dataset and repository, here is the audit of the components required for Phase 3:

* `docs/CONTEXTUAL_THOMPSON_PREREGISTRATION.md`: EXISTS (needs update/finalization as requested).
* `docs/CONTEXT_FEATURE_LEAKAGE_AUDIT.md`: EXISTS.
* existing Thompson implementation: EXISTS (`materials_adv/attacks/search/strategies.py`).
* Uniform baseline: EXISTS (`UniformSamplingSearch`).
* fixed-arm evaluation: EXISTS (supported within the framework).
* proposal/query budget separation: EXISTS (target-model vs proposal accounting).
* duplicate semantics: EXISTS.
* empirical pseudo-regret tooling: EXISTS (from non-contextual experiments).
* existing processed dataset: EXISTS (processed.csv / canonical benchmark).
* canonical random/scaffold split metadata: PARTIAL (Will be formalized into `context_source_split.json` containing 100+ disjoint sources).

Overall:
All major mathematical and simulation foundations are ready. What is missing is the explicit contextual Bayesian Linear model mapping the extracted features to arm probabilities (`ContextualThompsonSamplingSearch` appears stubbed or needs completion in `update_contextual.py`), the specific Train/Val/Test contextual feature splits, and the evaluation execution loop.
