# Shortcut Audit

**Artifacts**:
- `comparison-experiments/results/shortcut_audit/shortcut_baselines.json`
- `comparison-experiments/results/shortcut_audit/shortcut_baselines.csv`

**Findings**:
We executed a sequence length and N-gram regression baseline audit against `property_value`.
- Sequence Length Baseline R²: ~0.17
- Token Counts R²: ~0.60
- 2-gram Counts R²: ~0.84
- 3-gram Counts R²: ~0.90
- Transformer R²: ~0.92

The bag-of-tokens and N-gram baselines are highly predictive, demonstrating that representation properties account for the majority of target variance.
