# Explainability Analysis

**Artifacts**:
- `comparison-experiments/results/explainability/attribution_alignment.csv`
- `comparison-experiments/results/explainability/mcmc_occlusion_summary.csv`
- `comparison-experiments/results/raw_explainability/mcmc_occlusion.csv`

**Findings**:
We ran occlusion on 20 MCMC sources. Dynamic programming sequence alignment (Needleman-Wunsch for tokens) matched token changes to attribution scores.
- Adversarial edits systematically target high-attribution regions.
