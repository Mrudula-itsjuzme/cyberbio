# Explainability Analysis

**Artifacts**:
- `results/comparison/experiments/explainability/attribution_alignment.csv`
- `results/comparison/experiments/explainability/mcmc_occlusion_summary.csv`
- `results/comparison/experiments/raw_explainability/mcmc_occlusion.csv`

**Findings**:
We ran occlusion on 20 MCMC sources. Dynamic programming sequence alignment (Needleman-Wunsch for tokens) matched token changes to attribution scores.
- Adversarial edits systematically target high-attribution regions.
