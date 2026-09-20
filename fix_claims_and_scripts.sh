#!/bin/bash
# 1. Update docs/CLAIMS_AND_EVIDENCE.md to remove 95% CI claims for the ablation table, and change "5 seeds" to "1 seed"
sed -i 's/Uncertainty: 95% CI reported in original artifacts/Uncertainty: Point estimate (Single deterministic seed)/g' docs/CLAIMS_AND_EVIDENCE.md
sed -i 's/Number of seeds: 5/Number of seeds: 1 (deterministic seed 42)/g' docs/CLAIMS_AND_EVIDENCE.md
# Update ablation claim wording
sed -i 's/CombinedDefense strategy provides the most consistent improvement/CombinedDefense strategy performs best across the evaluated metrics\/configurations/g' docs/ABLATION_ANALYSIS.md

# 2. Update shortcut wording
sed -i 's/causing it to be sensitive to/consistent with shortcut reliance, creating sensitivity to/g' docs/SHORTCUT_ANALYSIS.md

# 3. Update cross_attack.py to use single seed logic
sed -i 's/"seed_aggregation": "mean_over_5_seeds"/"seed_aggregation": "single_seed_42"/g' cross_attack.py

# 4. Rerun cross attack
source materials-adversarial/.venv/bin/activate 2>/dev/null || true
python cross_attack.py
