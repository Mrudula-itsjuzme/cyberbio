# Negative and Null Results

1. **CEM Degradation**: Seed 20260920 R1 and R2 yielded *negative* G_carry, indicating that carry-state actively harmed search compared to reset-state.
2. **SMC == Validity-SMC Equivalence**: Implementation inspection reveals that ValiditySMC overrides the uniform family selection with validity_posterior ONLY in the resampling weight `score`, NOT in the candidate generation itself. Since `query_budget=25` is small, the loop deterministically iterates through the parent particles without triggering `resampled`. Thus, the carried `particles` remain exactly identical across SMC and Validity-SMC across all seeds and rounds.
3. **Query Mismatches**: 12/27 events experienced query mismatches where the attacker overshot or undershot budgets relative to the reset baseline.
