# Phase 2 Research Questions

**RQ: Does internal adaptation (carry state) outperform stateless resets?**
- **VERDICT**: NOT_TESTED
- **METRIC**: N/A
- **NUMERIC EVIDENCE**: N/A
- **LIMITATION**: Reset controls were not executed in this milestone.

**RQ: Do probabilistic attackers generate greater defender gain?**
- **VERDICT**: MIXED
- **METRIC**: Defender Gain Difference
- **NUMERIC EVIDENCE**: CEM (mean=0.038, std=0.192), SMC (mean=0.505, std=1.217), Validity-SMC (mean=0.703, std=1.424).
- **LIMITATION**: High variance makes mean differences statistically unreliable across n=3 bounds. Positive `G_adapt` does NOT directly imply defender gain.

**RQ: Does Validity-SMC improve search efficiency over SMC?**
- **VERDICT**: MIXED
- **METRIC**: valid_per_target_query and drift_per_target_query difference
- **NUMERIC EVIDENCE**: `valid_per_target_query` diff mean=0.2616 (std=0.2447). `drift_per_target_query` diff mean=-0.0698 (std=0.1065).
- **LIMITATION**: Validity constraints produced more valid outputs per query but degraded per-query drift capabilities.

**RQ: Is there robustness redistribution across iterations?**
- **VERDICT**: SUPPORTED_IN_PILOT
- **METRIC**: active_gain_difference > 0 AND other_gain_difference < 0 (via Evolutionary)
- **NUMERIC EVIDENCE**: 1 explicit iteration measured natively under CEM execution (total 1 across all 27).
- **LIMITATION**: Extremely rare relative to overall trajectory sizes.
