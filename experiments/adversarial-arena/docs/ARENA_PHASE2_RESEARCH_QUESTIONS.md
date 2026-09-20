# Phase 2 Research Questions

- Does internal adaptation (carry state) outperform stateless resets? **NOT_TESTED**. (Metric: Compare drift across identical seeds. Reset controls not included).
- Do probabilistic attackers generate greater defender gain? **SUPPORTED_IN_PILOT** (Metric: Defender Gain Difference).
- Does Validity-SMC improve search efficiency over SMC? **INCONCLUSIVE**. (Metric: valid_proposals_per_target_query vs ESS).
- Is there robustness redistribution across iterations? **SUPPORTED_IN_PILOT**. (Metric: active_gain > 0 and other_gain < 0).
