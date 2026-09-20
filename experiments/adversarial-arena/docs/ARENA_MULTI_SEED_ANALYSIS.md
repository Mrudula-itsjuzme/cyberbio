# Arena Multi-Seed Statistical Analysis

## 1. Why Multiple Seeds Matter
Adversarial learning landscapes are heavily non-convex, and Thompson sampling is inherently stochastic. A single trace may show a defender improving while an attack fails, but this could be a stochastic artifact of the random initialization rather than a robust algorithmic property. Multiple seeds isolate robust trends from statistical noise.

## 2. Seed Configuration
- Configuration identical to the preregistered pilot (seed `20260920`).
- Seeds used: `20260920`, `20260921`, `20260922`.
- Query budget: 25.
- Fixed 100 training and 10 validation sources.

## 3. Valid Seed Count
All **n = 3 seeds** successfully completed the 3-round curriculum. No rounds failed silently.

## 4. Aggregate Results (n = 3 seeds)
- **Round 0 Mean Drift**: 0.359 ± 0.081
- **Round 1 Mean Drift**: 0.734 ± 0.292
- **Round 2 Mean Drift**: 0.539 ± 0.170

## 5. Variance Across Seeds
Variance was significant, especially in Round 1 where the attacker choices diverged. High variance indicates that the specific adversarial samples discovered heavily dictated the subsequent trajectory of the defender.

## 6. Defender Gain
The defender consistently gained robustness against the directly trained attackers.
- Round 0 Defender Gain: 0.111 ± 0.098
- Round 1 Defender Gain: 0.442 ± 0.201

## 7. Held-out Transfer
Transfer to the held-out Evolutionary attacker was slightly negative.
- Round 0: -0.015 ± 0.051
- Round 1: -0.031 ± 0.091
This implies that robustness improvements on RandomSMILES and MCMC did not generalize to Evolutionary edits, and in some seeds, made the defender more vulnerable to them.

## 8. Adaptive Selection Gain
The Thompson controller selects a new attacker family each round. The Adaptive Selection Gain ($G_{select}$) measures if the newly selected family exposes more vulnerability against the same defender checkpoint.
- Round 0 $\rightarrow$ 1: 0.388 ± 0.336 (Positive: the controller successfully found a harder attacker)
- Round 1 $\rightarrow$ 2: -0.296 ± 0.409 (Negative: the next attacker was weaker)

## 9. Robustness Redistribution
We analyzed the hypothesis that improving against one attacker increases vulnerability to another. Across the 3 seeds, several instances of redistribution were observed, predominantly where training on MCMC caused degradation against the held-out Evolutionary attacker. This highlights non-trivial trade-offs in adversarial robustness.

## 10. Clean-Performance Tradeoff
Defender updates exhibited mixed tradeoffs. In some cases, robustness wins came at a clean-RMSE cost (`ROBUST_WIN_CLEAN_COST`). In others, clean performance actually improved alongside robustness (`ROBUST_WIN_CLEAN_WIN`), suggesting a regularizing effect of the hard examples.

## 11. Controller Stability
The Thompson controller's posterior parameters ($\alpha, \beta$) showed substantial instability across seeds. Different seeds collapsed onto different preferred attackers (e.g., heavily favoring MCMC in seed 20260921, but RandomSMILES in 20260922), indicating that the local adversarial examples strongly skewed the curriculum.

## 12. Early Trajectory Consistency
There is a **mixed trajectory** across the 3 seeds. While all seeds demonstrate a general pattern of early defender gain against the chosen attacker, the cross-attack transfer and clean-RMSE dynamics oscillate without a clear consistent trend at $t=3$.

## 13. Negative Results
- **Held-out Transfer**: Failed to consistently transfer robustness to the Evolutionary attacker.
- **Controller Convergence**: The Thompson controller did not agree on a universally hardest attacker across seeds, suggesting the "hardest" attacker is relative to the defender's current local minimum.

## 14. Limitations
With **n = 3 seeds**, these results represent early pilot evidence, not large-sample certainty. Confidence intervals are intentionally omitted, as 3 samples cannot reliably estimate population tails. Three rounds are insufficient to claim long-term convergence.
