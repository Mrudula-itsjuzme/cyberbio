# Arena Viva – Questions and Answers

## What is the Arena?
A scientific pipeline that stress-tests a trained materials property predictor (TransPolymer / Transformer Regressor) by iteratively generating adversarial inputs, fine-tuning the defender, and re-evaluating robustness.

## Why use real TransPolymer?
Ridge smoke is not scientifically valid evidence. TransPolymer is the canonical architecture used in the literature. Only evaluation against it has external validity.

## Why multiple seeds?
Stochasticity exists in attacker sampling (MCMC, RandomSMILES), Thompson reward draws, and hard-example selection. A single seed cannot distinguish systematic signal from random fluctuation. Three seeds give a first estimate of variance.

## Why isn't one deterministic run enough?
Even a deterministic run using a fixed seed depends on initialization. If results are strong on seed 20260920 but do not replicate on 20260921/22, the result is not robust. Standard deviation quantifies how much the finding depends on initialization.

## What does standard deviation tell us?
With n=3 seeds it estimates within-pilot variability. It does NOT establish population statistics. All reported values should be interpreted as mean ± std (n=3) with the explicit limitation that 3 seeds is insufficient for authoritative confidence intervals.

## What is Adaptive Selection Gain (G_select)?
G_select(t) = R(A_{t+1}, D_t) − R(A_t, D_t)

This measures how much stronger the NEXT attacker family selected by the Thompson controller is against the SAME defender checkpoint, relative to the PREVIOUS selected family. It is a curriculum-level quantity, not a within-attacker quantity.

## How is Adaptive Selection Gain different from Attacker Adaptation?
Adaptive Selection Gain compares two DIFFERENT attacker families (e.g., RandomSMILES vs MCMC) against the same defender. The controller switches between pre-existing families; their internal policies do not change.

True Attacker Adaptation would require a single attacker whose internal distribution or policy updates over time (e.g., CEM updating its proposal covariance, a learned generator fine-tuning, or a contextual bandit updating its selection policy). This capability does not yet exist in the current Arena.

## What is Held-Out Transfer?
T_Evo(t) = R(Evo, D_t) − R(Evo, D_{t+1})

Evolutionary is excluded from Thompson training selection. Each round, after the defender updates from RandomSMILES/MCMC examples, we evaluate Evolutionary against both the old (D_t) and new (D_{t+1}) defender. A positive value means the defender became less vulnerable to Evolutionary, even though Evolutionary was never in the training pool.

## What is Robustness Redistribution?
An event where training on one attacker reduces the defender's vulnerability to that attacker but simultaneously increases vulnerability to a different attacker. Formally: active_gain > 0 AND other_gain < 0.

## Does redistribution mean catastrophic forgetting?
Not automatically. Redistribution is an empirical observation about directional changes in vulnerability metrics. Catastrophic forgetting is a specific mechanism (overwriting of learned weights). Redistribution could arise from capacity constraints, gradient interference, or from the two attack families exploiting correlated vulnerabilities. Three rounds with two attackers is insufficient evidence to conclude catastrophic forgetting.

## Why can't three rounds establish convergence?
Convergence requires the metric to stop changing meaningfully. Three rounds show an early trend (increasing, decreasing, oscillating) but cannot rule out that the trend would reverse at round 4 or 10. We report "early trajectory consistent with X" not "convergence to X."

## Why keep individual seed traces?
Averaging conceals disagreement. If seeds 20260920 and 20260921 show gain while 20260922 shows loss, the mean shows near-zero gain, hiding the contradiction. Individual traces let readers judge whether the aggregate reflects a consistent signal or an artifact of averaging.

## What does the Thompson controller actually do?
Thompson sampling maintains Beta(α, β) posteriors over each arm. At each round it draws a sample from each arm's posterior and selects the highest draw. The α/β update used here is computed from the normalized drift reward clipped to [0,1]. It does not adapt the attacker's internal policy; it only selects which attacker family to use.

## Why is Evolutionary held out?
To measure whether robustness improvements driven by RandomSMILES/MCMC generalize to a structurally different attack family (evolutionary optimization). If transfer is positive, it suggests the defender learned general robustness features, not merely pattern-matching to the training attack family.

## What are the limitations of this pilot?
- n=3 seeds (3 rounds each): insufficient for population claims
- Single dataset (PI1M subset)
- Short fine-tuning (1 epoch per round)
- No architectural changes to defender between rounds
- MCMC and RandomSMILES are structurally similar (both do single-source edits); they are not maximally diverse attack families
