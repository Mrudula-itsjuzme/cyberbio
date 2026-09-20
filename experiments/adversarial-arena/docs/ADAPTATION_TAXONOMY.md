# Adaptation Taxonomy in the Adversarial Arena

The Arena distinguishes between two fundamentally different types of attacker adaptation.

## 1. Adaptive Selection Gain ($G_{select}$)

This was measured in Phase 1 (the baseline pilot).
The **Thompson Controller** maintains a probability distribution over *which static attacker family to run* (RandomSMILES vs MCMC). 

$$ G_{select}(t) = R(C_{t+1}, D_t) - R(C_t, D_t) $$

where $C$ is the controller's choice of family. The families themselves (RandomSMILES, MCMC) **do not change their internal distributions**. The gain comes solely from switching to a family that performs better against the current defender.

## 2. True Attacker Adaptation Gain ($G_{adapt}$)

This is measured in Phase 2.
Probabilistic attackers (CEM, SMC, Validity-SMC) maintain internal state (proposal distributions, particle weights) that they update based on target-model feedback.

$$ G_{adapt}(t) = R(A_{t+1}, D_t) - R(A_t, D_t) $$

where:
- $A_t$: The attacker's internal state (e.g. CEM edit probabilities) *before* proposing candidates in round $t$.
- $A_{t+1}$: The attacker's internal state *after* fitting to the elite proposals from round $t$.
- $D_t$: The defender checkpoint at round $t$.

True adaptation means the *same algorithm* becomes strictly better at attacking the *same model* by updating its own internal generative parameters.
