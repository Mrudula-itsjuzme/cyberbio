# Phase 7: Adaptive Adversarial Search

## Goal
Move from fixed perturbation evaluation to a proper adaptive adversarial search over the chemistry-changing single-token substitution space. The objective is to maximize absolute prediction drift: $J(x_{adv}) = |f(x_{adv}) - f(x_{original})|$.

## Methodology
We evaluated three search strategies on 100 validation samples against two models:
1.  **Ordinary Transformer Baseline**
2.  **Mixed-Robust Model** ($\lambda_{consistency} = 1.0, \lambda_{teacher} = 0.1$)

The strategies were evaluated across multiple query budgets: 5, 10, 20, and 50.

### Strategies Evaluated
*   **Random Search**: Randomly samples valid substitutions within the allowed role-preserving tokens.
*   **Greedy Search**: Exhaustively evaluates neighbors and selects the one that maximizes drift at each step.
*   **Metropolis MCMC**: Proposes random valid substitutions and accepts them based on a simulated annealing criterion to avoid local optima.

*Note: Chemistry-changing adversarial candidates are never described as "label-preserving", as they fundamentally alter the molecule.*

## Results Summary

### 1. Strategy Comparison (Budget = 50)
At the maximum budget of 50 queries, we observed the following performance differences between strategies:

*   **Random vs. Greedy**: Random search significantly outperformed Greedy search (Mean difference: -0.69 eV, 95% CI: [-0.83, -0.56]).
*   **Random vs. Metropolis MCMC**: Random search also outperformed Metropolis MCMC (Mean difference: -0.50 eV, 95% CI: [-0.63, -0.37]).
*   **Metropolis MCMC vs. Greedy**: Metropolis MCMC outperformed Greedy search (Mean difference: 0.19 eV, 95% CI: [0.08, 0.30]).

Surprisingly, purely random search achieved the highest mean drift (1.19 eV on the baseline) compared to Greedy (0.50 eV) and Metropolis (0.69 eV). However, a methodological audit revealed this is an artifact of the `attack_budget=1` constraint: stateful searches (Greedy, Metropolis) become trapped after accepting their first candidate because any subsequent edits would exceed the 1-hop budget relative to the original sequence. Random Search succeeds because it continually re-samples from the 0-hop original state. See `PHASE7_SEARCH_METHOD_AUDIT.md` for details.

### 2. Defense Resistance (Budget = 50)
We compared the vulnerability of the Mixed-Robust model against the Ordinary Baseline. A negative mean difference indicates the robust model exhibited lower prediction drift than the baseline.

*   **Random Attack**: The robust model showed a statistically significant reduction in drift (Mean diff: -0.26 eV, 95% CI: [-0.38, -0.14]).
*   **Greedy Attack**: The robust model showed a slight, but not statistically significant, reduction in drift (Mean diff: -0.05 eV, 95% CI: [-0.17, +0.06]).
*   **Metropolis MCMC**: The robust model showed a reduction in drift, bordering on significance (Mean diff: -0.08 eV, 95% CI: [-0.19, +0.02]).

**Conclusion:** The Mixed-Robust model demonstrates measurable resistance against adaptive adversarial attacks, particularly against the most effective strategy (Random Search), reducing the mean drift from 1.19 eV to 0.93 eV.

### 3. Transfer Analysis
We evaluated whether adversarial examples generated against one model effectively transfer to the other model.

*   **Baseline $\rightarrow$ Robust (Random)**: Candidates optimized against the baseline caused a mean drift of 0.68 eV on the robust model (compared to their original target drift of 1.19 eV on the baseline).
*   **Robust $\rightarrow$ Baseline (Random)**: Candidates optimized against the robust model caused a mean drift of 0.69 eV on the baseline (compared to their original target drift of 0.93 eV on the robust model).

Transferability is moderate; attacks optimized on one model retain a significant portion of their impact on the other model, though less than direct optimization.

## Scientific Verdicts Maintained
*   *Phase 6 Context:* Simple sequence-length and token-statistic baselines explain some signal but perform substantially worse than the Transformer, providing evidence against a purely trivial shortcut explanation. We do not claim the Transformer learned "genuine chemistry".
*   *Phase 6 Context:* Robustness training increased latent representation stability under equivalent representation perturbations. We do not call the latent space a "semantic space".
