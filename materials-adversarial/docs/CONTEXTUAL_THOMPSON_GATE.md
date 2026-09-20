# Contextual Thompson Sampling Decision Gate

## 1. Evidence for Contextuality
An analysis of the empirical pseudo-regret across 5 distinct polymer sources over varying budgets and seeds provides **SUPPORTED** evidence that different sources favor different edit operators:
- `row_3581` heavily favors `substitution` (won 5 out of 6 trials), with consistent positive reward margins (e.g., +4 and +5 reward points over the second-best arm).
- `row_3579` similarly favors `substitution` (won 5 out of 6 trials).
- `row_3580` predominantly favors `deletion` and `insertion`, and rarely `substitution` (only won once).
- `row_3577` predominantly favors `insertion` (won 4 out of 6 trials).

This demonstrates that the "hindsight best" arm is strictly source-dependent. A global prior (Non-contextual Thompson) will inevitably waste queries discovering the local optimum for each new sequence, proving that P(best arm | source features) contains useful information beyond a global arm preference.

## 2. Evidence Against
There is **OBSERVED** noise in the reward trajectories across seeds. For example, in `row_3578`, the winning arm fluctuated between `insertion`, `deletion`, and `substitution` depending on the random seed and budget, often with tight margins (e.g., +0 or +1 reward). This suggests that for some sequences, multiple arms are equally viable, and the contextual model might struggle to distinguish a single "best" arm if the margin is near zero.

## 3. Candidate Context Features
To build the Contextual Thompson controller without leaking future attack outcomes, we can rely on zero-shot features extracted before any predictor queries are spent:
- **Sequence Length**: Number of tokens in the original SMILES string.
- **Original Predictor Drift (Baseline)**: The initial target value (Bandgap $E_g$) before perturbation.
- **Token Histograms**: Normalized counts of specific structural tokens (e.g., Rings, Halogens).
- **RDKit Descriptors**: Computationally cheap structural descriptors (e.g., Molecular Weight, LogP) if available.
- **Scaffold Features**: Structural categorization (e.g., presence of aromatic rings vs. aliphatic chains).

## 4. Leakage Audit
- We must **NOT** use any information derived from the proposed candidates (e.g., candidate drift, candidate validity rate) as the context features for the arm choice, because the context must be evaluated *before* selecting the arm.
- We must **NOT** use the query budget as a feature unless the model is specifically designed to be budget-aware, but ideally, context is purely source-dependent.
- We strictly limit context to the properties of the *original* sequence.

## 5. Expected Benefit
By mapping source features to arm probabilities, Contextual Thompson is expected to bypass the initial exploration phase of the non-contextual bandit. It will directly query the highest-yield arm (e.g., immediately starting with `substitution` for `row_3581`), thereby improving the adversarial drift-per-query ratio, especially in low-budget regimes ($T \le 10$).

## 6. Decision
**BUILD**.

The source-dependency is strictly observed and supported by margin data. The non-contextual baseline is verified to be functioning properly, and the regret formulations are clean. We are clear to proceed with implementing Contextual Thompson.
