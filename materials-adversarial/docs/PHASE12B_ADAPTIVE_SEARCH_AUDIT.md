> [!WARNING]
> **Historical Document / Superseded Methodology**
> This file chronicles an earlier development phase of the project (Phases 1-14). Early iterations of the MCMC defense in these logs evaluated closed-loop training using **original-label inheritance** for chemistry-changing edits. As established in the Phase 9 Scientific Audit, this assumption is physically invalid without a DFT oracle.
> The final canonical benchmark (`canonical_benchmark_no_leakage.json` and the final `EXPERIMENTAL_PROTOCOL.md`) explicitly abandons label inheritance in favor of **label-free consistency regularization**, and promotes **Rand-SMILES Augmentation** as the primary, physically sound defense. Please refer to `RESEARCH_CONTRIBUTIONS.md` for the current scientific consensus.

> [!NOTE]
> **Status**: CANONICAL

# Phase 12B: Adaptive Chemistry Search Audit

## Overview
Phase 12B was initiated to rigorously audit and repair the adaptive search methodology from Phase 12. The previous Phase 12 implementation suffered from unbounded search defects which allowed edit creep, rendering the resulting "5.961 eV drift" scientifically invalid under our canonical threat model (which restricts edits to $\le 3$ from the original molecule).

In this audit, we implemented a non-monkey-patched harness that:
1. Validates candidates using RDKit natively.
2. Explores the space using Random, Greedy, and Metropolis-style stochastic algorithms.
3. Explicitly limits query budgets $Q \in \{10, 20, 50\}$.
4. Enforces a strict Levenshtein-based atom token substitution budget of $\le 3$ measured purely against the original source SMILES.
5. Captures cross-model transfer drift without penalizing the query budget.

---

## Research Questions & Verdicts

### RQ1: Was the previous 5.961 eV drift physically valid under the 3-edit threat model?
**Verdict:** **No.** Under the rigorously constrained 3-edit boundary, the maximum drift observed across any model, seed, or budget was **3.19 eV** (GraphMPNN via Metropolis). The massive 5.961 eV drift was an artifact of the unbounded search bug allowing edit-budget creep.

### RQ2: Does GraphMPNN genuinely exhibit higher peak sensitivity than the Transformer to bounded chemical edits?
**Verdict:** **Yes.** At the maximum budget ($Q=50$), the GraphMPNN yielded peak drifts of ~3.19 eV, whereas the Transformer plateaued at ~3.10 eV. Even under strict bounds, the graph representation is more responsive to atom substitutions than the sequence representation.

### RQ3: Which search strategy is most efficient for bounded material discovery/adversarial search?
**Verdict:** **Greedy Search.** 
While Metropolis uncovered the absolute maximum singular drift (3.19 eV), Greedy Search achieved the highest **mean** target drift for both models:
- **GraphMPNN Mean Drift ($Q=50$):** 0.967 eV (Greedy) vs. 0.868 eV (Random) vs. 0.764 eV (Metropolis).
- **Transformer Mean Drift ($Q=50$):** 0.855 eV (Greedy) vs. 0.744 eV (Random) vs. 0.659 eV (Metropolis).
Greedy optimization is highly effective at exploiting local gradients in the property manifold.

### RQ4: How does adversarial transferability compare between Graph and Sequence architectures?
**Verdict:** **Asymmetric but significant.**
When optimizing against GraphMPNN (Greedy $Q=50$), the average target drift of 0.967 eV transfers to the Transformer with a drift of **0.536 eV** (55% transfer rate).
When optimizing against the Transformer (Greedy $Q=50$), the average target drift of 0.855 eV transfers to the GraphMPNN with a drift of **0.584 eV** (68% transfer rate).
The GraphMPNN is more susceptible to "blind" chemical changes found via the sequence model than vice versa.

### RQ5: Do the models agree on which molecules are vulnerable?
**Verdict:** **Yes, moderate agreement.**
The Pearson correlation between GraphMPNN drift and Transformer drift across all valid generated candidates is **0.603**, with a Spearman rank correlation of **0.613**. Both architectures generally identify the same foundational chemical substitutions as highly impactful to the predicted bandgap.

### RQ6: Is the budget $Q=50$ sufficient to saturate the 3-edit neighborhood?
**Verdict:** **Likely.** Given the constraints (3 token edits), the mean and max drift curves show flattening between $Q=20$ and $Q=50$, suggesting that a budget of 50 local neighbor evaluations is sufficient to explore the most sensitive regions of the 3-edit space without needing an exhaustive oracle.

### RQ7: What is the nature of the top found candidates?
**Verdict:** The top 5 candidates discovered all successfully exhausted the 3-edit budget (e.g., source `[*]OC(=O)CCS(=O)(=O)CCC([*])=O` modified up to 3 times) to force a massive ~3.19 eV shift on the GraphMPNN, demonstrating that even a minimal 3-atom substitution can drastically alter predicted bandgap.

### RQ8: How should this impact our defensive strategy?
**Verdict:** High drift under chemistry-changing edits is **NOT physical prediction error**; the true bandgap legitimately changes. Because GraphMPNN perfectly solves the equivalent-SMILES serialization problem (0.0 eV drift), any non-zero drift it produces here is due solely to *physical* changes. We cannot train a defender to minimize this drift without independent property supervision (e.g., a DFT oracle).

## Conclusion
The Phase 12B Audit successfully corrects the developmental errors of Phase 12. The GraphMPNN remains our canonical architecture: it possesses perfect serialization invariance while exhibiting high, physically-grounded sensitivity to bounded chemical perturbations.
