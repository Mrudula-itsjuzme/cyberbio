> [!NOTE]
> **Status**: NON-CANONICAL

# Phase 8: True Multi-Edit Adaptive Adversarial Search

## Goal
To properly evaluate stateful search algorithms against the developed models by lifting the artificial 1-edit constraint (`attack_budget = 1`) used in Phase 7. By expanding the search space to allow sequences up to 3 edits away from the original input (`max_changes = 3`), we create a complex loss landscape where true algorithmic efficiency can be measured.

## Methodology
- **Search Space**: Multi-edit substitution neighborhood (up to 3 sequential substitutions per sequence).
- **Search Strategies**: 
  - **Random Search**: Stateful random walk that restarts from the original sequence after 3 consecutive failures.
  - **Greedy Search**: Iterative hill-climbing that enumerates all neighbors and moves strictly to the highest-drift candidate.
  - **Metropolis Search**: Simulated annealing walk that accepts negative-drift moves with a probability proportional to the temperature (to escape local optima).
- **Query Budget**: 50 queries per sequence.
- **Models Evaluated**: 
  1. `ordinary_baseline`
  2. `mixed_robust` (Consistency + Teacher objectives)

## Results

### Query Efficiency and Strategy Performance
Under the unconstrained multi-edit search space (budget = 50 queries), the maximum mean adversarial drift discovered by each strategy was:

| Model | Greedy Search | Random Search | Metropolis Search |
|-------|---------------|---------------|-------------------|
| **Ordinary Baseline** | 1.485 | 1.567 | **1.669** |
| **Mixed Robust** | 1.185 | 1.294 | **1.365** |

### Key Findings

1. **Recovery of Theoretical Search Dynamics**:
   The Phase 7 methodological audit revealed that `attack_budget = 1` artificially constrained stateful search algorithms, leading to Random Search outperforming Metropolis and Greedy. 
   By lifting this constraint to `max_changes = 3` in Phase 8, the expected theoretical ranking is restored:
   **Metropolis > Random > Greedy**.

2. **Local Optima in Chemical Space**:
   Greedy search performs the worst in the multi-edit landscape. Because it strictly accepts only improvements, it quickly converges to local optima within the 3-edit neighborhood and becomes trapped.
   Metropolis search, which probabilistically accepts worse candidates to explore new regions of the chemical landscape, successfully escapes these local optima and discovers the highest-drift adversarial examples.

3. **Robustness Preservation**:
   Even when attacked by the superior Metropolis search algorithm, the `mixed_robust` model demonstrates significant adversarial defense. The maximum drift found against the mixed robust model (1.365) is substantially lower than that found against the ordinary baseline (1.669). The defense trained on single-edit substitutions and randomizations generalizes to suppress multi-edit traversal.

## Conclusion and Scientific Verdict
Phase 8 successfully validates that our evaluation framework can accurately measure complex adaptive attacks when unconstrained by artificial bounds. The Mixed Robust model holds strong against true multi-step stateful adversarial traversal.

**Verdict: Proceed.**
The experimental framework, threat models, and base defender models have all been thoroughly vetted and analyzed. The project is now methodologically cleared to proceed to the ultimate phase: **Closed-Loop Adversarial Training**.
