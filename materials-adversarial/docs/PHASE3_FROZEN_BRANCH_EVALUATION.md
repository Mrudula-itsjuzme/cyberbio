> [!NOTE]
> **Status**: HISTORICAL

# Phase 3: Frozen Branch Evaluation

## 1. Goal
To evaluate whether the two-branch specialized model achieved genuine semantic specialization (Representation Invariance in Branch A, Chemistry Sensitivity in Branch B) by testing against frozen, unseen perturbation banks.

## 2. Leakage Prevention and Bank Generation
We reconstructed the training pairs used during Phase 2 training to ensure no candidate representations overlapped with the Phase 3 validation candidate banks.
- **Randomization Candidates:** 3,120 eligible unseen candidates (0 exact training overlaps).
- **Substitution Candidates:** 1,632 eligible unseen candidates (0 exact training overlaps).

The banks were frozen immutably with SHA256 hashes:
- `randomization_candidates.jsonl`: `06dd236bc93f66dbcbb6f68896d5f0c4a596ba4cbfa5de4b15d2f74fe3fd60de`
- `substitution_candidates.jsonl`: `d005f9352ec394561cd7da6a865b0d372429b248390d92ac066ea4a5a1ba29a0`

## 3. Prediction Robustness

| Model | Attack | Mean Drift | P95 Drift | Max Drift | Invariance / Stress Exceedance |
|-------|--------|------------|-----------|-----------|---------------------------------|
| Ordinary Baseline | Randomization | 0.614 eV | 1.821 eV | 4.010 eV | 0.000 |
| Architecture Control | Randomization | 0.382 eV | 1.154 eV | 3.752 eV | 0.210 |
| Specialized Model | Randomization | 0.392 eV | 1.180 eV | 2.842 eV | 0.189 |
| Ordinary Baseline | Substitution | 0.291 eV | 1.095 eV | 2.909 eV | 0.181 (exceedance) |
| Architecture Control | Substitution | 0.281 eV | 0.986 eV | 2.782 eV | 0.188 (exceedance) |
| Specialized Model | Substitution | 0.306 eV | 1.168 eV | 2.733 eV | 0.214 (exceedance) |

**Paired Comparisons vs Ordinary Baseline:**
- **Architecture Control (Rand):** +0.23 eV improvement (95% CI: 0.20 to 0.26), 65.8% improved.
- **Specialized Model (Rand):** +0.22 eV improvement (95% CI: 0.19 to 0.25), 64.4% improved.

**Insight:** The two-branch architecture intrinsically improves robustness to randomization over the ordinary baseline, but the specialized auxiliary losses provide *no additional robustness*.

## 4. Branch Selectivity

To test semantic specialization, we computed the selectivity ratio $S = d_{sub} / d_{rand}$ for both branches. True specialization would show $S_A$ behaving distinctly from $S_B$.

| Model | d_A_rand | d_B_rand | d_A_sub | d_B_sub | S_A | S_B |
|-------|----------|----------|---------|---------|-----|-----|
| Architecture Control | 1.80 | 2.39 | 0.97 | 1.43 | **0.54** | **0.60** |
| Specialized Model | 0.55 | 3.72 | 0.36 | 2.11 | **0.65** | **0.57** |

**Insight:** The selectivity ratios ($S_A$ and $S_B$) are virtually identical (~0.6) between the two branches in the Specialized Model, and they closely match the ratios seen in the unconstrained Architecture Control. The specialized model simply scaled down all distances in Branch A and scaled up all distances in Branch B, rather than learning semantically distinct responses to different perturbations.

## 5. Representation Collapse and Ablation

**Variance:**
- Architecture Control: Var A = 0.285, Var B = 0.533
- Specialized Model: Var A = 0.036, Var B = 0.783

**Ablation (Clean MAE when zeroing branch):**
- **Architecture Control:** Ablate A -> 0.567 eV, Ablate B -> 0.968 eV
- **Specialized Model:** Ablate A -> 0.509 eV, Ablate B -> 1.042 eV (Clean MAE is ~0.46 eV)

**Insight:** Branch A in the specialized model has severely suppressed variance. Because its variance is so low, it contributes very little to the final prediction (ablating it only worsens MAE to 0.509). Branch B has become the dominant path for all regression logic. The network satisfied the `loss_repr` by simply turning off Branch A's variance, forcing Branch B to handle both regression and `loss_chem`/`loss_div`.

## 6. Verdict and Interpretation

**Verdict: NOT SUPPORTED**

The hypothesis that the auxiliary multi-objective loss would induce genuine semantic specialization into "Representation Invariance" and "Chemistry Sensitivity" branches is definitively rejected.

**Negative Findings:**
- The specialized model did not learn semantic specialization; it learned **variance asymmetry**.
- The model bypassed the multi-objective constraints by suppressing the variance of Branch A (making it trivially invariant to everything) and relying almost exclusively on Branch B for regression.
- The two-branch architecture itself provides a modest intrinsic robustness benefit over the baseline, but the specialized losses do not improve upon it.

## 7. Status and Next Steps
Phase 3 is complete. The specialized architecture is **not worth keeping** as it relies on a collapsed-variance shortcut rather than true semantic disentanglement. Future work should focus on adversarial training or alternative regularizations that prevent magnitude-based shortcuts.
