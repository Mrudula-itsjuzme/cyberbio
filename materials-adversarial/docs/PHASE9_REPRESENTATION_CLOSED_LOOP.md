> [!NOTE]
> **Status**: NON-CANONICAL

# Phase 9: Representation-Preserving Closed-Loop Adversarial Training

## 1. Objective and Canonical Designation
This phase establishes the canonical closed-loop adversarial-learning experiment for this project. Unlike previous phases that used chemistry-changing multi-edit attacks (which are valid for vulnerability analysis but lack known physical labels), this phase uses **Representation-Preserving Adaptive Attacks**. 

The attacker searches over canonically equivalent molecular string representations (randomized SMILES). Because the underlying physical molecule remains identical, the measured Bandgap of the original sequence is scientifically valid as an adversarial training target.

**Primary Question**: Can a query-bounded attacker find canonically equivalent polymer string representations that cause large prediction drift, and can adversarial training on those worst-case representations reduce that drift under a fresh re-attack?

## 2. Experimental Setup
*   **Starting Defender (D0)**: The frozen Mixed-Robust model from Phase 4 (`mix_robust_0.1`).
    *   Checkpoint Hash: `b81aa245444431a31ae2d95ef25fe8b6f6aae39ac95c60fd05d3303af2567dc7`
    *   Clean MAE: 0.4411 eV | Clean R²: 0.8190
*   **Attacker Objective**: Maximize representation-induced drift: $|D0(x_{adv}) - D0(x_{original})|$ subject to $canonical(x_{adv}) == canonical(x_{original})$ and $x_{adv} \neq x_{original}$.
*   **Training Attack**: Best-of-N adaptive search on 500 deterministic TRAIN sources. Budget $Q=50$. (Seed: 20260901)
*   **Adversarial Training**: D1 initialized from D0 weights. Target: `L_clean + λ_adv * L_adv_repr + λ_cons * L_consistency`.
*   **Re-attack**: Fresh evaluation on 100 validation sources with new attack seeds (Seed: 77777777).

## 3. Training Attack Results (D0 Vulnerability)
The representation-preserving attacker proved extremely successful against D0:
*   **Valid equivalent candidates found**: 500/500 (100.0%)
*   **Mean Training Drift**: 0.6619 eV
*   **p90 Training Drift**: 1.1030 eV
*   **Max Training Drift**: 2.8670 eV
*   **Stress Success Rate (>=0.10 eV)**: 100.0%

This confirms that the defender is highly sensitive to purely syntactical changes in canonical representations when an adaptive search guides the generation.

## 4. Lambda Selection and Prediction Collapse
During D1 training, the model experienced severe clean performance degradation under adversarial fine-tuning.
*   `λ_adv = 0.1`: val MAE = 1.023 eV (Degradation +0.582 eV) -> Not feasible
*   `λ_adv = 0.5`: val MAE = 1.070 eV (Degradation +0.629 eV) -> Not feasible

Since all tested `λ_adv` values exceeded the strict `0.02` eV clean MAE degradation tolerance, `0.1` was selected as a fallback. 
**Crucially, D1 suffered Output Collapse:**
*   **D0 Prediction Std Dev**: 1.375 eV (R²: 0.819)
*   **D1 Prediction Std Dev**: 0.503 eV (R²: 0.142)

The adversarial objective pulled the model completely off its regression task, severely flattening the prediction range.

## 5. Fresh Re-Attack Evaluation
A fresh search on 100 validation sources yielded the following representation drift:

| Metric | D0 (Frozen) | D1 (Adversarially Trained) |
| :--- | :--- | :--- |
| **Mean Drift** | 0.5855 eV | 0.3098 eV |
| **p90 Drift** | 0.9054 eV | 0.7692 eV |
| **Max Drift** | 1.8620 eV | 1.4380 eV |
| **Stress Rate** | 100.0% | 72.0% |

### Closed-Loop Paired Improvement
*   **Mean Improvement**: +0.2757 eV
*   **95% CI**: [0.2061, 0.3397]
*   **Fraction Improved**: 80.0%
*   **Fraction Worsened**: 20.0%

### Attack Efficiency (Query Curves)
| Model | Q=5 | Q=10 | Q=20 | Q=50 |
| :--- | :--- | :--- | :--- | :--- |
| **D0** | 0.3909 eV | 0.4575 eV | 0.5152 eV | 0.5855 eV |
| **D1** | 0.1983 eV | 0.2463 eV | 0.2791 eV | 0.3098 eV |

The attacker required roughly 50 queries against D1 to reach the drift achievable in < 5 queries against D0, indicating an empirical increase in attack cost.

## 6. Adversarial Overfitting Check
*   D1 drift on *training* adversaries: 0.0722 eV
*   D1 drift on *fresh* adversaries: 0.3098 eV
*   **Gap**: 4.3x worse on fresh adversaries.
*   **Conclusion**: Strong evidence of adversarial overfitting. D1 largely memorized the specific worst-case strings generated during the training phase.

## 7. Secondary Regression Checks
Because D1's output collapsed, it exhibits artificially lower sensitivity across all frozen banks, but this is a side effect of flattened predictions, not true robustness.
*   **D0 Frozen Drift**: Rand: 0.216 | Sub: 0.190 | Del: 0.216
*   **D1 Frozen Drift**: Rand: 0.089 | Sub: 0.095 | Del: 0.102

## 8. Verdict: NOT SUPPORTED
While D1 theoretically demonstrated a statistically significant reduction in fresh adversarial drift (+0.276 eV, 95% CI [0.206, 0.340]) and forced the attacker to use more queries, the primary hypothesis is **NOT SUPPORTED** due to:
1.  **Output Collapse**: Clean performance plummeted to an R² of 0.142. The model achieved lower drift primarily by narrowing its entire output range.
2.  **Adversarial Overfitting**: The model was >4x more resilient to training representations than fresh equivalent representations.

**Conclusion**: Adaptive adversarial training on worst-case canonically equivalent polymer representations causes fatal degradation to the primary regression task under the current Transformer architecture and hyperparameter constraints.

## 9. Recommendations
Do not run a second cycle of closed-loop representation training. 
The immediate bottleneck is no longer generating strong attacks or defining valid targets; the bottleneck is the model's capacity to simultaneously map widely divergent syntactic representations of the same molecule to a high-variance continuous physical property without collapsing.
