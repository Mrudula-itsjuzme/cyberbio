> [!NOTE]
> **Status**: HISTORICAL

# Phase 4: Controlled Robustness Training and Evaluation

## Scientific Conclusion
**Verdict:** **SUPPORTED.** The evidence supports that robustness to representation perturbations can be learned directly through explicit paired-training objectives at the prediction layer without imposing strict (and unstable) semantic specialization constraints inside the architecture.

By moving the regularization from the intermediate embeddings (Phase 3) to the final prediction output (Phase 4), we achieved a substantial reduction in Randomization drift and a modest improvement in Substitution drift, while maintaining clean validation performance.

## Model Selection & Clean Metrics

All models were evaluated on the immutable Phase 3 Candidate Banks using the verified frozen TargetScaler.

| Model | Selection Criterion | Clean Validation MAE (eV) |
| --- | --- | --- |
| **Phase 1 Ordinary Baseline** | N/A | 0.460 eV |
| **Phase 3 Architecture Control** | N/A | 0.461 eV |
| **Randomization-Robust** | Best `lambda_consistency=1.0` | **0.443 eV** |
| **Mixed-Robust** | Best `lambda_teacher=0.1` (w/ `lambda_cons=1.0`) | **0.441 eV** |

*Note: Clean MAE improved over the baseline models, fulfilling the constraint that MAE degradation must be ≤ 0.02 eV.*

## Primary Robustness Matrix (Mean Drift)

| Model | Attack | Mean Drift (eV) ↓ | Stress Exceedance Rate ↓ |
| --- | --- | --- | --- |
| **Ordinary Baseline** | Randomization | 0.614 eV | N/A |
| **Architecture Control**| Randomization | 0.382 eV | N/A |
| **Randomization-Robust**| Randomization | **0.301 eV** | N/A |
| **Mixed-Robust** | Randomization | 0.308 eV | N/A |
| | | | |
| **Ordinary Baseline** | Substitution | 0.291 eV | 20.6% |
| **Architecture Control**| Substitution | 0.281 eV | 18.7% |
| **Randomization-Robust**| Substitution | 0.294 eV | 20.6% |
| **Mixed-Robust** | Substitution | **0.269 eV** | **18.5%** |

## Bootstrapped Improvements (95% CI)

We evaluate the statistical significance of the improvements over the frozen Architecture Control model.

### Randomization-Robust Model (vs Architecture Control)
* **Interpretation:** The explicit consistency loss successfully enforces prediction invariance to representation randomization. The 95% CI is entirely positive, providing evidence of a statistically significant improvement over the architectural baseline.

### Mixed-Robust Model (vs Architecture Control)
* **Substitution Improvement:** 0.012 eV (95% CI: `[0.003 eV, 0.030 eV]`)
* **Interpretation:** Distilling substitution predictions from the frozen Architecture Control teacher successfully reduces substitution drift by a statistically significant margin (CI completely positive). Crucially, this was achieved without sacrificing the randomization invariance learned from the consistency loss, yielding a more robust model.

## Answers to Phase 4 Research Directives

   **Yes.** The Randomization-Robust model reduced mean drift to 0.301 eV (a substantial reduction over the Ordinary Baseline and Architecture Control). Bootstrapping provides evidence that this improvement is highly significant.

2. **Can chemistry-preserving constraints (distillation) confer robustness against Substitution?**
   **Yes.** The Mixed-Robust model successfully improved Substitution drift to 0.269 eV (a significant improvement over both Baseline and Control) by distilling targets from the frozen Architecture Control.

3. **Does clean validation performance remain acceptable?**
   **Yes.** Surprisingly, the clean validation MAE actually *improved* with the regularization, dropping from 0.461 eV (Control) to 0.441 eV (Mixed-Robust). 

4. **Is the auxiliary specialization loss approach (Phase 3) necessary?**
   **No.** By moving the loss formulation to the prediction space, we completely circumvented the catastrophic representation variance collapse observed in Phase 3. 

## Final Conclusion
The Mixed-Robust Two-Branch Architecture provides a defense against both randomized and chemically-adversarial representation attacks. It establishes a viable defense mechanism for polymer bandgap prediction models operating on inherently ambiguous molecular representations without needing artificial interior embeddings.
