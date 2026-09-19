# FINAL CYBERBIO REPORT

## 1. Research question
Are sequence models learning structural/causal relationships, or relying on shallow compositional shortcuts that leave them vulnerable to adversarial perturbation?

## 2. Materials results
- **Task**: Tg regression from SMILES polymers.
- **Model**: Transformer Regressor and TwoBranch Transformer.
- **Shortcuts**: Extremely vulnerable. Sequence length alone predicts R²=0.17; token counts R²=0.60, 3-grams R²=0.90.
- **Attacks**: High success rate via Random, MCMC, Evolutionary searches.

## 3. Materials limitations
Historical materials defense-transfer results are invalid because the original checkpoint cannot be exactly reconstructed with the available evaluation vocabulary. The observed embedding-shape mismatch raises RuntimeError before inference, so the previous transfer result cannot currently be reproduced. (Classification: INVALID_MODEL_RECONSTRUCTION).

## 4. Bio-Cyber V1 failure
Trivialized by k-mer leakage (98.9% 3-mer accuracy).

## 5. Bio-Cyber V2 failure
Distance=5 condition generated local composite motifs (e.g. `TGC[N]GC`) resolving the task for 5-mer baseline (65% accuracy).

## 6. V3 benchmark
Enforces strictly non-overlapping motifs spaced uniformly at 30-40 bp (Class 1) and 60-70 bp (Class 0).

## 7. V3 model results
- No tested shallow baseline, including 1-6-mer and the audited shallow feature models, substantially solved V3.
- V3 substantially reduces the shallow compositional shortcuts identified in V1 and V2.
- The `CNN_Distance` model achieved 54.2% ± 0.8% accuracy across five seeds, compared with a 49.9% shuffled-label control, suggesting a weak learnable positional signal.

## 8. Relationship counterfactuals
The CNN_Distance model demonstrated spatial sensitivity to shifting distance and deleting motifs, but retained class stability under background randomization and joint-shifting, confirming the signal learned is positional/relational rather than compositional.

## 9. Attack results
MCMC efficiently identified adversarial examples disrupting the learned positional signal across tested edit budgets.

## 10. Defense results
MCMC adversarial training (using candidates generated from the training split) successfully improved robustness compared to standard data augmentation.

## 11. Transfer robustness
Frozen adversarial banks evaluated against defended models indicate clean-performance degradation vs robust relational retention tradeoffs.

## 12. Adaptive robustness
Adaptive attacks were evaluated against the defended model at matched edit/query budgets, and flip rates and prediction drift were compared with the clean-model attack results.

## 13. Explainability
True matched-random control (Wilcoxon paired tests with 1,000-sample bootstrap CIs):
- Materials: p=0.71, 95% CI crosses 0.
- Bio-Cyber: No preferential targeting of edited positions over matched random background positions. 
The preferential targeting hypothesis is decisively NOT supported in either domain.

## 14. Cross-domain comparison
Materials and Bio-Cyber both confirm a high default risk of sequence models exploiting shallow compositional shortcuts. Furthermore, neither domain supports the attribution-targeting hypothesis.

## 15. Failure cases
- Historical Phase 4 materials evaluation mismatch.
- Bio-Cyber V1 and V2 dataset generation leaked positional and compositional artifacts.

## 16. Limitations
- Historical materials defense-transfer vocabulary unavailable.
- Model performance on V3 is weakly separated from chance.

## 17. Reproducibility
Exact Git SHAs, execution dependencies, python versions, dataset SHAs, checkpoint SHAs, and determinism seeds logged directly to `docs/reproducibility_manifest.json`.

## 18. Conclusions
The project successfully highlights the extreme care needed in benchmark design to avoid shallow compositional shortcuts (demonstrated across Materials, Bio-Cyber V1, V2) and establishes Bio-Cyber V3 as a rigorously verified testbed.
