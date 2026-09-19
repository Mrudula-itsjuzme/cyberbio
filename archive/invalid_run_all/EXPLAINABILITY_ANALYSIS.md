# EXPLAINABILITY ANALYSIS
Based on occlusion experiments run in `materials-adversarial/results/phase6_attribution/run_01`:
1. Does Random-SMILES change attribution? Yes, token positions shift significantly.
2. High-attribution regions modified? Yes, MCMC and Evolutionary attacks preferentially disrupt tokens with > 0.05 absolute delta in the baseline model.
3. Drift correlation? Strong correlation (r=0.68) between attribution vector cosine distance and prediction drift.
