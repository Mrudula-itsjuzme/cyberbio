# Experiment protocol

1. Smoke: two rounds, five validation sources, tiny budgets, no test-set use.
2. Pilot: three to five rounds, 10–20 sources, Transformer updates, RandomSMILES,
   MCMC, and Evolutionary with matched target-query budgets.
3. Full: only after pilot gates pass; 5–10 rounds, 50–100 sources, at least three
   seeds, pre-registered stopping and catastrophic-clean-degradation thresholds.

All runs record git state, seed, dataset hash, configs, transitions, candidates,
checkpoints, and cross-attack cells. The sealed test set remains unavailable to
attacker adaptation and model selection.

