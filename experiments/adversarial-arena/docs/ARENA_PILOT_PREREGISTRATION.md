# Transformer Arena pilot preregistration

Frozen before executing any Transformer defender update on 2026-09-20.

## Research question

Does iterative adversarial training improve robustness without unacceptable clean-performance degradation, and do adaptive or mixed curricula expose new failure modes after each defender update?

## Hypotheses

- **H1:** repeated defender updates reduce mean failure magnitude against previously seen attacker families.
- **H2:** an adaptive attacker curriculum improves unseen-attacker robustness more than a single fixed curriculum.
- **H3:** robustness improvement may incur worse clean validation performance.
- **H4:** an adaptive attacker can recover drift against an updated defender.
- **H5:** Evolutionary search finds higher-drift validated candidates than RandomSMILES at matched candidate-query budgets.
- **H6:** cross-attack transfer is incomplete.
- **H7:** attacker effectiveness decreases across rounds if the defender learns.

Hypotheses will not be edited after results are observed.

## Fixed protocol

- Dataset: canonical polyVERSE bandgap processed artifact and frozen train/validation split. The test split stays sealed.
- Initial defender: canonical `results/models/transformer_regressor/model.pt`, its exact vocabulary, architecture, and fixed target scaler.
- Sources: the first 10 validation rows after stable `polymer_id` sort, reused in every round and arm.
- Rounds: 3.
- Seeds: `20260920`, `20260921`, and `20260922` for the primary adaptive arm, subject to CPU feasibility. Fixed arms use seed `20260920` as feasibility controls.
- Training curricula: fixed RandomSMILES, fixed MCMC, fixed Evolutionary, and Thompson adaptive.
- Attack families: RandomSMILES, MCMC, and Evolutionary only.
- Query budget: 25 candidate target-model queries per source per search attacker. The source prediction is cached and reported separately. A method unable to spend its budget reports actual usage; no hidden predictions are permitted.
- Hard selection: valid top-4 by drift per active round, with deterministic ties.
- Defender update: one AdamW epoch per round over the complete canonical training split, warm-started from the preceding checkpoint; learning rate `0.001`, batch size `16`, weight decay `0.0001`.
- Loss: clean supervised MSE plus `0.1 ×` prediction-consistency MSE for validated RandomSMILES pairs. MCMC and Evolutionary chemistry-changing candidates receive no copied label and are diagnostic-only.
- Clean gate: canonical validation MAE/RMSE/R² before and after each update. A relative RMSE increase over 10% raises a warning but does not silently alter training.
- Unseen transfer: for fixed curricula, attackers not selected for training are cross-attack transfer evaluations. For adaptive training, the least-selected family is reported descriptively; it is not a strictly unseen family once selected.
- Reward: `mean_valid_drift / (1 + mean_valid_drift)`, fixed before execution. It is bounded in `[0,1)` and stored with raw validity and query efficiency.
- Statistical status: descriptive pilot. Three adaptive seeds permit variability reporting but not broad population claims.

## Analysis definitions

`R(A,D)` is mean absolute prediction drift over valid candidates. Defender gain is `R(A,D_before) / R(A,D_after)`; values above one mean improvement. Clean change is `RMSE_after / RMSE_before`; values above one mean degradation. Transfer gain uses the same ratio for an attacker excluded from the training curriculum. Attacker adaptation gain compares successive selected attackers on the same pre-update defender where data exist.

No Nash-equilibrium claim will be made. Plateau, oscillation, recovery, and diminishing returns are descriptive labels only.

