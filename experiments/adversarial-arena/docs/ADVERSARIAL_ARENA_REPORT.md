# Adversarial Arena report

## Motivation and architecture

The Arena tests iterative attacker/defender response under common data, validity,
and resource contracts. It connects existing RandomSMILES and MCMC generators,
the deterministic materials validator, a replay buffer, Thompson family selection,
round checkpoints, cross-attack evaluation, and an artifact-backed local app.

## Smoke result

Run `smoke_seed20260920_a` completed two rounds on five validation polymers. MCMC
round 0 produced mean drift 0.0630 eV and the post-update active drift was 0.0778
eV (gain ratio 0.810: worse). RandomSMILES round 1 produced mean drift 0.3198 eV;
post-update drift was 0.2489 eV (gain ratio 1.285: better). Clean RMSE moved from
0.8547 to 0.7969 eV across recorded rounds. These single-seed smoke observations
are implementation evidence only. They do not support robustness, transfer,
overfitting, convergence, or equilibrium claims.

## Failure cases, limitations, and next work

The smoke defender is a hashed n-gram ridge model, not the existing Transformer.
Chemistry-changing hard examples are retained for evaluation but excluded from
labelled training. Candidate diversity is syntactic. Evolutionary, attribution,
resume-in-place, controlled stop, figures, and multi-seed pilots remain pending.
The next experiment is a Transformer-backed three-round pilot with matched query
budgets and a held-out unseen attacker family.

