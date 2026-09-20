# Negative and Null Results

- CEM seed 20260920 R1/R2 had negative raw G_carry.
- Other CEM seeds were positive.
- Therefore CEM is heterogeneous, not globally degraded.
- 12/27 events were query-mismatched.
- Only 6 matched post-init events remain.
- SMC and Validity-SMC produced identical realized particle trajectories and attack metrics under the tested configuration.
- Validity-SMC maintained an extra validity posterior, but it produced no observed incremental effect.

## Mechanism Check (SMC vs Validity-SMC)
- **FACT**: Particle states were identical across SMC and Validity-SMC.
- **FACT**: Drifts were identical.
- **FACT**: Validity-SMC has an extra validity posterior.
- **FACT**: The posterior participates in scoring.
- **INFERENCE**: The posterior did not alter realized selection dynamics under this budget.
