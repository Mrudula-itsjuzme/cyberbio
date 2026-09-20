# VALIDITY BOUNDARY METHOD GATE

## Analysis 1: Exploratory RDKit-parseability neighborhood
**Classification**: `PRELIMINARY_NO_BOUNDARY_SIGNAL`
(See historical records for details. Early parseability testing indicated a negative relationship.)

---

## Analysis 2: Canonical project-validity neighborhood + Robustness Check
**Classification**: `NO_BOUNDARY_SIGNAL`

**Analysis**:
A robust replication using the exact canonical tokenizer, exact benchmark vocabulary, canonical proposal families (`DeletionAttack`, `SubstitutionAttack`, `InsertionAttack`, `RearrangementAttack(window_size=3)`), and the real `RDKitValidityChecker` was performed.

- **Sample**: $n=600$ valid candidates across 361 unique sources, minimum 5 evaluated unique neighbors per candidate.
- **Top 5% Drift**: Mean $u_{valid} = 0.875 \pm 0.065$ (compared to all-valid $u_{valid} = 0.871 \pm 0.080$).
- **Pooled Correlation**: Spearman $\rho = 0.027$ (Cluster-Bootstrap 95% CI: $[-0.049, 0.111]$).
- **Within-Source Correlation**: Spearman $\rho = -0.071$ ($p = 0.083$).

**Decision**:
The canonical empirical evidence confirms that there is no meaningful association between target-model prediction drift and local boundary uncertainty. High-drift adversarial candidates are not significantly closer to (nor further from) the validity boundary than average valid candidates. Actively steering a search toward the boundary (Boundary-SMC) is not justified by the geometric properties of the candidate space.

**Gate Outcome**:
HALT BOUNDARY-SMC IMPLEMENTATION. The negative result is supported and frozen.
