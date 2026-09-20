# VALIDITY BOUNDARY METHOD GATE

## Analysis 1: Exploratory RDKit-parseability neighborhood
**Classification**: `PRELIMINARY_NO_BOUNDARY_SIGNAL`
(See historical records for details. Early parseability testing indicated a negative/null relationship.)

---

## Analysis 2: Canonical project-validity neighborhood + Robustness Check
**Classification**: `NO_DETECTABLE_BOUNDARY_ASSOCIATION`

**Analysis**:
Under the canonical tokenizer, vocabulary, proposal operators, validator, sampled candidate cohort, and k=10 local-neighborhood definition, prediction drift showed no detectable association with local feasibility uncertainty.

- **Sample**: $n=1000$ probed candidates across 5 methods (Substitution, Insertion, Deletion, Randomization, Rearrangement), yielding robust evaluations.
- **Pooled Correlation**: Spearman $\rho = 0.027$ (Cluster-Bootstrap 95% CI: $[-0.049, 0.111]$).
- **Source-level Correlation**: Spearman $\rho = 0.057$ ($p = 0.279$).
- **Within-Source Correlation**: Spearman $\rho = -0.071$ ($p = 0.083$).
- **Method Stratification**: See `results/validity_boundary/canonical_boundary_by_method.csv`. Null/flat association structure applies across individual attack geometries.
- **Probe Stability ($u_{valid}$ std)**: Mean: 0.1207, Median: 0.1170, P90: 0.2381, Max: 0.3987.
- **K-Sensitivity**: `K20_SENSITIVITY_NOT_RUN`. (Sufficient signal stability established via repeated k=10 probes).

**Decision**:
The canonical empirical evidence confirms that there is no meaningful association between target-model prediction drift and local boundary uncertainty. High-drift adversarial candidates are not significantly closer to (nor further from) the validity boundary than average valid candidates.

**Gate Outcome**:
`NO_GO` for Boundary-SMC. No positive empirical signal justifies a boundary-seeking optimizer.
