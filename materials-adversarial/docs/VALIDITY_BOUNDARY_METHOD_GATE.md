# VALIDITY BOUNDARY METHOD GATE

## Analysis 1: Exploratory RDKit-parseability neighborhood
**Classification**: `PRELIMINARY_NO_BOUNDARY_SIGNAL`

**Analysis**:
The initial descriptive boundary study was performed using actual candidates generated in previous benchmarks. Local feasibility probing evaluated $k=10$ string mutations (character-level deletions/substitutions) per candidate, validating with real RDKit parser logic (`Chem.MolFromSmiles`).

The Spearman correlation between target-model absolute prediction drift and local boundary uncertainty ($u_{valid}$) was found to be negative ($\rho = -0.212$, $p = 0.019$). High drift is NOT statistically concentrated at the boundary where local validity is uncertain; if anything, higher drift is found in more deeply valid neighborhoods.

---

## Analysis 2: Canonical project-validity neighborhood
**Classification**: `NO_BOUNDARY_SIGNAL`

**Analysis**:
A robust replication using the exact canonical validator (`RDKitValidityChecker` via `Candidate`) and canonical proposal families (`CompositeProposalOperator` over `DeletionAttack`, `SubstitutionAttack`, and `InsertionAttack`) was performed on a method-stratified cohort.

- **Sample**: $n=600$ valid candidates across 361 unique source polymers.
- **Top 5% Drift**: Mean $u_{valid} = 0.669$ (compared to all-valid $u_{valid} = 0.743$). High drift candidates are located in *less* boundary-adjacent (more valid) local neighborhoods on average.
- **Pooled Correlation**: Spearman $\rho = -0.202$ ($p = 6.45 \times 10^{-7}$).
- **Source-level Correlation**: Spearman $\rho = -0.207$ ($p = 7.30 \times 10^{-5}$).
- **Method Stratification**: Verified across individual search methods, the trend remains structurally negative.

**Decision**:
The canonical empirical evidence strongly rejects the core hypothesis (H1). The relationship is inverted: prediction drift does not increase as candidates approach the validity boundary, but rather increases as candidates move into more secure/deeply valid structural space. Actively steering a search toward the boundary (Boundary-SMC) is fundamentally contraindicated by the data.

**Gate Outcome**:
HALT BOUNDARY-SMC IMPLEMENTATION. The negative result is supported and frozen.
