---
title: "CyberBio Technical Report — Part II, Version 2: Attacks, Search, Defenses, and Physical Validation"
author: "CyberBio Project Documentation"
date: "September 2026"
---

> **Evidence policy.** Statements labelled **Project fact** are grounded in the current `Mrudula-itsjuzme/cyberbio` repository. Sections labelled **Mathematical background** explain standard theory needed to understand the implementation. Historical results are kept only when their status is explicit: canonical, historical, rejected, blocked, or developmental-unverified.

# Scope of this volume

This volume answers: **how are candidates generated, what does “valid” mean, what does the search optimize, how are query budgets enforced, what defenses were attempted, and why does chemistry-changing robustness require an external oracle?**

# 1. Threat model

Let $f$ be a fixed property predictor and $x$ a clean polymer representation. For a feasible candidate set $\mathcal C(x)$, a generic untargeted sensitivity search is

$$
x^*=
\arg\max_{x'\in\mathcal C(x)}
J(x'),
$$

with

$$
J(x')=|f(x')-f(x)|.
$$

This objective measures **prediction drift**. It becomes a true adversarial-error objective only if the underlying physical label should remain unchanged, or if an independent oracle supplies the changed ground truth.

# 2. Two semantics, two rules

## 2.1 Equivalent-SMILES control

If canonical graph identity is preserved,

$$
\operatorname{canon}(x')=\operatorname{canon}(x),
$$

then the project treats the candidate as representation-preserving. The original physical target may be reused.

## 2.2 Chemistry-changing stress

For substitutions, deletions, insertions, and bounded multi-edit search,

$$
\operatorname{canon}(x')\neq\operatorname{canon}(x).
$$

The original $E_g$ must **not** be assumed to remain correct.

# 3. Equivalent-SMILES generation

`SmilesRandomizationAttack` parses the source molecule with RDKit, then uses a seeded randomized-SMILES vector generator. The candidate is accepted as a representation control only when it is a different serialization of the same molecular graph.

The representation drift is

$$
\Delta_{repr}(x,x')=|f(x')-f(x)|.
$$

For a graph-native model, this should collapse to numerical noise because both strings produce the same graph.

# 4. Chemical plausibility gates

The project uses multiple layers of chemical checks depending on experiment generation.

## 4.1 RDKit parse and sanitization

A candidate must parse into a molecular graph and pass sanitization/valence checks. This rejects malformed strings, but does not prove synthesizability.

## 4.2 Attachment-star preservation

For polymer repeat units, wildcard `*` attachment count is preserved:

$$
\#\star(x')=\#\star(x).
$$

## 4.3 Molecular-weight bound

Some historical validators require

$$
0.5\le \frac{MW(x')}{MW(x)}\le 1.5.
$$

## 4.4 Morgan fingerprint Tanimoto similarity

For binary fingerprint vectors $a,b$,

$$
T(a,b)=
\frac{|a\cap b|}{|a|+|b|-|a\cap b|}.
$$

Historical MCMC-style experiments often used

$$
T\ge 0.5.
$$

This is a structural-similarity constraint, not a proof of label preservation.

# 5. Local chemistry-changing operators

## 5.1 Substitution

A token- or atom-level substitution replaces one mutable chemical unit. Different experiment generations used different operator semantics, so the operator must always be named together with its implementation.

A single substitution can be written abstractly as

$$
x'=\operatorname{substitute}(x,i,a\to b).
$$

## 5.2 Deletion

Deletion removes a mutable unit, subject to validity:

$$
x'=\operatorname{delete}(x,i).
$$

Deletion is chemistry-changing and was evaluated as a fixed stress in the canonical architecture work.

## 5.3 Insertion

Insertion exists in historical attack inventories, but the release audit marks single-edit insertion as **not canonically evaluated** in the final fixed-bank/search claims. Version 2 therefore does not silently promote it to a final result.

## 5.4 Rearrangement

Rearrangement appears in historical sequence-model experiments. It belongs to the Transformer-era robustness story, not the final GraphMPNN canonical chemistry-stress result.

# 6. Historical Metropolis-style sequence search

The earlier `ProbabilisticMCMCAttack` chooses among substitution, insertion, and deletion proposals, scores valid candidates, and accepts/rejects with a Metropolis-style rule.

Its total score is approximately

$$
S(x')=C(x')+|f(x')-f(x_0)|,
$$

where $C$ is a coarse chemical-validity score.

For current state $x_t$ and proposal $x'$, it uses

$$
\Delta=S(x')-S(x_t),
$$

$$
p_{acc}
=
\begin{cases}
1,&\Delta>0,\\
\exp(\Delta/T),&\Delta\le 0.
\end{cases}
$$

The implementation is explicitly documented as **Metropolis-style model-guided stochastic search**, not formal Metropolis-Hastings, because it does not include the reverse proposal ratio

$$
\frac{q(x_t\mid x')}{q(x'\mid x_t)}.
$$

# 7. Phase 12B bounded adaptive search

The Phase 12B repair changed the center of gravity of the attack story. Candidates are constrained to be at most three token edits from the **original** source, not merely from the previous state.

## 7.1 Levenshtein recurrence used in Phase 12B

For token sequences $a_{1:m}$ and $b_{1:n}$, define a dynamic-programming table $D$. Boundary conditions:

$$
D(i,0)=i,
\qquad
D(0,j)=j.
$$

Recurrence:

$$
D(i,j)=
\min
\begin{cases}
D(i-1,j)+1,\\
D(i,j-1)+1,\\
D(i-1,j-1)+\mathbf 1[a_i\neq b_j].
\end{cases}
$$

Phase 12B rejects candidates with

$$
D(x_{source},x')>3.
$$

Important later audit nuance: Framework V2 showed that canonicalized SMILES string distance can overstate **chemical** edit count. Therefore Phase 12B’s token-distance threat model and Framework V2’s graph-based edit accounting are related but not interchangeable.

# 8. Random search

Random search repeatedly samples valid neighbors of the source and keeps the candidate with the largest drift.

For evaluated candidates $x'_1,\ldots,x'_Q$,

$$
x^*=\arg\max_k J(x'_k).
$$

Its strength is simplicity and low algorithmic bias; its weakness is inefficient exploitation.

# 9. Greedy search

Greedy search generates a small neighborhood from the current state and moves to the strongest improving candidate.

If $\mathcal N(x_t)$ is the local candidate batch,

$$
x_{t+1}
=
\arg\max_{x'\in\mathcal N(x_t)}J(x')
$$

when the best candidate improves the stored maximum. This is strongly exploitative and can get stuck in local optima.

# 10. Phase 12B Metropolis search

For drift

$$
J_t=|f(x_t)-f(x_0)|,
$$

and candidate drift $J'$, the repaired Phase 12B search uses

$$
\Delta=J'-J_t,
$$

with $T=0.1$ in `run_phase12b_adaptive_search_audit.py`:

$$
p_{acc}=\begin{cases}
1,&\Delta>0,\\
\exp(\Delta/T),&\Delta\le 0.
\end{cases}
$$

Again, this is a search heuristic, not a stationary-distribution MCMC claim.

# 11. Cross-model transfer

A candidate optimized for model $A$ can be rescored by model $B$ without charging the search budget if transfer evaluation is explicitly outside the attack budget. Define

$$
J_A(x')=|f_A(x')-f_A(x)|,
$$

$$
J_B(x')=|f_B(x')-f_B(x)|.
$$

Transfer asks whether high-sensitivity candidates are architecture-specific or shared.

# 12. Framework V2: why the forensic audit mattered

The first Framework V2 developmental benchmark was invalid for three independent reasons:

1. it used a Python hash stub instead of the GraphMPNN;
2. its sources came from train/test instead of validation;
3. query accounting was broken.

Those 6–7 eV mean drifts are therefore forbidden evidence.

The repaired `GraphMPNNPredictor` binds the canonical checkpoint, canonical scaler, strict graph parsing, and auditable model-call accounting. A later `verified_run_1` is the only Framework V2 benchmark that may be cited.

# 13. Exact query accounting

A declared query budget $Q$ must count every target-model candidate evaluation consistently. The repaired contract is:

$$
q_{used}\le Q.
$$

The source prediction can be computed once and explicitly treated as uncharged if the protocol says so. All candidate scoring must flow through one counting path; otherwise a search can silently exceed its budget.

Drift efficiency can be reported as

$$
\eta_{query}=\frac{J(x^*)}{q_{used}}.
$$

# 14. Evolutionary search

Framework V2 includes population-based evolutionary search under the verified benchmark. General structure:

1. maintain a population of valid candidates;
2. score fitness with $J(x')$;
3. select high-fitness candidates;
4. mutate/recombine under the operator and validator;
5. preserve elites;
6. stop at the query budget.

It explores more broadly than greedy search but makes query accounting and duplicate control especially important.

# 15. Operator semantics are part of the threat model

The forensic audit found that Framework V2 atom substitution and Phase 12B token substitution are not identical attacks. Framework V2 can touch aromatic atoms under its operator; the Phase 12B token mutator did not mutate lowercase aromatic tokens. Therefore numerical comparisons across the two threat models must not be presented as if only the search strategy changed.

Similarly, a historical “functional group replacement” operator was renamed conceptually because the implementation behaved more like aliphatic-carbon substitution and could consume multiple independent graph edits per application.

# 16. Historical sequence-model defenses

## 16.1 Random-SMILES augmentation

For equivalent representations $x$ and $x_r$ with the same true label $y$, training can safely include both:

$$
\mathcal L_{aug}
=
\ell(f_\theta(x),y)+\ell(f_\theta(x_r),y).
$$

This is scientifically clean because the graph identity is unchanged.

## 16.2 Representation consistency

An additional consistency objective can be

$$
\mathcal L_{inv}
=
\|f_\theta(x)-f_\theta(x_r)\|_2^2.
$$

## 16.3 Chemistry-changing label-free consistency

For a chemistry-changing candidate $x_c$, the original physical label must not be inherited. A historical regularization approach instead used detached model behavior:

$$
\mathcal L_{cons}
=
\ell
\left(
 f_\theta(x_c),
 \operatorname{stopgrad}(f_\theta(x))
\right).
$$

This can reduce sensitivity but does **not** prove improved physical accuracy.

# 17. Historical two-branch specialization objectives

The project tried to separate representation invariance and chemistry sensitivity in latent space.

Property loss:

$$
\mathcal L_{prop}=\ell(f(x),y).
$$

Representation branch:

$$
\mathcal L_{inv}
=
\|z_{repr}(x)-z_{repr}(x_r)\|_2^2.
$$

Chemistry-sensitivity margin:

$$
\mathcal L_{sens}
=
\max
\left(
0,
 m-\|z_{chem}(x)-z_{chem}(x_c)\|_2
\right).
$$

A diversity penalty discouraged identical branches. The experiment did not establish semantic disentanglement; branch polarization/collapse provided a cheaper route through the objective.

# 18. Why the canonical final model needs less representation defense

Once the GraphMPNN became canonical, equivalent-SMILES invariance was solved structurally. The representation-defense question therefore moved from “how do we regularize a Transformer to ignore serialization?” to “which representation makes serialization irrelevant by construction?”

That architectural pivot is more important than any one historical defense table.

# 19. The oracle problem

For chemistry-changing candidates, define model change

$$
\Delta_G=G(x')-G(x)
$$

and unknown true physical change

$$
\Delta_T=T(x')-T(x).
$$

The physically meaningful adversarial error is

$$
E_G=|\Delta_G-\Delta_T|.
$$

A large $|\Delta_G|$ alone cannot tell whether the model is oversensitive, undersensitive, or accurate.

# 20. Oracle classes

The repository distinguishes conceptually:

- **Matched oracle**: reproduces the same physical property/protocol as the dataset;
- **Calibratable surrogate oracle**: computes a related property and requires empirical calibration;
- **Mismatched oracle**: computes a different physical target and cannot be used directly.

The exact DFT functional/basis/pseudopotential setup that generated the local polyVERSE `bandgap_chain.csv` values is unresolved. Therefore a future matched-oracle claim must begin with calibration rather than assuming PBE or HSE06.

# 21. HPC oracle software architecture

The repository now includes a backend-agnostic HPC framework with implemented contracts for:

- backend interface;
- partial Quantum ESPRESSO and VASP backends;
- structure preparation provenance;
- deterministic calibration/adversarial manifests;
- SLURM array generation;
- backend checks;
- bundle export;
- output ingestion;
- calibration MAE/RMSE/$R^2$/Pearson analysis;
- adversarial $\Delta_T$, model error, and sensitivity analysis;
- uncertainty bookkeeping.

Execution remains externally blocked by missing cluster access and physically justified quantum-chemistry inputs such as pseudopotentials and exact parameter choices.

# 22. Attack → oracle → defense loop

A scientifically complete chemistry-changing loop is:

```text
source polymer x
      |
      v
bounded candidate generation x'
      |
      v
validity + threat-model checks
      |
      v
model prediction drift ΔG
      |
      v
independent physical oracle → ΔT
      |
      v
true response error E = |ΔG - ΔT|
      |
      v
oracle-backed robust training / recalibration
```

The current project reaches the oracle handoff but does not fabricate the missing physical labels.

# 23. Final status table

| Component | Scientific status |
|---|---|
| Equivalent-SMILES control | canonical |
| Single substitution/deletion stress | canonical |
| Phase 12B bounded ≤3 search | canonical |
| 5.961 eV Phase 12 maximum | rejected |
| Framework V2 developmental hash-stub numbers | rejected |
| Framework V2 `verified_run_1` | verified under its own stated operator/budget protocol |
| Sequence-model augmentation/consistency | historical evidence |
| Chemistry-changing drift as physical error | blocked without oracle |
| HPC oracle execution | software prepared, external physical compute blocked |
