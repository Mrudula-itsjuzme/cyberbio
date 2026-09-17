# Paper Contributions

1. **Rigorous Separation of Perturbation Types**
   We establish a formal separation between representation-preserving perturbations (equivalent SMILES syntax changes) and chemistry-changing perturbations (substitutions/deletions), demonstrating that robustness claims in material informatics must explicitly delineate which perturbation class is being addressed.

2. **Empirical Analysis of Representation Vulnerabilities**
   We empirically demonstrate that sequence models (Transformers) can be highly sensitive to equivalent SMILES serialization, while graph-based representations (GraphMPNN) remove serialization dependence by mathematical construction (reducing representation drift from 0.614 eV to ≈0 eV).

3. **Systematic Comparison of Robustness Interventions**
   We evaluate multiple robustness interventions—including data augmentation, architecture control, and adversarial fine-tuning—highlighting failed and adverse defense outcomes such as post-hoc adversarial collapse during min-max training.

4. **Bounded Chemically Valid Adaptive Attack Framework**
   We propose and implement a strict, chemically valid, adaptive attack search framework (using bounded graph edits) and provide cross-model transfer analysis, revealing that model sensitivities to specific chemical motifs are shared across different architectures.

5. **Oracle-Aware Adversarial Evaluation**
   We demonstrate that chemistry-changing prediction drift cannot be automatically equated with adversarial error without independent physical supervision. We outline an oracle-aware framework and a reproducible surrogate-QC protocol to formally bridge the gap between ML prediction drift and physical truth.

## What We Can Claim

* GraphMPNN achieves better clean validation performance than the evaluated Transformer baselines under the canonical protocol.
* Graph representation makes equivalent-SMILES serialization changes invariant up to numerical precision.
* Chemically valid bounded edits can induce substantial GraphMPNN prediction changes under the tested search protocol.
* Cross-model transfer indicates that some discovered sensitivities are shared across model classes.
* The current evidence cannot determine whether chemistry-changing prediction drift corresponds to physical change or model error.

## What We Cannot Claim

* Chemistry attacks are proven adversarial failures.
* 3.19 eV is a confirmed physical prediction error.
* GraphMPNN is universally robust.
* Lower chemistry drift is always better (it may indicate over-smoothing of physically distinct materials).
* The surrogate structures precisely reproduce the original polyVERSE dataset geometries.
* PBE/SSSP physics settings match the original dataset's exact protocol.
* True chemistry-changing attack→defend training has been completed (blocked by oracle access).
* Oracle calibration succeeded (the pilot remains unexecuted).
