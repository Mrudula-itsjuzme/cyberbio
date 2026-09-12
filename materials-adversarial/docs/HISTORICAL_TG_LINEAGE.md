# Historical Lineage Document: OpenPoly Glass-Transition Temperature ($T_g$)

**Status**: Historical / Exploratory Lineage (Superseded by active polyVERSE Bandgap project)

---

## 1. Overview & Dataset
- **Target Property**: Polymer Glass-Transition Temperature ($T_g$).
- **Units**: Kelvin (K).
- **Dataset Source**: OpenPoly dataset.
- **Usable Records**: 247 clean samples after deduplication, RDKit canonicalization, and conflict resolution.
- **Split Structure**: Scaffold split (204 train / 37 validation / 6 sealed test).

---

## 2. Experimental Progression & Key Findings

### Phase 1: Initial Attacks & The Length Hypothesis
- **Baseline Model**: Small Transformer encoder predicting $T_g$ in Kelvin. Clean Test MAE on sealed 6-sample test set: **52.02 K** (Val MAE: **79.04 K**).
- **Initial Observation**: Insertion and deletion attacks (length-changing) produced large prediction drifts (~30 K), whereas single substitution and local rearrangement (length-preserving) produced smaller drifts (~8 K).
- **Initial Working Hypothesis**: "Length-changing token attacks are uniquely destructive to polymer sequence models."

### Phase 2: Defense Ablations & Multi-Seed Replication
- **Phase 2A (Adversarial Training)**: Appending adversarial variants to training appeared to improve clean Test MAE to 46.20 K and reduce max deletion drift from 153.9 K to 81.0 K.
- **Phase 2B (Scaler Audit)**: Audit revealed that Phase 2A's clean MAE shift was primarily a confounder caused by `TargetScaler` re-fitting on augmented target distributions. When the clean target scaler was strictly frozen, only 1.38 K of the clean improvement survived.
- **Phase 2C (5-Seed Replication)**: Multi-seed replication across seeds `20260815`–`20260819` showed zero net clean MAE improvement (+0.81 ± 4.58 K Val MAE) and un-reproducible mean drift reductions. The only consistent finding was a reduction in worst-case (maximum) drift.

### Phase 2F: Forensic Architecture Audit (Scientific Refutations)
A rigorous forensic audit disproved the core hypotheses of the early $T_g$ work:
1. **Refutation of Length Hypothesis**: Length-preserving controls (shuffling or reversing sequence tokens) produced prediction drifts of **30.44 K to 32.05 K**, matching or exceeding insertion/deletion drifts. The model was globally sensitive to token restructuring, not specifically length-sensitive.
2. **Length Shortcut Learning**: A 1-parameter linear regression predicting $T_g$ solely from sequence length achieved a Validation MAE of **76.77 K**, outperforming the Transformer's **79.04 K**. The model had learned sequence length as a shortcut rather than physical chemistry.
3. **Invalid Target Preservation**: Assuming insertion/deletion/substitution preserved the original $T_g$ target was physically false. SMILES randomization (`Chem.MolToSmiles(..., doRandom=True)`) was identified as the true representation-preserving control.

---

## 3. Lessons Learned & Scientific Corrections
1. **Small Sample Limits**: The 247-sample $T_g$ dataset was too small for high-dimensional Transformer representation learning without severe shortcut reliance.
2. **Scaler Hygiene**: Retraining scalers on augmented datasets creates artificial metric shifts; target scalers must remain strictly frozen across baseline and defended evaluations.
3. **Semantic Distinction**: Atomic token mutations change stoichiometry or molecular formula and must be treated as **stress tests**, not label-preserving augmentations.

---

## 4. Preservation Notice
All code, scripts, and raw result artifacts for the historical $T_g$ experiments remain intact in `materials-adversarial/results/` and historical documentation (`docs/PROJECT_WORKSPACE.md`, `docs/ARCHITECTURE_FORENSIC.md`). They are preserved strictly as exploratory lineage context and must not be mixed into active polyVERSE Bandgap configurations or evaluation pipelines.
