# Project Question and Decision Log

This is a chronological decision ledger for quick review before presentations and meetings.

---

**Date/Phase:** Phase 1 (Baseline)
**Question:** Can a Transformer predict polymer Bandgap reasonably well from PSMILES sequences?
**Hypothesis:** A character-level Transformer regressor can learn the mapping from syntax to scalar Bandgap values.
**Evidence:** The Ordinary Baseline achieved a Validation MAE of ~0.460 eV on the polyVERSE dataset.
**Decision:** The model performs reasonably well. Proceed to adversarial testing.
**Next action:** Generate adversarial representation attacks (Phase 2).

---

**Date/Phase:** Phase 2 (Adversarial Testing)
**Question:** Is the model sensitive to representation perturbations?
**Hypothesis:** The model overfits to the canonical SMILES syntax and will fail when presented with alternative but equivalent syntax or minor chemistry edits.
**Evidence:** Randomization drift was 0.614 eV; substitution drift was 0.291 eV. The model is highly fragile.
**Decision:** We need a defense mechanism.
**Next action:** Design an architecture intended to learn robustness.

---

**Date/Phase:** Phase 3 (Architecture & Specialization)
**Question:** Can we assign internal branches to different robustness concepts?
**Hypothesis:** An explicit two-branch architecture with targeted loss functions can force one branch to learn representation-invariance (for randomization) and the other to learn chemistry-sensitivity (for substitutions).
**Evidence:** The branches polarized in variance (Branch A variance collapsed to ~0.036), but selectivity ratios showed no actual semantic specialization. The architecture control model achieved similar baseline robustness simply by having more parameters.
**Decision:** Reject the semantic branch specialization hypothesis.
**Next action:** Abandon internal latent specialization and move the robustness constraints to the output layer.

---

**Date/Phase:** Phase 4 (Output-Level Robustness)
**Question:** Does output-level randomization consistency improve robustness beyond architecture?
**Hypothesis:** Penalizing the difference between predictions on clean and randomized inputs directly at the final output will improve robustness without causing representation collapse.
**Evidence:** Randomization drift improved from 0.382 eV (Architecture Control) to 0.301 eV (Randomization-Robust). Clean Validation MAE actually improved (0.461 -> 0.443 eV).
**Decision:** Adopt output-level prediction consistency training.
**Next action:** Test if teacher distillation can similarly fix substitution drift.

---

**Date/Phase:** Phase 4 (Mixed Robustness)
**Question:** Does substitution teacher-consistency provide an additional benefit?
**Hypothesis:** Penalizing the difference between the model's prediction and a frozen teacher's prediction on substitution candidates will prevent erratic extrapolation on chemistry-changing edits.
**Evidence:** Substitution drift improved from 0.281 eV to 0.269 eV (95% CI: `[0.003, 0.030]`), while randomization drift and clean MAE remained excellent.
**Decision:** Adopt mixed-robustness (randomization consistency + teacher distillation) as the final pipeline.
**Next action:** Freeze documentation and prepare for unseen transfer attacks (Phase 5).

---

**Date/Phase:** Phase 5 (Future Transfer)
**Question:** Does robustness learned from randomization + substitution transfer to an unseen attack like deletion?
**Hypothesis:** The model has learned general stability that will mitigate extreme drift on deletion candidates.
**Evidence:** *Not yet tested.*
**Decision:** *Pending Phase 5 execution.*
**Next action:** Generate deletion candidates and evaluate frozen Phase 4 models.
