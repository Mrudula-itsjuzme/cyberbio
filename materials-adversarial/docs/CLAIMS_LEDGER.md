
# SCIENTIFIC STATUS UPDATE (2026-09-22)
**Final Local Status:** CONVERGENCE_FAILED
**Reason:** Frozen convergence protocol (12 Å vacuum) exceeds the practical memory envelope of the local 15 GiB host (exhausted at 80/360 Ry on the 22-atom C/H smoke structure before SCF).

**Methodological State:**
- No protocol freeze occurred.
- No 12-polymer calibration was run.
- No attack-pair DFT was run.
- No DFT sign-agreement or magnitude-error results exist.
- No chemistry-changing drift is established as physical error.
- The finite capped-oligomer PBE HOMO-LUMO gap is not validated as the historical periodic-chain target.
- The 8 Å C/H smoke calculation is infrastructure evidence only, not calibration evidence.
- Frozen signed model predictions exist for four attack pairs (all negative) and remain unchanged.
- Test Suite: 462 passed, 1 skipped.
- Provisioning (QE 7.6 + SSSP 1.3.0) is verified.
- **Next defensible action:** Run the unchanged convergence design on a higher-memory host (after preregistering relaxation force threshold).

*(Note: Any prior statements regarding "QE unavailable" are superseded by this updated status.)*

---

# Scientific Claims Ledger

This ledger tracks the hypotheses generated during the project and their final evidentiary status based on rigorous audits.

| Claim | Status | Evidence | Limitations | Canonical Source |
| :--- | :--- | :--- | :--- | :--- |
| Sequence models natively learn chemical equivalence | **REJECTED** | Ordinary Transformer drifted ~0.614 eV on Equivalent-SMILES edits. | - | Phase 1 |
| Branch-specialization fixes equivalent-SMILES drift | **REJECTED** | Two-Branch control drifted identically to the ordinary baseline (~0.380 eV). | - | Phase 2 |
| Post-hoc representation adversarial training yields robustness | **REJECTED** | Multi-phase adversarial retraining caused catastrophic representation collapse (MAE exploded to ~2.0 eV). | - | Phase 9, 10 |
| GNN representations eliminate equivalent-SMILES drift | **SUPPORTED** | GraphMPNN equivalent-SMILES drift is strictly zero by structural definition. | - | Phase 11C, 12 |
| Chemistry-changing edits generate bounded prediction drift | **SUPPORTED** | A ≤3 edit budget capped maximum response to ~3.19 eV. | Local optima may exist. | Phase 12B |
| Original 5.961 eV drift is a true adversarial response | **REJECTED** | Derived from unbounded edit budget inflation causing non-physical fragmentation. | - | Phase 12 |
| Phase 11B GNN drift numbers were accurate | **REJECTED** | Missing padding_mask evaluation bug artificially inflated Transformer MAE to ~1.85 eV. | - | Phase 11B/C |
| Prediction drift on chemistry changes equates to absolute error | **BLOCKED** | QE/SSSP provisioning and a smoke test succeeded, but convergence failed within the local memory envelope. No calibration or paired DFT result exists, so true target response remains unknown. | Requires higher-memory frozen convergence, protocol freeze, and calibration before pair execution. | Physical-oracle gate 2026-09-22 |
| Target dataset generated via HSE06/PBE | **NOT_TESTED** | Speculative literature assumption. Exact metadata for `bandgap_chain.csv` is absent. | Must calibrate empirical oracle. | Phase 13A |
