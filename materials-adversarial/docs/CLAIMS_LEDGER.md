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
