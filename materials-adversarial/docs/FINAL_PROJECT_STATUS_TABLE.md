# Final Project Status Table

| Component | Status | Canonical Evidence | Limitations | Next Required Action |
| :--- | :--- | :--- | :--- | :--- |
| Dataset Provenance | SUPPORTED | Exact 2D repeat units identified | Original 3D geometries and DFT configuration missing | Retrieve source metadata if possible |
| Transformer Baseline | COMPLETE | `FINAL_RESULTS.md` | Limited generalizability to sequence artifacts | N/A |
| Two-branch Experiment | REJECTED | Semantic hypothesis invalidated | High equivalent-SMILES variance | Adopt permutation-invariant architecture |
| Representation Robustness | COMPLETE | GraphMPNN equivalent-SMILES drift ≈ 0 eV | Does not protect against chemical edits | N/A |
| Adversarial-collapse Study | COMPLETE | Failed min-max training | Post-hoc collapse | Oracle-backed dynamic training |
| GraphMPNN Selection | COMPLETE | Highest clean validation MAE | Only tested on specific bandgap regression | N/A |
| Fixed Chemistry Stress | COMPLETE | Structurally removed | - | N/A |
| Bounded Adaptive Attack | COMPLETE | Large prediction changes found | Cannot confirm if changes are physical | Physical evaluation via oracle |
| Cross-model Transfer | SUPPORTED | Weak/moderate transfer of sensitivities | Transferability does not equal physical truth | Physical evaluation via oracle |
| Oracle Feasibility | SUPPORTED | Integration framework built | Missing HPC compute | Provision HPC access |
| Structure Reconstruction | SUPPORTED | ETKDG + MMFF94 works for 2D topologies | 3D polymer entanglement not modeled | Reconstruct true solid-state cell |
| Surrogate Geometry | COMPLETE | Transparent `CALIBRATABLE_SURROGATE` | Capped oligomers instead of periodic boundary | Calibrate against reference data |
| HPC Bundle | COMPLETE | Secure execution package generated | Unverified on real cluster | Transfer and run `verify_cluster_environment.py` |
| Real QC Calibration | BLOCKED | `BACKEND_EXECUTION_BLOCKED` | No local SLURM/QE | Provide HPC access |
| Oracle-backed Adversarial Validation | BLOCKED | Awaiting calibration success | Requires calibrated surrogate | Run after calibration pilot passes |
| Attack→Defend Loop | FUTURE | - | Entirely blocked by oracle unavailability | Implement post-calibration |
