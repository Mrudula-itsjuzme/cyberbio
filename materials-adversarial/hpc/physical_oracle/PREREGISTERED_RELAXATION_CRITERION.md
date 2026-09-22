# PREREGISTERED RELAXATION CRITERION

This document explicitly preregisters the geometry-relaxation criteria for the PBE finite oligomer study prior to execution on a higher-memory host. This is a preregistered design choice, not a result-derived setting. It must not be tuned from future attack-pair outcomes.

## Criteria
- **Force Threshold (`forc_conv_thr`)**: 1.0D-3 Ry/Bohr
- **Total Energy Threshold (`etot_conv_thr`)**: 1.0D-4 Ry
- **Stopping Condition**: The structural relaxation is considered converged when all components of all forces are less than the force threshold AND the change in total energy between successive steps is less than the energy threshold.
- **Action on Failure**: If the maximum number of steps (`nstep=100`) is reached without convergence, the structure is marked as `FAILED_RELAXATION`. No fallback to looser criteria is permitted.

## Rationale
These values represent a conservative standard for Quantum ESPRESSO PBE calculations on finite oligomers, ensuring precise forces for reliable HOMO-LUMO gap extraction without imposing impossible tolerances for soft polymeric degrees of freedom.
