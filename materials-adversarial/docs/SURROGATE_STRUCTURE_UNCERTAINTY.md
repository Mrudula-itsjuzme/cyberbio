# Surrogate Structure Uncertainty

Because the 3D structures generated are explicitly surrogate models and not exact reconstructions of the polyVERSE dataset geometries, we must quantify and propagate structural uncertainty through the oracle pipeline. 

This ensures that downstream QC results are never mistakenly conflated with absolute ground truth.

## Uncertainty Components

The surrogate generation step records the following axes of uncertainty for each polymer structure:

### 1. Conformer Spread
We sample the conformational phase space by generating an ensemble of structures via ETKDG and optimizing them with MMFF94/UFF. The uncertainty field must record:
- Total number of generated conformers.
- The spread (range and standard deviation) of the final force-field minimized energies.
- The energy of the selected conformer relative to the ensemble.

### 2. Oligomer-Length Dependence
For the `CAPPED_OLIGOMER` mode, the exact polymer is approximated by a finite $n$-mer. By generating $n=2$, $n=3$, and $n=4$ variants, the variation in subsequent QC observables (e.g., HOMO-LUMO gap scaling) represents length-dependent uncertainty.

### 3. Capping Dependence
Terminating a wildcard attachment point with a cap (e.g., Hydrogen) introduces chemical boundaries that don't exist in an infinite chain. This introduces charge/polarization uncertainty at the termini.

### 4. Periodic-Cell Assumptions
For `PERIODIC_CHAIN_SURROGATE` modes, since original lattice vectors (chain-axis length) and vacuum padding sizes are missing, any assumed box size introduces structural strain and electrostatic interactions across vacuum. This is a severe source of uncertainty if the surrogate lattice dimension differs from the dataset's native relaxed dimension.

### 5. Force-Field Dependence
We record whether MMFF94 or UFF was used. Since these are empirical parameters, they do not perfectly match the DFT potential energy surface.

## Propagation into the Oracle

The `surrogate_structure_uncertainty` must be explicitly appended to the QC execution manifests. 

Future calibration protocols will require measuring:
1. **Absolute Alignment**: Does the Surrogate QC bandgap correlate at all with the Dataset Bandgap?
2. **Relative Stability**: Are the conclusions regarding bandgap changes under adversarial modifications stable regardless of the surrogate construction choices? 
