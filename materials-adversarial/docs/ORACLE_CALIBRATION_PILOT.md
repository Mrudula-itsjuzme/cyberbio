# Oracle Calibration Pilot Status

Based on the structure-provenance investigation, the exact representations of the 1D periodic polymers from the `bandgap_chain.csv` dataset are limited to 2D wildcard SMILES graphs (e.g., `[*]CC([*])`).

There are no `.cif`, `POSCAR`, or `.xyz` files in the dataset repository. There are also no automated 3D builder scripts or explicit DFT geometry parameters (vacuum padding, cutoffs, functionals) specified for the dataset samples.

Therefore, the pilot cannot be run using identical geometries to those used by the original dataset authors.

**Verdict:** `SURROGATE_STRUCTURE_ONLY`

Any Quantum ESPRESSO or VASP calculations performed using structures generated from this repository will necessarily be surrogate calculations (approximate physical models), rather than exact reproductions of the dataset's ground truth.
