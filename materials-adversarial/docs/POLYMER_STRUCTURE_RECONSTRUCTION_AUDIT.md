# Polymer Structure Reconstruction Audit

## Objective
To determine if the 1D periodic chain structures in the `bandgap_chain.csv` canonical dataset can be defensibly reconstructed for Oracle (Quantum ESPRESSO/VASP) physics calculations.

## Methodology
1. Audited the representation of molecules in the source datasets.
2. Verified topological unambiguousness using RDKit for exactly two `[*]` attachment points.
3. Inspected the repository and dataset distributions for physical metadata (coordinates, lattice parameters, DFT functional configurations).

## Findings
- **Topological Viability:** The 1D polymer chains are unambiguous at the 2D topology level (connectivity). For example, `[*]CC([*])` uniquely denotes a repeating ethylene unit.
- **Physical Geometry Viability:** There are no raw geometry files (`.cif`, `.xyz`, or `POSCAR`) in the repository.
- **DFT Parameters:** The exact Quantum ESPRESSO or VASP parameters (e.g., cutoff energies, k-point sampling, pseudo-potentials, vacuum padding for 1D chains) are not provided in the repository.
- **Conclusion:** Any 3D structure generated locally would be an assumption. Conformational searching, geometry optimization, and lattice scaling would introduce uncontrolled degrees of freedom that deviate from the dataset's exact physical ground truth.

## Resulting Action
The integration of physical physics Oracles is paused from acting as a "ground truth replacement". Rather, any physics validation done locally must be categorized as a **Surrogate Oracle**.

To resolve this limitation, we will prepare a data request to the original authors for the specific structure files.
