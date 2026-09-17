# Data Request: 1D Periodic Chain Polymer Structures

**To:** Authors of the polyVERSE dataset (Ramprasad Group)
**Subject:** Inquiry Regarding Exact 1D Periodic Geometries for `bandgap_chain.csv`

Dear Authors,

We are conducting an independent robustness audit on materials representation models and have been working extensively with the `bandgap_chain.csv` dataset from the polyVERSE repository.

To perform adversarial physical validation against our models, we are seeking to run new Quantum ESPRESSO / VASP calculations on slightly modified variants of the dataset's polymer chains. 

To ensure our baseline physical calculations exactly match the dataset's reported DFT bandgaps, we would like to request the original 3D geometry files (e.g., `.cif`, `.xyz`, or `POSCAR`) used for the 1D periodic chain calculations. The current repository provides 2D wildcard SMILES (e.g., `[*]CC([*])`), which allow for unambiguous topological graphs but leave open numerous degrees of freedom for 3D conformational optimization and lattice parameter scaling.

If the structures are available, could you please advise on how to access them? Alternatively, if there is a specific algorithmic protocol or script (including vacuum sizes, cutoff energies, and k-point mesh choices) used to generate these 3D geometries from the 2D SMILES, we would greatly appreciate that information.

Thank you for your time and for providing this valuable resource to the materials informatics community.

Sincerely,
[Your Name/Team]
