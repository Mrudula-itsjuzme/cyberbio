# Generalizing the Adversarial Framework

To extend this framework to a new domain, the following components must be supplied:
- **RepresentationAdapter**: Converts domain objects to model inputs.
- **ValidityChecker**: Ensures the candidate is valid in the domain.
- **ConstraintSet**: Ensures edits respect the threat model.
- **AttackOperator**: Generates candidate edits.
- **Predictor**: The target ML model.
- **OptionalOracle**: The physical ground-truth verifier.

## Examples
### CHEMISTRY
- **Representation**: SMILES / graph
- **Validity**: RDKit + valence
- **Operators**: substitution, motif swap, scaffold-preserving edit
- **Oracle**: DFT / calibrated property reference

### DNA FUTURE
- **Representation**: nucleotide sequence
- **Validity**: alphabet + biological constraints
- **Operators**: base substitution / motif edit
- **Oracle**: experimental or validated biological reference

### PROTEIN FUTURE
- **Representation**: amino-acid sequence / structure
- **Validity**: sequence/structural constraints
- **Operators**: residue substitution / motif edit
- **Oracle**: experimental/biophysical reference
