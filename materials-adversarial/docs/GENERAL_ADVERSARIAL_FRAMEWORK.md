# General Adversarial Framework

The repository is structured around a modular, domain-agnostic architecture designed to decouple the generation and verification of adversarial edits from the underlying data representation. 

## Generic Architecture Pipeline

1. **RepresentationAdapter**: Converts raw domain data (e.g., SMILES strings) into a manipulatable mathematical object (e.g., a PyTorch Geometric molecular graph).
2. **AttackOperator**: Defines the allowable bounds of atomic transformations (e.g., single-node substitution, edge deletion).
3. **ConstraintSet**: Imposes strict domain rules (e.g., valence limits, charge conservation) that prevent physically impossible geometries.
4. **ValidityChecker**: Mathematically guarantees that the output of the `AttackOperator` satisfies the `ConstraintSet`.
5. **SearchStrategy**: Navigates the edit space (e.g., iterative greedy search) to locate modifications that maximize loss.
6. **Predictor**: The ML model under test (e.g., GraphMPNN or Transformer).
7. **OptionalOracle**: An independent physical truth-engine (e.g., Quantum ESPRESSO) to verify whether prediction drift reflects model error or accurate physical change.
8. **Evaluation**: Aggregates metric drift, bounds tracking, and robustness scoring.

## Current Chemistry Implementation

In the current scope, the framework is instantiated specifically for polymer bandgap prediction:
* **Adapter**: `materials_adv.graph_utils` converting SMILES to PyTorch Geometric `Data` objects.
* **Validator & Constraints**: `materials_adv.chemistry` enforcing RDKit valency and connectivity.
* **Operators**: Bounded substitutions (`C`, `N`, `O`, `S`, etc.) and deletions, maintaining structural integrity.
* **Oracle**: `CALIBRATABLE_SURROGATE` geometry mapping to Quantum ESPRESSO plane-wave DFT.

## Future Domain Adapters (e.g., DNA / Proteins)

The architecture is designed to swap domains seamlessly. For a future genomic expansion (e.g., adversarial DNA regulatory sequences):
* **Representation**: Swap graphs for nucleotide one-hot encoding or k-mer tokenization.
* **Validator**: Swap RDKit valency checks for motif-preservation or reading-frame alignment checks.
* **Constraints**: Enforce biological validity (e.g., GC content bounds, non-destructive start/stop codons).
* **Operators**: Sequence transitions, transversions, and strictly bounded indels.

Crucially, **we reuse the core logic**:
* Search trajectories and budget accounting.
* Metric tracking (e.g., prediction drift, MAE calculation).
* Logging and experiment pipelines.
* Oracle comparison logic (where the DNA oracle might be an external binding-affinity simulation or specialized biological API rather than DFT).
