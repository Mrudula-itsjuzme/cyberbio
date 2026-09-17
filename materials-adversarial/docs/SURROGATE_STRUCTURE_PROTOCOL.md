# Surrogate Structure Protocol

## Scope and Purpose

The original 3D periodic geometry for the `bandgap_chain.csv` dataset is unavailable, and the exact DFT protocols (functionals, vacuum padding, pseudopotentials) used to generate them are incompletely known. The dataset provides wildcard PSMILES which robustly support 2D repeat connectivity, but do not encode 3D conformational structure. 

Therefore, any 3D structures generated locally are constructed independently. Their purpose is strictly for **calibration** and as a **relative physical reference** for the adversarial modifications. 
They are **NOT** replicas of the original polyVERSE structures.

## Construction Modes

This protocol explicitly separates two modes of geometry construction, which must never be silently mixed:

### 1. CAPPED_OLIGOMER
Builds finite $n$-repeat oligomers from the validated repeat topology.
- Supports configurable lengths ($n = 2, 3, 4$).
- Caps terminal attachment sites based on a simple declared capping rule (e.g., `[H]`).
- Persists explicit properties including length, rule, charge, multiplicity, and structural warnings.

### 2. PERIODIC_CHAIN_SURROGATE
Constructs an explicit repeating-chain topology with one connection across a periodic boundary.
- Decouples topological periodicity from 3D lattice parameters.
- **Requires** explicit user/configuration for the chain-axis cell length, vacuum dimensions, and orientation convention. 
- Automatically fails if arbitrary or missing cell dimensions are detected.

## 3D Embedding

- We use deterministic conformer generation via RDKit `ETKDG`.
- Random seeds and embedding statuses are explicitly recorded.
- Multiple conformers are generated (e.g., 5-10) to sample the phase space.

## Preoptimization

Initial embeddings are relaxed using Force-Fields.
- Supports `MMFF94` with a fallback to `UFF`.
- Force-field optimized structures are **NOT** to be called DFT-quality geometries.

## Structural Validation

Every output structure must preserve:
- Atomic valence
- Graph connectivity and original topology
- No remaining wildcard atoms
- Proper terminal capping (if applicable)
- Disconnected fragment absence

Structures failing validation are logged with failure modes and dropped.
