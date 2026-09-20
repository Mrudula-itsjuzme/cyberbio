# THREAT MODEL

## 1. Attacker Goal
The primary goal of the attacker is to find adversarial sequence perturbations (PSMILES strings) that maximize the prediction error (prediction drift) of a trained Transformer model predicting polymer bandgap (E_g in eV), without changing the underlying ground-truth property beyond a reasonable boundary, or to expose sensitivity to semantically equivalent representations.

## 2. Attacker Knowledge
The framework primarily assumes a **White-Box** or **Grey-Box** threat model where the attacker has access to the model's prediction function to compute gradients (for attribution/guidance) or to repeatedly query the model to evaluate fitness (e.g., in MCMC or evolutionary search).

## 3. Permitted Transformations
- **Representation-Preserving**: Randomizing the SMILES string (generating different valid traversal paths of the same molecular graph).
- **Chemistry-Changing**: Single-token substitutions, insertions, deletions, and local rearrangements (MCMC / Search).

## 4. Constraint Verification Table

| Constraint | Implemented? | Code path | Exact check | Notes |
| :--- | :--- | :--- | :--- | :--- |
| RDKit parsing | Yes | `src/materials_adv/validation/representation.py` | `Chem.MolFromSmiles(psmiles, sanitize=True)` | Fails string if it cannot be parsed. |
| Attachment star preservation | Yes | `src/materials_adv/validation/plausibility.py` | `n_attach == 2` | Requires exactly two attachment points. |
| Ring protections | Yes | `src/materials_adv/attacks/token_space.py` | `protect_ring_closures=True` | Excludes ring digits from `editable_positions`. |
| Branching protections | Yes | `src/materials_adv/attacks/token_space.py` | `protect_branches=True` | Excludes `()` from `editable_positions`. |
| Molecular-weight bounds | Yes | `src/materials_adv/domain/chemistry/plausibility.py` | `enforce_mw_bounds` | Checked during MCMC/Domain generation. |
| Tanimoto threshold | Yes | `src/materials_adv/domain/chemistry/attacks/probabilistic.py` | `compute_tanimoto_similarity(...) < min_tanimoto` | Penalizes/rejects MCMC candidates. |
| Attack budget | Yes | `src/materials_adv/attacks/base.py` | `attack_budget` | Passed to generator logic. |
| Token protections | Yes | `src/materials_adv/attacks/token_space.py` | `editable_positions()` | Defines what indices can be mutated. |

## 5. Success Criterion
The attack is considered successful if it maximizes the absolute prediction drift while remaining within the defined validity constraints.

## 6. Perturbation Budget
For chemistry-changing attacks, the perturbation budget (`attack_budget`) is strictly constrained (typically 1-3 token changes) to prevent the molecule from drifting completely out of the domain.

### Physical Oracle Clarification
These checks reduce invalid or unconstrained sequence corruption but do not constitute quantum-chemical verification of unchanged physical properties.
