# Chemical Plausibility and Domain Constraint Validation

> [!IMPORTANT]
> **Explicit Thesis Positioning Statement**:
> *"This work is a proof-of-concept for constrained adversarial robustness in materials sequence modelling, not a claim of complete chemical realism or universally improved predictive performance."*

## 1. Domain Validation Architecture

Adversarial attacks on material sequences cannot rely on arbitrary character mutations or unconstrained latent perturbations. The module [`plausibility.py`](file:///home/mrudula/Downloads/DL_cyberbio/materials-adversarial/src/materials_adv/domain/chemistry/plausibility.py) implements `ChemicalPlausibilityValidator`, a domain-specific filter enforcing four layers of chemical constraint checking:

```
Candidate PSMILES (x')
   │
   ▼
[Layer 1: RDKit Syntax & Valence Sanitization] ──(Invalid)──► REJECT (Score: -100)
   │ (Valid)
   ▼
[Layer 2: Polymer Attachment Star '*' Balance] ──(Imbalanced)─► REJECT
   │ (Balanced)
   ▼
[Layer 3: Molecular Weight Bounds [0.5, 1.5] MW] ─(Out of Bounds)► REJECT
   │ (In Bounds)
   ▼
[Layer 4: Tanimoto Similarity Cutoff S_Tanimoto >= 0.5] ─(< 0.5)──► REJECT (Score: -50)
   │ (>= 0.5)
   ▼
ACCEPT CANDIDATE (Score: 1.0 + |f_theta(x') - f_theta(x)|)
```

---

## 2. Four-Layer Constraint Verification

### Layer 1: RDKit Syntax & Valence Sanitization
- Parses candidate SMILES string via `Chem.MolFromSmiles(candidate_smiles)`.
- Invokes `Chem.SanitizeMol(mol)` to verify valence rules (e.g. tetravalent carbon $\text{C}$, trivalent nitrogen $\text{N}$, divalent oxygen $\text{O}$), aromaticity ring kekulization, and radical electron checks.
- Candidates with hypervalent atoms (e.g. 5-bonded carbon) or broken aromatic rings are rejected immediately.

### Layer 2: Polymer Attachment Star (`*`) Balance
- In polymer repeat unit PSMILES, wildcard star atoms `*` represent polymer backbone connectivity points.
- Function `check_polymer_attachment_balance(original_smiles, candidate_smiles)` ensures that candidate sequences maintain the exact count of attachment stars present in the original polymer (typically 2 for linear polymers):
  $$\text{Count}_*(x') == \text{Count}_*(x)$$

### Layer 3: Molecular Weight Ratio Bounds ($MW$)
- Prevents extreme sequence truncation or arbitrary side-chain inflation.
- Function `check_molecular_weight_bounds()` enforces:
  $$0.5 \cdot MW(x) \le MW(x') \le 1.5 \cdot MW(x)$$
  where $MW$ is exact molecular weight in $\text{g/mol}$ computed via `rdkit.Chem.Descriptors.ExactMolWt` (or $\Delta MW \le 50\text{ g/mol}$).

### Layer 4: Tanimoto Structural Similarity Threshold ($S_{\text{Tanimoto}}$)
- Computes Morgan circular fingerprints ($\text{radius} = 2$, $2048$ bits) for $x$ and $x'$:
  $$S_{\text{Tanimoto}}(x, x') = \frac{|\mathbf{fp}(x) \cap \mathbf{fp}(x')|}{|\mathbf{fp}(x) \cup \mathbf{fp}(x')|}$$
- Enforces $S_{\text{Tanimoto}}(x, x') \ge 0.5$. Candidates falling below $0.5$ are penalized, preventing off-target perturbations that transform the material into an unrelated chemical class.

---

## 3. Scientific Spectrum of Validity vs. Synthesizability

It is critical to maintain a crisp distinction between hierarchical levels of validity:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          HIERARCHY OF CHEMICAL VALIDITY                                 │
├───────────────────────────────────────────┬─────────────────────────────────────────────┤
│ 1. Syntactic Validity                     │ RDKit parses SMILES string without error    │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ 2. Chemical Valence Validity              │ Atoms satisfy formal valence & kekulization │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ 3. Structural & Domain Similarity         │ S_Tanimoto >= 0.5, MW ratio in [0.5, 1.5]   │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ 4. Scientific Plausibility                │ Preserves bioisosteric functional groups    │
├───────────────────────────────────────────┼─────────────────────────────────────────────┤
│ 5. Real-World Synthesizability (Wet-Lab)  │ Requires reaction pathway & SA score check  │
└───────────────────────────────────────────┴─────────────────────────────────────────────┘
```

> [!WARNING]
> **Scientific Boundary**: RDKit validity and Tanimoto similarity thresholding ($S_{\text{Tanimoto}} \ge 0.5$) prove that an adversarial sequence is syntactically correct, valence-compliant, and structurally close to the source polymer. This establishes **computational plausibility** in digital SMILES space. However, RDKit parseability **does not guarantee wet-lab synthetic accessibility (SA)** or thermodynamic/kinetic stability under physical laboratory conditions. Wet-lab reaction pathway verification remains out of scope for digital computational surrogate audits.
