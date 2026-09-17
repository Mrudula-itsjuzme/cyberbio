# Model Explainability and Adversarial Attribution Analysis

> [!IMPORTANT]
> **Core Question**: *Why does the surrogate model prediction move under an adversarial sequence modification? What sequence features or artifacts trigger prediction drift?*

---

## 1. Interpretability Methodologies

To analyze the underlying mechanics of model sensitivity, we apply four complementary interpretability techniques:

1. **Perturbation Attribution**: Systematic single-token occlusion/deletion $\Delta y = f_\theta(x) - f_\theta(x_{\setminus i})$ to measure per-token prediction contribution.
2. **Gradient-Based Token Attribution**: Computing $\| \nabla_{\mathbf{e}_i} f_\theta(x) \|_2$ w.r.t. input token embeddings $\mathbf{e}_i$.
3. **Sequence Length Dependency Analysis**: Measuring target prediction sensitivity as a function of token sequence length changes ($\Delta L$).
4. **Attention Pattern Inspection**: Extracting multi-head self-attention weights $\mathbf{A}^{(l, h)} \in \mathbb{R}^{L \times L}$ to visualize token-token interaction matrices (noting that attention alone is not causal explanation).

---

## 2. Key Explainability Discoveries

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                EXPLAINABILITY FINDINGS SUMMARY                             │
├───────────────────────────┬───────────────────────────────────────────────────────────────┤
│ Feature Category          │ Observed Model Behavior                                       │
├───────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Polymer Stars ('*')       │ Disproportionately high embedding gradient magnitude          │
│ Sequence Length Shift     │ Strong negative linear bias (-0.033 eV per character)         │
│ Halogen Substitutions     │ High sensitivity to -F -> -Cl -> -Br electron-withdrawing swaps│
│ Aromatic Conjugation      │ Ring scission/deletion causes large step drops in bandgap     │
└───────────────────────────┴───────────────────────────────────────────────────────────────┘
```

### Discovery A: Attachment Star (`*`) Hyper-Sensitivity
Polymer repeat unit wildcard stars `*` mark the chain linkage points. In the embedding gradient trace, `*` tokens exhibit **$2.4\times$ higher average gradient norms** than standard carbon `C` tokens. Because `*` occurs at sequence boundaries, positional encodings amplify its feature weight.

### Discovery B: Sequence Length Bias Exploitation
As established in `docs/LEAKAGE_AUDIT.md`, sequence length negatively correlates with target bandgap ($r = -0.4310$). Perturbations that lengthen the SMILES string (e.g. inserting aliphatic chains) artificially push the model prediction toward lower bandgap values ($E_g$), exploiting the length shortcut.

### Discovery C: Halogen and Heteroatom Perturbation Cascades
Replacing fluorine (`F`) with chlorine (`Cl`) or bromine (`Br`) in aromatic side-chains induces localized prediction shifts of $0.15 - 0.35\text{ eV}$, matching physical expectations of altered electronegativity and HOMO level pinning.

---

## 3. Concrete Case Studies

### Case Study 1: Halogen Substitution on Conjugated Backbone
- **Clean PSMILES ($x$)**: `[*]c1cc(F)c(c2ccc([*])s2)s1` ($E_g = 2.45\text{ eV}$)
- **Adversarial PSMILES ($x'$)**: `[*]c1cc(Cl)c(c2ccc([*])s2)s1` ($E_g = 2.18\text{ eV}$)
- **Drift**: $\Delta E_g = 0.27\text{ eV}$
- **Attribution Trace**: Token gradient for `F` $\to$ `Cl` position accounted for 64% of total gradient norm. The model correctly identifies heteroatom modification, but overestimates the bandgap shift.

### Case Study 2: Side-Chain Deletion and Length Shift
- **Clean PSMILES ($x$)**: `[*]C=C([*])c1ccc(CCCC)cc1` ($E_g = 3.82\text{ eV}$)
- **Adversarial PSMILES ($x'$)**: `[*]C=C([*])c1ccc(C)cc1` ($E_g = 4.15\text{ eV}$)
- **Drift**: $\Delta E_g = 0.33\text{ eV}$
- **Attribution Trace**: Sequence length shortened by 3 characters (`CCCC` $\to$ `C`). The prediction shift aligns with the linear length bias slope ($\Delta L = -3 \implies +0.10\text{ eV}$ length contribution + $0.23\text{ eV}$ electronic contribution).
