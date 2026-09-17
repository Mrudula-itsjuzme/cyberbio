# Architecture Walkthrough: End-to-End Tracing of a Polymer Sequence

## 1. Tracing Example Sequence

We trace a single polymer repeat unit through the complete attack, validation, prediction, and defensive update pipeline:
$$\text{Source PSMILES } (x): \text{ "[*]c1ccc([*])cc1" (Poly(p-phenylene))}$$
$$\text{Ground-Truth Bandgap } (y): 3.20 \text{ eV}$$

---

## 2. Step-by-Step System Execution Trace

```
1. INPUT REGISTRATION
   Source PSMILES: x = "[*]c1ccc([*])cc1"
   Attachment Stars (*): 2 stars present (Polymer repeat unit)

2. PSMILES TOKENIZATION
   PSmilesTokenizer.tokenize(x) -> ['[*]', 'c', '1', 'c', 'c', 'c', '([*])', 'c', 'c', '1']
   Sequence Length: L = 10 tokens

3. EMBEDDING & TRANSFORMER ENCODER
   Token Indices -> Embedding E_tok + 1D Positional E_pos -> Tensor X [1, 10, 64]
   Transformer Encoder Layer 1 & 2 (Multi-Head Self-Attention, N_h=4)
   Masked Mean Pooling -> Pooled Embedding Vector h in R^64

4. TWO-BRANCH SPECIALIZED HEAD & REGRESSION
   Branch A (f_repr): z_repr = LayerNorm(ReLU(h W_A)) in R^32
   Branch B (f_chem): z_chem = LayerNorm(ReLU(h W_B)) in R^32
   Concat Fusion: z_fused = [z_repr || z_chem] in R^64
   Linear Regressor -> Scaled output y_scaled = 0.421
   TargetScaler Inverse Transform -> f_theta(x) = 3.18 eV (Clean Error: |3.18 - 3.20| = 0.02 eV)

5. ATTACK GENERATION (ProbabilisticMCMCAttack)
   Select Proposal: Bioisosteric Functional Group Swap (Halogen -F for aromatic H)
   Proposed Candidate (x'): "[*]c1cc(F)c([*])cc1" (2-fluoro-1,4-phenylene)

6. CHEMICAL PLAUSIBILITY VALIDATION (ChemicalPlausibilityValidator)
   - Layer 1: RDKit Parse & Sanitize Chem.MolFromSmiles(x') -> VALID
   - Layer 2: Attachment Star Count -> Count(*)(x') = 2 == Count(*)(x) = 2 -> PASSED
   - Layer 3: MW Ratio Check -> MW(x') / MW(x) = 94.08 / 76.09 = 1.236 in [0.5, 1.5] -> PASSED
   - Layer 4: Tanimoto Similarity -> S_Tanimoto(x, x') = 0.625 >= 0.5 -> PASSED

7. ADVERSARIAL SCORING & DRIFT MEASUREMENT
   Model Evaluation: f_theta(x') = 2.85 eV
   Prediction Drift: L_adv = |f_theta(x') - f_theta(x)| = |2.85 - 3.18| = 0.33 eV
   Total Score L_total(x') = 1.0 + 0.33 = 1.33
   Metropolis Acceptance: Delta = 1.33 - 1.00 = +0.33 > 0 -> ACCEPT candidate

8. DEFENDER CLOSED-LOOP UPDATE
   Clean Loss: L_clean = (3.18 - 3.20)^2 = 0.0004
   Adv Loss:   L_adv   = (2.85 - 3.20)^2 = 0.1225
   Fused Loss (lambda=0.5): L = 0.5 * (0.0004) + 0.5 * (0.1225) = 0.06145
   Adam Backprop: Update weights theta -> Hardens model against 2-fluoro substitution drift.
```
