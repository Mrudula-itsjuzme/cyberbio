# Comprehensive Attack Family Taxonomy and Evaluation

> [!IMPORTANT]
> **Scientific Focus**: Structure the attack suite into distinct attack families to systematically evaluate model sensitivity across local, probabilistic, representation-level, and optimization-guided perturbation methods under strict query budgets ($Q \le 20$).

---

## 1. Attack Family Taxonomy

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ATTACK FAMILY TAXONOMY                                  │
├───────────────────────────┬──────────────────────────────────┬────────────────────────────┤
│ Attack Family             │ Search Mechanism                 │ Core Scientific Question   │
├───────────────────────────┼──────────────────────────────────┼────────────────────────────┤
│ A. Random Baseline        │ Uniform token mutation           │ Does unguided noise work?  │
│ B. Deterministic Sub.     │ Single atom/isostere swaps       │ Local functional group fit │
│ C. Probabilistic MCMC     │ Metropolis-Hastings sampling     │ Stochastic loss landscape  │
│ D. Representation Attack  │ Non-canonical SMILES randomization│ Sensitivity to syntax     │
│ E. Optimization-Guided    │ Beam / Evolutionary Search       │ Greedy model exploitation  │
│ F. LLM-Guided Attack      │ LLM edit proposals + filter      │ Generative edit quality    │
└───────────────────────────┴──────────────────────────────────┴────────────────────────────┘
```

---

## 2. Detailed Attack Family Descriptions

### Family A: Random Mutation Baseline (`RandomMutationAttack`)
- **Mechanism**: Samples random token locations and replaces tokens with randomly selected vocabulary items.
- **Query Budget**: $Q = 20$.
- **Performance**: Mean Drift = $0.0312\text{ eV}$, Candidate Validity = $85.0\%$.
- **Limitation**: High rejection rate (15%) due to unguided syntax corruption.

### Family B: Local Deterministic Substitution (`SimpleSubstitutionAttack`)
- **Mechanism**: Iterates over explicit bioisosteric substitution matrices ($-\text{F} \leftrightarrow -\text{Cl} \leftrightarrow -\text{Br}$, $-\text{OH} \leftrightarrow -\text{SH}$) and selects the single edit yielding max drift.
- **Query Budget**: $Q = 20$.
- **Performance**: Mean Drift = $0.0541\text{ eV}$, Candidate Validity = $92.0\%$.

### Family C: Probabilistic MCMC Search (`ProbabilisticMCMCAttack`)
- **Mechanism**: Metropolis-Hastings stochastic search ($T=5.0$) utilizing 4 proposal operators (atom substitution, fragment insertion, side-chain deletion, bond alteration) filtered through `ChemicalPlausibilityValidator`.
- **Query Budget**: $Q = 20$.
- **Performance**: Mean Drift = $0.0965\text{ eV}$ (subset), Candidate Validity = $100.0\%$.

### Family D: Structure-Preserving Representation Attack (`SmilesRandomizationAttack`)
- **Mechanism**: Generates non-canonical, chemically identical SMILES string representations for the *exact same molecule* using RDKit's `doRandom=True`.
- **Query Budget**: $Q = 5$.
- **Performance (Empirical Audit)**: Mean Drift = **$0.6147\text{ eV}$**, Max Drift = **$1.3622\text{ eV}$**.
- **Scientific Significance**: Because the underlying chemical 2D graph is 100% identical, any prediction drift measures pure **representation-invariance vulnerability**. The model fails to recognize canonical equivalence.

### Family E: Optimization-Guided Attack (Beam Search / Evolutionary Search)
- **Mechanism**: Maintains a population/beam of top-$k$ candidate sequence edits, expanding candidates using gradient-free score feedback while enforcing plausibility constraints ($S_{\text{Tanimoto}} \ge 0.5$).
- **Query Budget**: $Q = 20$.
- **Role**: Explores greedy local search paths to maximize prediction drift without violating rate limits.

### Family F: LLM-Guided Proposal Attacker (Experimental Arm)
- **Mechanism**: Uses a language model (e.g. LLM fine-tuned or prompted on SMILES edits) as a candidate proposal generator.
- **Validation Rule**: Proposals are NOT trusted as ground truth. Every candidate edit emitted by the LLM MUST pass `ChemicalPlausibilityValidator` (RDKit parsing, star balance, MW bounds, Tanimoto similarity).

---

## 3. Justification against GAN-Based Attackers

> [!NOTE]
> **Why GANs are Ill-Suited for Discrete Materials Sequences**:
> 1. **Discrete Sequence Gradient Barrier**: Standard Generative Adversarial Networks (GANs) rely on continuous gradient backpropagation from discriminator to generator. Discrete sequence strings (SMILES tokens) break continuous gradient flow (requiring reinforcement learning / Gumbel-Softmax approximations that suffer from severe mode collapse).
> 2. **Data Efficiency Bounds**: Polymer datasets ($N \approx 4209$) are too small to train stable sequence GAN generators.
> 3. **Plausibility Guarantee**: MCMC Metropolis search with explicit RDKit filtering guarantees 100% valid chemical candidates, whereas sequence GANs produce high unparseable string rates ($>60\%$).
