# Probabilistic MCMC Attack Generator

## 1. Algorithm Overview and Theoretical Grounding

The **Probabilistic MCMC Attack Generator** (`ProbabilisticMCMCAttack`) formulates adversarial sequence generation as a **model-guided stochastic search** over discrete chemical mutation space. 

Instead of unguided random mutations or exhaustive enumeration, the algorithm samples discrete proposal edits from a candidate distribution $q(x' | x)$ and accepts or rejects candidates using a **Metropolis-Hastings acceptance probability**:

$$\alpha(x \to x') = \min\left(1, \exp\left( \frac{\mathcal{L}_{\text{total}}(x') - \mathcal{L}_{\text{total}}(x)}{T} \right)\right)$$

where $T > 0$ is the search temperature and $\mathcal{L}_{\text{total}}$ is the multi-objective candidate score.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               PROBABILISTIC MCMC SEARCH FLOW                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Current State: Sequence x_t with prediction drift L(x_t)                             │
│ 2. Propose Mutation: Sample operator o in {Sub, Ins, Del, Bioisostere} -> Candidate x'  │
│ 3. Chemical Validity Gate: RDKit Parse + Valence + Star Balance                         │
│    - If invalid: Reject candidate immediately (L_chem = -100)                           │
│ 4. Score Candidate: Compute model drift L_adv = |f_theta(x') - f_theta(x)|              │
│ 5. Metropolis Accept/Reject:                                                            │
│    - Delta = L_total(x') - L_total(x_t)                                                 │
│    - Accept with probability min(1, exp(Delta / T))                                     │
│ 6. Update State: If accepted, set x_{t+1} = x'; record best candidate                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Proposal Operators & Bioisosteric Substitution Matrices

The proposal distribution $q(x' | x)$ selects uniformly from four modification operators:

1. **Role-Preserving Atom Substitution**: Replaces token $s_i$ with an atom token $t' \in \mathcal{V}_{\text{atom}}$ ($s_i \neq t'$).
2. **Token Insertion**: Inserts an atom token $t' \in \mathcal{V}_{\text{atom}}$ into non-branching locations.
3. **Token Deletion**: Deletes non-ring, non-attachment tokens.
4. **Bioisosteric Functional Group Replacement**: Substitutes chemically similar functional groups according to pre-defined bioisosteric matrices:

$$\mathcal{M}_{\text{bio}} = \begin{cases} 
\text{-F} \longleftrightarrow \text{-Cl} \longleftrightarrow \text{-Br} & \text{(Halogen Isosteres)} \\
\text{-OH} \longleftrightarrow \text{-SH} \longleftrightarrow \text{-NH}_2 & \text{(Polar / Chalcogen Isosteres)} \\
\text{-CH}_3 \longleftrightarrow \text{-CF}_3 \longleftrightarrow \text{-CH}_2\text{CH}_3 & \text{(Alkyl / Haloalkyl Isosteres)}
\end{cases}$$

---

## 3. Multi-Objective Scoring Function

A proposed candidate $x'$ is scored via:

$$\mathcal{L}_{\text{total}}(x') = \mathcal{L}_{\text{chem}}(x') + \mathcal{L}_{\text{adv}}(x')$$

where:
- $\mathcal{L}_{\text{chem}}(x') = 1.0$ if $x'$ is RDKit parseable and $S_{\text{Tanimoto}}(x, x') \ge 0.5$; otherwise $\mathcal{L}_{\text{chem}}(x') = -100.0$.
- $\mathcal{L}_{\text{adv}}(x') = |f_\theta(x') - f_\theta(x)|$ (prediction drift in eV).

---

## 4. Algorithmic Pseudocode

```python
Algorithm 1: Probabilistic MCMC Attack Generator (ProbabilisticMCMCAttack)
---------------------------------------------------------------------------
Input: Source PSMILES x, Target Predictor f_theta, Vocabulary V, 
       Steps N_steps=20, Temperature T=5.0, Similarity Cutoff S_min=0.5
Output: Adversarial Candidate x_adv

1: x_current = x
2: y_orig = f_theta(x)
3: score_current = 1.0  # L_chem(x) = 1.0, L_adv(x) = 0.0
4: best_candidate = x
5: max_drift = 0.0

6: for step = 1 to N_steps do:
7:     Sample proposal operator o ~ Uniform({Substitution, Insertion, Deletion, Bioisostere})
8:     x_proposed = ApplyOperator(o, x_current, V)
9:     
10:    # Chemical Validity & Plausibility Gate
11:    if not IsValidRDKit(x_proposed) or Tanimoto(x, x_proposed) < S_min:
12:        score_proposed = -100.0
13:    else:
14:        y_pred = f_theta(x_proposed)
15:        drift = |y_pred - y_orig|
16:        score_proposed = 1.0 + drift
17:        if drift > max_drift:
18:            max_drift = drift
19:            best_candidate = x_proposed
20:    
21:    # Metropolis Acceptance Rule
22:    delta = score_proposed - score_current
23:    if delta > 0:
24:        accept_prob = 1.0
25:    else:
26:        accept_prob = exp(delta / T)
27:    
28:    u ~ Uniform(0, 1)
29:    if u < accept_prob:
30:        x_current = x_proposed
31:        score_current = score_proposed

32: return best_candidate
```

---

## 5. MCMC Parameters & Best Candidate Retention

- **Temperature ($T$)**: Default $T = 5.0$. Controls stochastic exploration rate. Higher $T$ allows temporary moves into lower-drift regions to escape local minima.
- **Query Accounting**: Every forward call $f_\theta(x_{\text{proposed}})$ is logged. The total queries per source molecule are strictly bounded by $Q \le N_{\text{steps}}$.
- **Best Candidate Retention**: The generator tracks and returns the candidate yielding maximum valid prediction drift across all MCMC steps.
