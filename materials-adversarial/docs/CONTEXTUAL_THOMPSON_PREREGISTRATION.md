# Pre-Registration: Contextual Thompson Sampling

## 1. Scientific Hypotheses
- **H1**: Different source polymers exhibit different reward distributions over perturbation arms (substitution, insertion, deletion, rearrangement).
- **H2**: A contextual controller can use source-level features available BEFORE search to improve low-budget cumulative reward relative to non-contextual Thompson Sampling.
- **H3**: Contextual Thompson should provide the largest benefit at small target-model query budgets (e.g., $T \le 10$), where blind exploration is expensive.
- **H4**: At sufficiently large budgets ($T \ge 25$), ordinary Thompson may approach contextual performance because it has enough observations to learn the source-specific arm distribution online.
- **H5**: Contextual features may fail if the apparent source dependence observed previously is mostly sampling noise.

## 2. Experimental Design
- **Splits**: We will construct a dataset of at least 50 source polymers, strictly split into CONTEXT TRAIN SOURCES, CONTEXT VALIDATION SOURCES, and CONTEXT TEST SOURCES. 
- **Splitting Strategy**: Both a Random Split and a Scaffold-Aware Split will be conducted to evaluate out-of-distribution (OOD) generalization.
- **Budgets**: 5, 10, 25, 50 queries.
- **Seeds**: Minimum of 3 seeds (e.g., 42, 123, 456).
- **Baselines to Compare**:
  1. Uniform arm selection
  2. Non-contextual Thompson (Beta-Bernoulli)
  3. Contextual Thompson (Bayesian Linear)
  4. Best global fixed arm (the prior before context)
  5. Oracle hindsight best arm (Upper Bound)
- **Primary Endpoint**: Cumulative reward at small budget ($T=5, 10$) on HELD-OUT test sources.

## 3. Reward Function
- Reward is strictly binary ($r \in \{0, 1\}$).
- $r=1$ if and only if the candidate is valid and its prediction drift exceeds the verified baseline MAE threshold (`0.4619 eV`). Otherwise, $r=0$.
- Duplicate/cached evaluations will receive a reward of 0 and skip posterior updates.

## 4. Context Extraction
- Simple interpretable structural features: sequence length, molecular weight, ring count, branch count, aromatic fraction, heteroatom fraction, attachment count, token frequencies.
- All features must be normalized using TRAIN statistics only.

## 5. Algorithmic Approach
- **Bayesian Linear Thompson Sampling**: Models the reward for arm $a$ as $r \approx \mathbf{z}(x)^T \theta_a + \epsilon$.
- Zero-shot priors for held-out sources will be drawn from the Bayesian Linear regression fit on the training data.
- Online adaptation will merge the context prior with within-source observed rewards during the search.

## 6. Ablation & Control
- A mandatory **Shuffled-Context Control** experiment will pair test sequences with randomly shuffled context vectors. This isolates the true value of the structural features from baseline algorithmic behaviors.

## 7. Status
FINALIZED AND LOCKED prior to observing held-out test outcomes.
