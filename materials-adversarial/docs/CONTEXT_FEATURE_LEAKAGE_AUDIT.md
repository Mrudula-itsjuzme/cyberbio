# Context Feature Leakage Audit

To strictly ensure that Contextual Thompson Sampling does not receive an unfair advantage through future information leakage, we define the following rules and audit controls for context features $\mathbf{z}(x)$.

## 1. Allowed Features (Zero-Shot)
The following features are strictly permissible because they can be calculated deterministically from the original, unperturbed source SMILES sequence $x$ before any search operators are invoked:
- Sequence length (token count).
- Molecular weight (via RDKit).
- Normalized token histograms (frequencies of specific structural components like `C`, `O`, `c`).
- Simple topological descriptors (Ring count, branch count, aromatic fraction, heteroatom fraction, attachment point count).

## 2. Forbidden Features (Leaked)
The following features are strictly prohibited because they depend on the search trajectory, candidate generation, or external oracles:
- **Candidate Drift**: Cannot be used, as it requires spending a target-model query.
- **Candidate Validity Rate**: Cannot be used, as it requires sampling the proposal operator.
- **Best Arm (Hindsight/Test)**: Cannot be passed into the context vector under any circumstances.
- **Target-Model Predictions**: Even the clean prediction of the original source $x$ will **not** be used as a context feature in Version 1 to completely avoid any debate regarding "free" queries, query accounting, or baseline leakage.

## 3. Training & Normalization Leakage Control
- **Data Splitting**: The dataset will be rigorously split into disjoint Train, Validation, and Test sets. A source polymer from the Test set will never be observed during the training of the contextual prior.
- **Feature Normalization**: Standard scaling (mean centering and variance scaling) will be applied. The mean and variance statistics will be calculated **exclusively** on the Train set. Test features will be normalized using these frozen Train statistics.

## 4. Cold Start Initialization
- When Contextual Thompson encounters a new test source $x_{test}$, its prior parameters $\boldsymbol{\theta}_a$ for each arm must be initialized entirely from the generalized mapping learned on the training set.
- The controller will not peek at $x_{test}$'s historical attack results or pre-run the search to seed the prior.

## 5. Audit Validation Check
A unit test will be included in the test suite to assert that the `ContextFeatureExtractor` takes only a single string $x$ and returns the same vector $\mathbf{z}(x)$ irrespective of the search iteration, query budget, or current controller trace.
