# VIVA STUDY GUIDE
## Adversarial Robustness in Polymer Property Prediction

### 1. What problem are we solving?
We are evaluating the robustness of machine learning models used to predict polymer properties from sequence strings. Specifically, we test whether a model's predictions remain stable when the input sequence is changed in a way that preserves the chemistry (representation robustness) or modified slightly via chemically plausible local edits (structural robustness).

### 2. Why polymer property prediction?
Accelerated materials discovery relies on ML proxies to screen vast spaces of candidate polymers for properties like bandgap, glass transition temperature, or conductivity. If these models are brittle, they cannot be reliably used for high-throughput screening or generative design.

### 3. What is PSMILES?
Polymer SMILES (PSMILES) is an extension of the SMILES (Simplified Molecular-Input Line-Entry System) notation that introduces attachment points (e.g., `[*]` or `*`) to denote the connection points of a repeating polymer unit.

### 4. Why can multiple SMILES strings describe the same structure?
SMILES strings are derived from a graph traversal of the molecule. Because there are multiple valid starting nodes and paths (especially for rings and branches), a single molecule can be serialized into many distinct but chemically equivalent strings.

### 5. Why is representation sensitivity bad?
If a model gives two different predictions for two chemically identical strings, the model has learned the syntax of the string rather than the underlying physics of the molecule.

### 6. What is TransPolymer / the Transformer doing?
It uses a self-attention mechanism to learn contextual relationships between tokens (atoms, bonds, brackets) in the PSMILES string, ultimately compressing the sequence into a latent vector that a linear head maps to a property value.

### 7. Input pipeline
PSMILES string $\rightarrow$ Tokenization $\rightarrow$ Embedding $\rightarrow$ Transformer Encoder $\rightarrow$ Regression Head $\rightarrow$ Bandgap Prediction.

### 8. Tokenization
We use a chemical-aware tokenizer that splits the string by meaningful characters (e.g., separating `C`, `=`, `O`, `[nH]`, `Cl`).

### 9. Training objective
Mean Squared Error (MSE) loss against the true standardized bandgap values.

### 10. Clean evaluation metrics
MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and $R^2$ (Coefficient of Determination) measured on a held-out test set.

### 11. RandomSMILES attack
Generating multiple valid, non-canonical SMILES strings for the same molecule and passing them to the model to maximize the drift from the canonical prediction.

### 12. Why RandomSMILES is representation-preserving
The underlying molecular graph (atoms and connectivity) is perfectly identical; only the traversal path used to generate the string string is different.

### 13. MCMC attack
A stochastic search (Markov Chain Monte Carlo-style) that iteratively proposes small edits (insert, delete, substitute) to the PSMILES sequence, evaluating the prediction drift and applying chemical constraints.

### 14. What Markov Chain Monte Carlo means here
It is a probabilistic search guiding the sequence towards higher prediction drift. It accepts edits that increase drift and sometimes accepts edits that decrease it (with some probability) to avoid local minima, though it doesn't compute strict detailed balance (it acts more like simulated annealing).

### 15. Candidate proposal
Randomly selecting an editable token index and applying a substitution, deletion, or insertion from a valid vocabulary.

### 16. Acceptance/rejection concept
The proposed molecule must pass syntactic (RDKit) and heuristic plausibility checks. If valid, its adversarial score (drift) is evaluated against the current best.

### 17. Perturbation budget
The maximum number of tokens that can be changed (usually 1-3) to ensure the edited molecule remains locally similar to the original.

### 18. RDKit validation
Ensures the string can be parsed into a valid chemical graph (valences satisfied, rings closed). It provides syntactic, not scientific, validity.

### 19. Tanimoto similarity
A fingerprint-based measure of structural similarity (0 to 1). We use it to ensure the MCMC edits do not completely destroy the original polymer's identity.

### 20. Molecular-weight constraints
Ensures the edited polymer unit has a comparable mass to the original, preventing runaway insertions.

### 21. Difference between: syntactic validity, chemical plausibility, and physical-property invariance
- **Syntactic validity**: RDKit parses it without error.
- **Chemical plausibility**: It has 2 attachment points, no exotic elements, sensible MW.
- **Physical-property invariance**: The new structure actually has the same real-world bandgap (requires a physical oracle like DFT, which we lack).

### 22. RandomSMILES augmentation defense
Training the model by dynamically randomizing the PSMILES representation of the training data in each epoch.

### 23. MCMC consistency defense
Training the model to minimize the difference in predictions between the original molecule and an MCMC-generated adversarial variant (Adversarial Training).

### 24. Combined defense
Applying both RandomSMILES augmentation and MCMC adversarial training.

### 25. Why cross-attack transfer matters
It shows whether a defense against one type of attack (e.g., representation) confers robustness against another (e.g., structural), isolating the model's failure modes.

### 26. What the ablation study tells us
By turning defenses on and off (while holding split, architecture, and seeds constant), we identify that Combined Defense is necessary to address both vulnerabilities simultaneously.

### 27. Sequence-length shortcut hypothesis
The hypothesis that the model partially relies on the raw length of the string to predict properties, making it highly sensitive to insertions or deletions.

### 28. Pearson correlation
A measure of linear association (e.g., between sequence length and absolute error).

### 29. Spearman correlation
A measure of monotonic rank association.

### 30. Why correlation does not prove causation
Sequence length is naturally correlated with molecular weight/complexity, which *does* affect the bandgap. The correlation alone doesn't prove the model ignores chemistry, only that it is sensitive to length.

### 31. Random vs scaffold split
- Random: IID distribution of data.
- Scaffold: Grouping by molecular core structure, testing OOD generalization.

### 32. Why scaffold split is harder / more meaningful
It prevents the model from memorizing highly similar structures, forcing it to learn generalized chemical rules.

### 33. MAE
Mean Absolute Error. The average absolute magnitude of the errors.

### 34. RMSE
Root Mean Squared Error. Penalizes large errors more heavily than MAE.

### 35. R²
Proportion of variance in the dependent variable predictable from the independent variable (1.0 is perfect, 0 is baseline mean).

### 36. Prediction drift
The absolute difference in prediction (eV) between the original and the perturbed molecule.

### 37. Multi-seed validation
Running experiments across multiple random weight initializations (e.g., 42, 123, 456).

### 38. Why seed variance matters
Deep learning models can converge to different local minima. Multi-seed averages prove the defense isn't a statistical fluke.

### 39. Major final findings
- TransPolymer is brittle to both representations and local edits.
- RandomSMILES augmentation fixes representation brittleness and improvides evidence scaffold OOD performance.
- Defenses are mostly orthogonal.
- Sequence length acts as a measurable proxy associated with failure.

### 40. Limitations
We cannot prove that the MCMC edits physically maintain the same bandgap without a computational oracle (DFT). Thus, we measure *regularization consistency*, not guaranteed physical correctness.

### 41. Why DFT is not part of the current core result
It requires specialized quantum chemistry setups (e.g., VASP/Quantum Espresso) and massive compute budgets, which is outside the scope of auditing the neural network's internal consistency.

### 42. What DFT would add
It would allow us to penalize the model against the *true* new bandgap, proving whether the model is learning the true physical manifold.

### 43. Why we did NOT add LLM/GAN/RL attacks
The core goal was rigorous evaluation of fundamental robustness, not generating arbitrarily complex attacks that obfuscate the underlying sensitivity (representation vs chemistry).

### 44. Future work
Integrating a DFT oracle to validate the physical properties of the generated adversarial candidates, or extending to sequence-agnostic architectures like Graph Neural Networks.

### 45. Novelty / contribution of the work
Systematically decomposing polymer sequence model vulnerabilities into representation artifacts versus local structural artifacts, and demonstrating that data augmentation acts as a structural regularizer in scaffold splits.
