# ARCHITECTURE / PIPELINE EXPLAINED

## 1. Dataset $\rightarrow$ PSMILES
We start with a dataset of polymers where the 3D structure is represented as a 1D string (PSMILES). Attachment points (`[*]`) indicate where the polymer repeat unit connects to itself. The ground truth is the Bandgap ($E_g$) in eV.

## 2. Tokenization
The string is split into discrete chemical tokens. For example, `[*]Oc1...` becomes `['[*]', 'O', 'c', '1', ...]`. Brackets and specific halogen atoms are kept together.

## 3. Transformer
The tokens are mapped to learned dense vectors (Embeddings) and combined with positional encodings. The Transformer Encoder uses self-attention to weigh the importance of different tokens relative to each other, capturing the polymer's topology.

## 4. Bandgap Prediction
The latent representation from the Transformer is pooled and passed through a linear layer to output a single continuous scalar value (eV).

## 5. Attack Generation
We take a successfully predicted molecule and apply an attack.
- **RandomSMILES**: We generate a new string for the exact same graph.
- **MCMC**: We swap, insert, or delete a token (e.g., `O` $\rightarrow$ `N`).

## 6. Validity Checks
The candidate string is passed to RDKit to ensure the syntax is valid (rings closed, valences okay). It then passes heuristic checks (must have exactly 2 attachment points, reasonable molecular weight, minimum Tanimoto similarity to the original).

## 7. Prediction Drift
The valid candidate is passed back through the Model. We calculate the absolute difference between the original prediction and the new prediction. The MCMC algorithm uses this drift as a reward signal to guide the search.

## 8. Defense
We train the model using **Data Augmentation** (passing random representations during training) and **Adversarial Training** (forcing the model to output the same prediction for the MCMC candidate as the original).
