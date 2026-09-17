# Target Model Architecture: TransPolymer Sequence Transformer

## 1. End-to-End Pipeline Overview

The primary target model is a **Two-Branch Sequence Transformer Regressor** (`TwoBranchTransformerRegressorModel`), adapting the **TransPolymer** paradigm for polymer property prediction:

```
PSMILES Sequence String (x)
   │
   ▼
[Chemical PSMILES Tokenizer] (Atom, bond, bracket & wildcard '*' tokens)
   │
   ▼
[Token Indices + Attention Mask] (Tensor shape: [Batch, SeqLen])
   │
   ▼
[Token Embedding Layer] + [1D Positional Encodings]
   │
   ▼
[Transformer Encoder Layers] (Multi-Head Self-Attention + FFN + LayerNorm + Residuals)
   │
   ▼
[Sequence Masked Pooling] (Mean pooling over non-padded positions -> Shared vector h in R^d)
   │
   ▼
┌───────────────────────────────────────┴───────────────────────────────────────┐
│                                                                               │
▼                                                                               ▼
[Branch A: Representation Invariance]                           [Branch B: Chemistry Sensitivity]
z_repr = ReLU(Linear(h)) + LayerNorm                            z_chem = ReLU(Linear(h)) + LayerNorm
│                                                                               │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
                   [Fused Linear Regression Head] (Concat(z_repr, z_chem) -> 1)
                                        │
                                        ▼
                  [TargetScaler Inverse Transform] -> Band Gap E_g (eV)
```

---

## 2. Chemical Tokenization & Embedding Pipeline

### A. PSMILES Tokenizer (`PSmilesTokenizer`)
The tokenizer parses chemical strings into discrete sub-word tokens using regex pattern matching (`TOKEN_PATTERN`), preserving chemical entity integrity (e.g. `[nH]`, `[C@@H]`, `Cl`, `Br`, `*`):
$$s = (s_1, s_2, \dots, s_L)$$
where $L$ is sequence length. Special tokens include `<pad>` (ID 0), `<unk>` (ID 1), `<bos>` (ID 2), `<eos>` (ID 3).

### B. Token Embeddings and Positional Encoding
Each token ID $s_i$ is mapped to a continuous embedding vector $\mathbf{E}_{\text{tok}}[s_i] \in \mathbb{R}^{d_{\text{model}}}$. A 1D learned positional embedding $\mathbf{E}_{\text{pos}}[i] \in \mathbb{R}^{d_{\text{model}}}$ is added to supply order information:

$$\mathbf{X}_i = \mathbf{E}_{\text{tok}}[s_i] + \mathbf{E}_{\text{pos}}[i], \quad \mathbf{X} \in \mathbb{R}^{L \times d_{\text{model}}}$$

Default dimensions: $d_{\text{model}} = 64$, $\text{max\_seq\_len} = 256$.

---

## 3. Transformer Encoder Layers

The model consists of $N_{\text{layers}} = 2$ stacked `nn.TransformerEncoderLayer` modules.

### A. Scaled Dot-Product Attention
For query $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$, key $\mathbf{K} = \mathbf{X}\mathbf{W}^K$, and value $\mathbf{V} = \mathbf{X}\mathbf{W}^V$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{pad}}\right) \mathbf{V}$$

where $\mathbf{M}_{\text{pad}} \in \{0, -\infty\}^{B \times L \times L}$ is the key padding mask ensuring that zero-padded sequence elements receive zero attention weight.

### B. Multi-Head Attention (MHA)
Attention is computed across $N_h = 4$ independent heads:

$$\text{MHA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_{N_h}) \mathbf{W}^O$$
$$\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_i^Q, \mathbf{X}\mathbf{W}_i^K, \mathbf{X}\mathbf{W}_i^V)$$

### C. Position-Wise Feed-Forward Network & Layer Normalization
Residual connections and Layer Normalization (`LayerNorm`) are applied around attention and FFN blocks:

$$\mathbf{H}^{(1)} = \text{LayerNorm}\left(\mathbf{X} + \text{MHA}(\mathbf{X})\right)$$
$$\text{FFN}(\mathbf{H}^{(1)}) = \text{ReLU}\left(\mathbf{H}^{(1)} \mathbf{W}_1 + \mathbf{b}_1\right) \mathbf{W}_2 + \mathbf{b}_2$$
$$\mathbf{H}^{(2)} = \text{LayerNorm}\left(\mathbf{H}^{(1)} + \text{FFN}(\mathbf{H}^{(1)})\right)$$

Inner feed-forward dimension: $d_{\text{ff}} = 128$, dropout rate $\text{p} = 0.1$.

---

## 4. Sequence Masked Pooling

The output sequence representation $\mathbf{H}^{(N)} \in \mathbb{R}^{L \times d_{\text{model}}}$ is pooled into a single fixed-size representation vector $\mathbf{h} \in \mathbb{R}^{d_{\text{model}}}$ using masked mean pooling:

$$\mathbf{h} = \frac{\sum_{i=1}^L \mathbf{H}_i^{(N)} \cdot (1 - M_i)}{\sum_{i=1}^L (1 - M_i) + \epsilon}$$

where $M_i = 1$ if position $i$ is padding and $M_i = 0$ otherwise.

---

## 5. Two-Branch Specialized Head Architecture

To capture both representation invariance (canonical SMILES equivalence) and chemical sensitivity (functional group modifications), the pooled embedding $\mathbf{h}$ is split into two specialized branches:

1. **Branch A (Representation-Invariance Branch $f_{\text{repr}}$)**:
   $$\mathbf{z}_{\text{repr}} = \text{LayerNorm}\left(\text{ReLU}\left(\mathbf{h} \mathbf{W}_A + \mathbf{b}_A\right)\right) \in \mathbb{R}^{d_{\text{branch}}}$$

2. **Branch B (Chemistry-Sensitivity Branch $f_{\text{chem}}$)**:
   $$\mathbf{z}_{\text{chem}} = \text{LayerNorm}\left(\text{ReLU}\left(\mathbf{h} \mathbf{W}_B + \mathbf{b}_B\right)\right) \in \mathbb{R}^{d_{\text{branch}}}$$

3. **Fusion Regression Head**:
   The representations from both branches are concatenated and projected to the scaled prediction $\hat{y}_{\text{scaled}}$:
   $$\mathbf{z}_{\text{fused}} = [\mathbf{z}_{\text{repr}} \,||\, \mathbf{z}_{\text{chem}}] \in \mathbb{R}^{2 \cdot d_{\text{branch}}}$$
   $$\hat{y}_{\text{scaled}} = \mathbf{z}_{\text{fused}} \mathbf{W}_{\text{reg}} + b_{\text{reg}}$$

Branch dimension: $d_{\text{branch}} = 32$.

---

## 6. Target Inverse Scaling

The model outputs a normalized value $\hat{y}_{\text{scaled}} \sim \mathcal{N}(0, 1)$. The unscaled physical band gap prediction $\hat{y} = f_\theta(x) \in \text{eV}$ is obtained via `TargetScaler`:

$$\hat{y} = (\hat{y}_{\text{scaled}} \cdot \sigma_{\text{target}}) + \mu_{\text{target}}$$

where $\mu_{\text{target}}$ and $\sigma_{\text{target}}$ are the training dataset mean and standard deviation.

---

## 7. Comparative Model: GraphMPNNPredictor

In addition to the Transformer sequence regressor, the codebase supports a graph message-passing neural network (`GraphPredictor` / `GraphMPNNPredictor`):
- **Input**: Graph adjacency matrix $\mathbf{A} \in \{0,1\}^{N \times N}$ and node feature matrix $\mathbf{X}_{\text{node}} \in \mathbb{R}^{N \times 7}$ (encoding atom type, formal charge, hybridization, aromaticity).
- **Message Passing**: 3 Graph Convolutional layers (`node_dim=7`, `hidden_dim=64`).
- **Separation**: `GraphMPNNPredictor` is evaluated as an independent graph model pathway for cross-model attack transferability tests.
