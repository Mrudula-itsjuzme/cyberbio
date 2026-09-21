---
title: "CyberBio Technical Report — Part I, Version 2: Foundations, Transformer, and GraphMPNN"
author: "CyberBio Project Documentation"
date: "September 2026"
---

> **Evidence policy.** Statements labelled **Project fact** are grounded in the current `Mrudula-itsjuzme/cyberbio` repository. Sections labelled **Mathematical background** explain standard theory needed to understand the implementation. Historical results are kept only when their status is explicit: canonical, historical, rejected, blocked, or developmental-unverified.

# Scope of this volume

This volume answers the question: **what exactly happens mathematically between a polymer representation and a bandgap prediction?** It contains both the historical sequence-model path and the final graph-native path.

# 1. Prediction problem and notation

## 1.1 Project fact: active task

The active materials task is supervised regression. For a dataset

$$
\mathcal D = \{(x_i,y_i)\}_{i=1}^N,
$$

$x_i$ is a polymer representation and $y_i\in\mathbb R$ is its electronic bandgap $E_g$ in eV. The active polyVERSE lineage contains 4,209 usable records.

## 1.2 Core notation

| Symbol | Meaning |
|---|---|
| $B$ | batch size |
| $L$ | sequence length |
| $V$ | vocabulary size |
| $d$ | Transformer model width, $d=64$ |
| $H$ | number of attention heads, $H=4$ |
| $d_k$ | width per attention head, $d/H=16$ |
| $N$ | number of graph nodes, padded to at most 256 |
| $F$ | graph node-feature width, $F=7$ |
| $y$ | physical bandgap target in eV |
| $\tilde y$ | normalized target |
| $f_\theta$ | learned predictor |

# 2. PSMILES as a structured sequence

## 2.1 Project fact: tokenizer grammar

The repository does not split chemistry blindly into characters. `tokenizer.py` explicitly protects bracket atoms, extended ring closures, two-letter atoms such as `Cl` and `Br`, ordinary atom tokens, ring digits, branch parentheses, and bond symbols. This matters because splitting `Cl` into `C` and `l` would manufacture a string-processing artifact and then mislabel it as model robustness.

The tokenizer is a deterministic partition

$$
\tau : x \mapsto (t_1,t_2,\ldots,t_L).
$$

A vocabulary maps tokens to integer IDs:

$$
\nu(t_j)=z_j,\qquad z_j\in\{0,1,\ldots,V-1\}.
$$

Padding ID 0 is reserved in the main tokenizer design.

## 2.2 Padding and masks

A batch needs a common length. Short sequences are padded:

$$
Z\in\mathbb N^{B\times L_{\max}}.
$$

A Boolean padding mask $M$ marks padded positions. In the PyTorch path used here, `True` means “ignore this position.”

The padding mask is not cosmetic. A later Phase 11B architecture-comparison bug omitted it during Transformer evaluation, letting padded zeros contaminate mean-pooled representations. Phase 11C repaired this and restored the historical Transformer metrics.

# 3. Target scaling

## 3.1 Project fact: `TargetScaler`

The target is standardized using statistics fit on the training targets:

$$
\mu = \frac1{N_{tr}}\sum_{i\in tr} y_i,
$$

$$
\sigma = \sqrt{\frac1{N_{tr}}\sum_{i\in tr}(y_i-\mu)^2},
$$

$$
\tilde y_i=\frac{y_i-\mu}{\sigma}.
$$

The inverse transform is

$$
y_i = \sigma \tilde y_i + \mu.
$$

The Framework V2 forensic audit later verified the canonical scaler as train-fitted and verified that inverse scaling is applied exactly once in the repaired GraphMPNN prediction path.

# 4. Transformer encoder from first principles

> **Mathematical background.** The equations in Sections 4–12 are standard Transformer mathematics. **Project mapping** boxes connect them to the repository implementation.

## 4.1 Why an embedding exists

A token ID is an index, not a useful continuous representation. An embedding matrix

$$
E\in\mathbb R^{V\times d}
$$

maps token ID $z_j$ to

$$
\mathbf e_j=E[z_j]\in\mathbb R^d.
$$

For a batch,

$$
X_{emb}\in\mathbb R^{B\times L\times d}.
$$

**Project mapping.** `TransformerRegressorModel` uses `nn.Embedding(vocab_size + 1, d_model, padding_idx=0)` with $d=64$ in the canonical small Transformer configurations.

## 4.2 Why position must be represented

Self-attention alone does not know whether a token appeared first or tenth. The implementation uses a learned positional embedding matrix

$$
P\in\mathbb R^{L_{max}\times d}
$$

and forms

$$
H^{(0)}_j=E[z_j]+P[j].
$$

**Project mapping.** `pos_encoder = nn.Embedding(max_seq_len, d_model)`.

# 5. Self-attention from first principles

## 5.1 Query, key, and value projections

For one attention head, every token representation is projected into three spaces:

$$
Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V.
$$

If $H\in\mathbb R^{L\times d}$ and the head width is $d_k$, then

$$
W_Q,W_K,W_V\in\mathbb R^{d\times d_k},
$$

so

$$
Q,K,V\in\mathbb R^{L\times d_k}.
$$

Interpretation:

- a **query** says what information the current token is seeking;
- a **key** describes how another token can be matched;
- a **value** contains the information that can be aggregated if the match is strong.

## 5.2 Pairwise compatibility

The raw score between token $i$ and token $j$ is

$$
s_{ij}=\mathbf q_i^\top\mathbf k_j.
$$

Matrix form:

$$
S=QK^\top\in\mathbb R^{L\times L}.
$$

Every row answers: “for this query token, how compatible is every key token?”

## 5.3 Why divide by $\sqrt{d_k}$

If entries of $q$ and $k$ have roughly unit variance, the variance of a dot product grows with $d_k$. Very large logits push softmax toward near-one-hot outputs and weak gradients. Scaled attention therefore uses

$$
\widetilde S=\frac{QK^\top}{\sqrt{d_k}}.
$$

For this project’s usual $d=64$ and $H=4$,

$$
d_k=\frac{64}{4}=16,
\qquad
\sqrt{d_k}=4.
$$

## 5.4 Padding mask inside attention

Padded positions must receive zero attention probability. Conceptually define

$$
M_{ij}=\begin{cases}
0,&\text{if key position }j\text{ is real},\\
-\infty,&\text{if key position }j\text{ is padding}.
\end{cases}
$$

Then

$$
S' = \frac{QK^\top}{\sqrt{d_k}} + M.
$$

Because $e^{-\infty}=0$, softmax assigns padded keys zero probability.

## 5.5 Softmax

For query position $i$,

$$
\alpha_{ij}
=
\frac{\exp(S'_{ij})}
{\sum_{m=1}^{L}\exp(S'_{im})}.
$$

Each row sums to one:

$$
\sum_j\alpha_{ij}=1.
$$

## 5.6 Weighted value aggregation

The contextual output for token $i$ is

$$
\mathbf o_i=\sum_j\alpha_{ij}\mathbf v_j.
$$

Matrix form:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V.
$$

# 6. Numerical attention toy example

Suppose one query sees scaled, unmasked scores

$$
[2,1,0].
$$

Softmax gives approximately

$$
[0.665,0.245,0.090].
$$

If the three scalar values are $[4,10,-2]$, the aggregated output is

$$
0.665(4)+0.245(10)+0.090(-2)
\approx 4.93.
$$

The output is not “the most similar token.” It is a weighted mixture of all allowed values.

# 7. Multi-head attention

A single head can emphasize one compatibility pattern. Multi-head attention computes several in parallel:

$$
\text{head}_h
=
\operatorname{Attention}(Q_h,K_h,V_h),
\qquad h=1,\ldots,H.
$$

The outputs are concatenated:

$$
O_{cat}=\operatorname{Concat}(\text{head}_1,\ldots,\text{head}_H),
$$

then projected:

$$
O=O_{cat}W_O.
$$

For $H=4$ and $d=64$, each head operates on 16-dimensional projections while the concatenated result returns to 64 dimensions.

# 8. Residual connections, layer normalization, and feed-forward network

## 8.1 Residual path

A residual connection adds the block input back:

$$
R=H+\operatorname{Dropout}(O).
$$

This gives gradients a direct path through deep stacks and lets the attention branch learn a correction rather than reconstructing the entire representation.

## 8.2 Layer normalization

For a token vector $r\in\mathbb R^d$,

$$
\mu_r=\frac1d\sum_{k=1}^{d}r_k,
$$

$$
\sigma_r^2=\frac1d\sum_{k=1}^{d}(r_k-\mu_r)^2,
$$

$$
\operatorname{LN}(r)
=
\gamma\odot\frac{r-\mu_r}{\sqrt{\sigma_r^2+\epsilon}}+\beta.
$$

## 8.3 Position-wise feed-forward network

Each token independently passes through an MLP:

$$
\operatorname{FFN}(h)
=W_2\phi(W_1h+b_1)+b_2.
$$

The project’s encoder layer uses $d_{ff}=128$ in its documented small configuration. PyTorch’s `TransformerEncoderLayer` handles the attention, residual, normalization, and feed-forward internals.

# 9. Two encoder layers

The output of encoder layer 1 becomes the input of layer 2:

$$
H^{(1)}=\operatorname{EncLayer}_1(H^{(0)}),
$$

$$
H^{(2)}=\operatorname{EncLayer}_2(H^{(1)}).
$$

The tensor remains

$$
B\times L\times 64.
$$

# 10. Masked sequence pooling

The sequence model must become a single vector for regression. With a valid-token mask $m_j\in\{0,1\}$,

$$
\mathbf h
=
\frac{\sum_{j=1}^{L}m_jH^{(2)}_j}
{\sum_{j=1}^{L}m_j}.
$$

Thus

$$
\mathbf h\in\mathbb R^{64}.
$$

**Project mapping.** `encode()` multiplies token states by `(~padding_mask)` and divides by the count of valid positions.

# 11. Historical two-branch Transformer

The specialized model forms two learned projections from the pooled state:

$$
\mathbf z_{repr}
=
\operatorname{LN}
\left(
\operatorname{ReLU}(W_A\mathbf h+b_A)
\right),
$$

$$
\mathbf z_{chem}
=
\operatorname{LN}
\left(
\operatorname{ReLU}(W_B\mathbf h+b_B)
\right),
$$

with

$$
\mathbf z_{repr},\mathbf z_{chem}\in\mathbb R^{32}.
$$

They are concatenated:

$$
\mathbf z=[\mathbf z_{repr};\mathbf z_{chem}]\in\mathbb R^{64},
$$

and the normalized scalar prediction is

$$
\widehat{\tilde y}=\mathbf w^\top\mathbf z+b.
$$

Finally,

$$
\hat y=\sigma\widehat{\tilde y}+\mu.
$$

The important historical result is not that these branches “learned chemistry semantics.” That hypothesis was not supported. The architecture remains valuable as an ablation/control in the project history.

# 12. Transformer training mathematics

## 12.1 Mean-squared error

For normalized targets,

$$
\mathcal L_{MSE}
=
\frac1B\sum_{i=1}^{B}
(\widehat{\tilde y}_i-\tilde y_i)^2.
$$

For one prediction,

$$
\frac{\partial \mathcal L}{\partial \widehat{\tilde y}}
=
2(\widehat{\tilde y}-\tilde y)
$$

up to batch averaging. Backpropagation applies the chain rule through the regression head, branch layers, pooling, Transformer layers, positional embeddings, and token embeddings.

## 12.2 Adam

For gradient $g_t=\nabla_\theta\mathcal L_t$,

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

with bias-corrected moments

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

and update

$$
\theta_{t+1}
=
\theta_t-
\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Project scripts commonly use Adam with learning rate $10^{-3}$ in the small experimental models.

# 13. Why sequence serialization creates a structural vulnerability

A Transformer receives an ordered token sequence. Two equivalent SMILES strings can induce different token orders, different positional embeddings, different query-key interactions, and therefore different pooled states. Nothing in the architecture forces

$$
f(x_1)=f(x_2)
$$

for two strings encoding the same graph.

Augmentation can teach approximate invariance, but it is learned rather than structural.

# 14. Graph representation

## 14.1 Project fact: graph node features

`graph_dataset.py` maps every atom to seven features:

$$
\mathbf x_i=
[
Z_i,
\deg(i),
q_i,
a_i,
hyb_i,
H_i,
\star_i
]\in\mathbb R^7,
$$

where $Z$ is atomic number, $q$ formal charge, $a$ aromaticity indicator, and $\star$ flags wildcard attachment atoms.

The dataset also constructs:

- adjacency $A\in\{0,1\}^{N\times N}$;
- 3-dimensional edge attributes containing bond order, aromatic flag, and conjugation flag;
- a node mask for padded graph positions.

The current `GraphPredictor` itself uses the adjacency and node features; the edge-attribute tensor is constructed but not consumed by that model class.

## 14.2 Self-loops

The dataset sets

$$
A_{ii}=1
$$

for every real node. This gives each node an explicit self-neighbor in the message aggregation.

# 15. GraphMPNN from first principles

> **Mathematical background + exact project mapping.** The following equations match the simple GraphMPNN implementation in `graph_predictor.py`.

## 15.1 Initial node embedding

Raw 7-dimensional features are mapped to hidden width 64:

$$
H^{(0)}=XW_{emb}+b_{emb},
$$

where

$$
X\in\mathbb R^{B\times N\times 7},
\qquad
H^{(0)}\in\mathbb R^{B\times N\times 64}.
$$

## 15.2 Degree normalization

The implementation computes node degree

$$
d_i=\sum_j A_{ij}
$$

and row-normalized adjacency

$$
\widetilde A_{ij}=\frac{A_{ij}}{d_i+10^{-8}}.
$$

This is a random-walk-style row normalization, not the symmetric $D^{-1/2}AD^{-1/2}$ normalization used by some GCN formulations.

## 15.3 Neighbor message

For layer $\ell$,

$$
M^{(\ell)}=\widetilde A H^{(\ell)}.
$$

For node $i$,

$$
\mathbf m_i^{(\ell)}
=
\sum_j\widetilde A_{ij}\mathbf h_j^{(\ell)}.
$$

Because self-loops are present, this mixture includes the node’s own previous state.

## 15.4 Node update

The code has separate transforms for the node state and neighbor message:

$$
\mathbf u_i^{(\ell+1)}
=W_{node}^{(\ell)}\mathbf h_i^{(\ell)}+b_{node}^{(\ell)}
+
W_{msg}^{(\ell)}\mathbf m_i^{(\ell)}+b_{msg}^{(\ell)}.
$$

Then

$$
\mathbf h_i^{(\ell+1)}
=\operatorname{ReLU}(\mathbf u_i^{(\ell+1)}).
$$

Finally padded positions are zeroed:

$$
\mathbf h_i^{(\ell+1)}
\leftarrow
m_i\mathbf h_i^{(\ell+1)}.
$$

The canonical model uses three such layers.

# 16. Receptive field intuition

After one message-passing layer, a node representation depends on its one-hop neighborhood. After two layers it can incorporate two-hop information; after three, three-hop information can influence the node state. This is why the model can learn local chemical topology without relying on a particular text traversal.

# 17. Global graph pooling

After the final graph layer, valid node vectors are averaged:

$$
\mathbf h_G
=
\frac{1}{N_G}
\sum_{i=1}^{N_G}\mathbf h_i^{(3)}.
$$

The pooled graph vector lies in

$$
\mathbb R^{64}.
$$

Mean pooling makes the output independent of atom indexing order, provided the graph construction describes the same graph.

# 18. Graph regression head

The project uses

$$
\mathbf r=\operatorname{ReLU}(W_1\mathbf h_G+b_1),
\qquad \mathbf r\in\mathbb R^{32},
$$

then

$$
\widehat{\tilde y}=W_2\mathbf r+b_2.
$$

The canonical checkpoint contains 27,585 parameters according to the release/audit artifacts.

# 19. Why equivalent SMILES collapse to the same graph

If two valid SMILES strings decode to the same molecular graph, RDKit graph construction yields the same underlying atom/bond topology up to node permutation. A permutation-equivariant message-passing computation followed by permutation-invariant mean pooling produces the same graph-level output up to floating-point effects.

This is the structural reason the GraphMPNN solved the project’s original serialization vulnerability rather than merely learning to tolerate a subset of alternate strings.

# 20. Strict graph evaluation and a subtle historical defect

`GraphDataset` historically had a dangerous fallback: if a SMILES string failed to parse, non-strict mode walked forward to another parseable row. This is tolerable only if it never fires on clean training data. It is unacceptable in adversarial evaluation because an invalid candidate could be scored as a different molecule.

The repaired Framework V2 predictor constructs `GraphDataset(..., strict=True)`, so invalid SMILES raise instead of silently substituting a neighbor. This detail belongs in the model chapter because data-loader behavior changes the meaning of an attack result.

# 21. Transformer vs GraphMPNN: what changed mathematically

| Aspect | Transformer | GraphMPNN |
|---|---|---|
| Primitive | ordered tokens | atoms + adjacency |
| Position | explicit learned positional embedding | graph connectivity |
| Interaction | dense self-attention between all token positions | local normalized neighbor aggregation |
| Serialization invariance | must be learned | structural for graph-equivalent SMILES |
| Pooling | masked mean over sequence positions | mean over graph nodes |
| Canonical project role | historical baseline / ablation | final predictor |

# 22. Exact tensor-shape traces

## 22.1 Transformer

$$
(B,L)
\to
(B,L,64)
\to
(B,L,64)
\to
(B,L,64)
\to
(B,64)
\to
(B,32)+(B,32)
\to
(B,64)
\to
(B,1).
$$

## 22.2 GraphMPNN

$$
(B,N,7)
\to
(B,N,64)
\to
(B,N,64)
\to
(B,N,64)
\to
(B,N,64)
\to
(B,64)
\to
(B,32)
\to
(B,1).
$$

# 23. Evaluation metrics

For predictions $\hat y_i$ and targets $y_i$:

$$
\mathrm{MAE}
=
\frac1N\sum_i|y_i-\hat y_i|,
$$

$$
\mathrm{RMSE}
=
\sqrt{\frac1N\sum_i(y_i-\hat y_i)^2},
$$

$$
R^2
=
1-
\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y)^2}.
$$

MAE stays in eV and is easy to interpret. RMSE penalizes large errors more strongly. $R^2$ measures variance explained relative to the target mean.

# 24. Complete forward traces

## 24.1 Transformer trace

1. Read PSMILES string.
2. Tokenize with chemistry-aware grammar.
3. Convert tokens to IDs and build padding mask.
4. Lookup token and positional embeddings.
5. Run two Transformer encoder layers.
6. Mask and mean-pool sequence positions.
7. Historically, project through two 32-D branch heads.
8. Concatenate branch vectors and regress one normalized scalar.
9. Inverse-transform to eV.

## 24.2 GraphMPNN trace

1. Parse SMILES with RDKit.
2. Construct atom features, adjacency, self-loops, and mask.
3. Project 7-D atom features to 64-D hidden states.
4. For each of three graph layers, average normalized neighbor states and combine node + message linear projections.
5. Apply ReLU and mask padded nodes.
6. Mean-pool valid nodes into a 64-D graph vector.
7. Apply a 64→32→1 MLP head.
8. Inverse-transform the normalized scalar to eV.

# 25. What this volume establishes

The Transformer failure and GraphMPNN success are not magic architectural labels. They follow from what each model treats as its primitive object. The Transformer sees a token ordering; the GraphMPNN sees a molecular graph. That representational choice determines which invariances are learned and which are built in.
