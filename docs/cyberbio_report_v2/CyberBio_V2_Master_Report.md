---
title: "CyberBio V2 — Complete Technical Report"
author: "Pedamallu Sai Mrudula"
date: "September 2026"
---

# CyberBio V2 — Complete Technical Report

This is the single-source combined edition of the audited CyberBio V2 documentation. The rendered PDF edition contains 40 pages and four top-level bookmarks: Overview, Part I, Part II, and Part III.

\newpage

> **Evidence policy.** Statements labelled **Project fact** are grounded in the current `Mrudula-itsjuzme/cyberbio` repository. Sections labelled **Mathematical background** explain standard theory needed to understand the implementation. Historical results are kept only when their status is explicit: canonical, historical, rejected, blocked, or developmental-unverified.

# How to use this document

This is the map, not the warehouse. It explains what the project became, why each major turn happened, what is currently canonical, and where to go for the full derivations. It intentionally avoids drowning the reader in result tables.

The technical documentation is split into three companion volumes:

1. **Part I — Foundations, Transformer, and GraphMPNN**: representations, tokenization, scaling, the Transformer from first principles, the GraphMPNN from first principles, and exact code-to-math mappings.
2. **Part II — Attacks, Search, Defenses, and Physical Validation**: threat models, perturbation semantics, validators, bounded search, query accounting, defenses, the physical-oracle problem, and the HPC handoff.
3. **Part III — Experiments, Audits, Evolution, and Canonical Evidence**: the full experimental chronology, rejected hypotheses, bugs found and repaired, canonical result tables, Framework V2 forensic audit and verified rerun, Bio-Cyber halt, reproducibility, and the final claims ledger.

# 1. What CyberBio is

**Project fact.** CyberBio is an adversarial robustness project for scientific machine learning. Its active materials track asks a deceptively simple question:

> When a model predicts a polymer property from a digital representation, is it responding to the underlying chemistry, to the arbitrary serialization of that chemistry, or to shortcuts in the data?

The active target is the **electronic bandgap** $E_g$ of polymer chains, measured in electron-volts (eV), using the polyVERSE `bandgap_chain.csv` lineage with 4,209 records.

The repository also contains a synthetic Bio-Cyber branch. That branch was deliberately **halted** after a leakage audit showed that a regularized 3-mer linear baseline reached 98.9% accuracy. The project therefore does not use the Bio-Cyber deep-learning results as scientific evidence.

# 2. The central scientific split

The project eventually learned that two superficially similar perturbations are scientifically different.

## 2.1 Representation-preserving changes

A molecule can have multiple valid SMILES strings that encode the same molecular graph. If $x$ and $x'$ are two serializations of the same graph, then the physical polymer is unchanged. Ideally,

$$
\mathrm{mol}(x')=\mathrm{mol}(x)
\quad\Longrightarrow\quad
f(x')\approx f(x).
$$

This is a true invariance problem. The label may be inherited because the chemistry did not change.

## 2.2 Chemistry-changing changes

Substitution, deletion, insertion, and related edits can change the molecular graph. In that regime,

$$
\mathrm{mol}(x')\neq \mathrm{mol}(x),
$$

so the real bandgap can legitimately change too. A large model prediction change is therefore **not automatically an adversarial error**. It is only prediction sensitivity until an independent physical oracle supplies the true changed property.

That distinction is the spine of the final project.

# 3. The project story in six turns

## Turn 1 — Sequence model baseline

The project began with PSMILES strings and a Transformer regressor. The Transformer was useful because it gave a controlled sequence model whose internal mechanics, attacks, and failure modes could be studied explicitly.

## Turn 2 — Equivalent-SMILES vulnerability

Randomized but chemically identical SMILES exposed a representation problem. The same molecule could receive noticeably different predictions depending only on serialization. This showed that a sequence model can learn the string as much as the chemistry.

## Turn 3 — Defense experiments and failed semantic specialization

The project tried augmentation, consistency objectives, and a two-branch Transformer intended to separate representation invariance from chemistry sensitivity. The branches differentiated numerically, but semantic specialization was not demonstrated. Later adversarial fine-tuning also produced collapse in some phases. These negative results are preserved rather than rewritten as successes.

## Turn 4 — Architecture pivot to GraphMPNN

The project then moved to a graph-native predictor. A GraphMPNN consumes atoms and bonds rather than a particular SMILES traversal. In the final architecture audit, this removed equivalent-SMILES drift to floating-point noise while also improving clean validation error and reducing parameter count. The GraphMPNN therefore became the canonical predictor.

## Turn 5 — Bounded chemistry stress

Once representation invariance was structurally fixed, the project stress-tested the GraphMPNN with genuine chemistry edits. An early adaptive search produced an invalid 5.961 eV maximum because edits accumulated beyond the intended threat model. Phase 12B repaired the search by bounding each candidate to at most three edits from the **original** source. The repaired search found a canonical maximum around 3.19 eV.

That number is **model drift, not proven model error**.

## Turn 6 — The physical-oracle boundary

To know whether a chemistry-changing prediction is wrong, the project defined the true adversarial error against an independent physical oracle. If

$$
\Delta_G = G(x')-G(x)
$$

is the model change and

$$
\Delta_T = T(x')-T(x)
$$

is the true physical change, then the relevant error is

$$
E_G = |\Delta_G-\Delta_T|.
$$

The repository now contains a calibrated oracle design, candidate manifests, ingestion/analysis code, and a SLURM/Quantum-ESPRESSO handoff package. Execution is blocked externally by unavailable HPC physics infrastructure, not by missing ML code.

# 4. The current end-to-end architecture

```text
polyVERSE polymer record
        |
        v
clean representation and bandgap target
        |
        +-----------------------------+
        |                             |
        v                             v
sequence representation          molecular graph
(Transformer history)            (canonical GraphMPNN)
        |                             |
        v                             v
prediction in scaled space ---> inverse target scaling ---> bandgap prediction in eV
                                      |
                                      v
                              adversarial evaluator
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
          equivalent-SMILES control            chemistry-changing search
          same graph, same label               different graph, unknown true label
                    |                                   |
                    v                                   v
          invariance measurement                 prediction-drift measurement
                                                        |
                                                        v
                                             physical-oracle requirement
```

# 5. The canonical predictor at a glance

The final predictor is a compact three-layer message-passing neural network.

For each atom $i$, the graph dataset supplies a 7-dimensional feature vector containing atomic number, degree, formal charge, aromaticity, hybridization, hydrogen count, and wildcard/attachment flag. The adjacency matrix includes self-loops.

A graph layer forms normalized neighbor messages

$$
\mathbf m_i^{(\ell)}
=
\sum_j \widetilde A_{ij}\mathbf h_j^{(\ell)},
$$

then updates

$$
\mathbf h_i^{(\ell+1)}
=
\operatorname{ReLU}
\left(
W_{\text{node}}\mathbf h_i^{(\ell)}
+
W_{\text{msg}}\mathbf m_i^{(\ell)}
\right).
$$

After three layers, valid node embeddings are mean-pooled into a graph vector and passed through a small MLP regression head. Part I derives every term and maps it to `graph_predictor.py`.

# 6. Why the Transformer still matters

Although GraphMPNN is the canonical final predictor, the Transformer is not dead documentation. It is essential to the research story because it explains:

- how the original vulnerability arose;
- why padding masks, tokenization, and positional encodings matter;
- why string models can be sensitive to equivalent serializations;
- why architecture and defense effects must be separated;
- how the project discovered shortcut learning and evaluation bugs.

Part I therefore teaches the Transformer in full rather than treating it as an obsolete footnote.

# 7. Attack/search taxonomy

The repository contains multiple perturbation and search ideas, but they do not share one evidentiary status.

The evidence status is easier to read as a short taxonomy rather than a results table:

- **Equivalent / randomized SMILES:** same graph, different serialization; canonically evaluated.
- **Single substitution:** chemistry-changing local stress; canonically evaluated.
- **Single deletion:** chemistry-changing local stress; canonically evaluated.
- **Insertion:** chemistry-changing edit; mentioned historically, but not promoted as a canonical fixed-bank result.
- **Bounded multi-substitution search:** up to three edits from the source under the stated threat model; canonically evaluated.
- **Random / greedy / Metropolis search:** canonically evaluated in Phase 12B; Framework V2 later verifies strategy benchmarking under a different operator definition.
- **Evolutionary search:** verified in Framework V2 `verified_run_1` under its own stated model/operator/budget contract.
- **Motif / aliphatic-carbon operator:** implemented, but operator capacity and edit accounting require separate reporting.
- **Scaffold-preserving operator:** implemented direction, not promoted as canonical scientific evidence.
- **LLM / RL / GAN directions:** proposal or feasibility work, not canonical measured evidence.

# 8. Why “exact” means tracking protocol, not just numbers

The repository contains historical documents, superseded runs, repaired scripts, and a verified Framework V2 rerun. Therefore a number is only meaningful together with:

- model checkpoint and scaler;
- source split;
- perturbation operator;
- edit definition;
- query budget;
- validator;
- whether the candidate changes chemistry;
- whether the result is canonical, historical, rejected, blocked, or developmental.

For example, a string Levenshtein distance can badly overstate chemical edit count after canonicalization, while a graph-based atom-change measure may still show a single edit. Version 2 of the documentation treats those protocol details as first-class scientific objects.

# 9. What the project can claim

**Supported:**

- sequence serialization can create a real representation-invariance failure;
- graph-native modeling structurally removes that particular serialization dependence;
- bounded chemistry edits can induce substantial prediction sensitivity;
- search strategies can be compared under fixed model/operator/budget contracts;
- a chemistry-changing drift cannot be equated with physical error without an independent property oracle;
- the Bio-Cyber synthetic dataset used here contains severe k-mer leakage and is unsuitable for the original DL claim.

**Not supported:**

- every chemistry-changing high-drift example is an adversarial failure;
- 3.19 eV is a confirmed physical prediction error;
- the invalid 5.961 eV developmental maximum;
- GraphMPNN is universally robust;
- low chemistry drift is always desirable;
- the historical DFT functional/basis set for `bandgap_chain.csv` is known;
- a GAN, RL attacker, or real-provider LLM attack was established as a canonical result.

# 10. Current external boundary

The ML and software side now contains an HPC oracle framework with:

- backend contracts;
- Quantum ESPRESSO and VASP skeleton backends;
- deterministic calibration and adversarial manifests;
- SLURM job generation;
- result ingestion;
- calibration metrics;
- adversarial $\Delta_T$ / error analysis;
- uncertainty bookkeeping.

The remaining blocker is physical execution: a real cluster, pseudopotentials/POTCAR, and scientifically justified DFT settings. The project is therefore at a clean handoff boundary rather than an unfinished-model boundary.

# 11. Recommended reading order

Read this overview first. Then use **Part I** if you need to defend the model mathematics, **Part II** if you need to explain how the attacks/search/defenses/oracle work, and **Part III** if someone asks “which experiment actually supports that claim?”

\newpage

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

\newpage

> **Evidence policy.** Statements labelled **Project fact** are grounded in the current `Mrudula-itsjuzme/cyberbio` repository. Sections labelled **Mathematical background** explain standard theory needed to understand the implementation. Historical results are kept only when their status is explicit: canonical, historical, rejected, blocked, or developmental-unverified.

# Scope of this volume

This volume answers: **how are candidates generated, what does “valid” mean, what does the search optimize, how are query budgets enforced, what defenses were attempted, and why does chemistry-changing robustness require an external oracle?**

# 1. Threat model

Let $f$ be a fixed property predictor and $x$ a clean polymer representation. For a feasible candidate set $\mathcal C(x)$, a generic untargeted sensitivity search is

$$
x^*=
\arg\max_{x'\in\mathcal C(x)}
J(x'),
$$

with

$$
J(x')=|f(x')-f(x)|.
$$

This objective measures **prediction drift**. It becomes a true adversarial-error objective only if the underlying physical label should remain unchanged, or if an independent oracle supplies the changed ground truth.

# 2. Two semantics, two rules

## 2.1 Equivalent-SMILES control

If canonical graph identity is preserved,

$$
\operatorname{canon}(x')=\operatorname{canon}(x),
$$

then the project treats the candidate as representation-preserving. The original physical target may be reused.

## 2.2 Chemistry-changing stress

For substitutions, deletions, insertions, and bounded multi-edit search,

$$
\operatorname{canon}(x')\neq\operatorname{canon}(x).
$$

The original $E_g$ must **not** be assumed to remain correct.

# 3. Equivalent-SMILES generation

`SmilesRandomizationAttack` parses the source molecule with RDKit, then uses a seeded randomized-SMILES vector generator. The candidate is accepted as a representation control only when it is a different serialization of the same molecular graph.

The representation drift is

$$
\Delta_{repr}(x,x')=|f(x')-f(x)|.
$$

For a graph-native model, this should collapse to numerical noise because both strings produce the same graph.

# 4. Chemical plausibility gates

The project uses multiple layers of chemical checks depending on experiment generation.

## 4.1 RDKit parse and sanitization

A candidate must parse into a molecular graph and pass sanitization/valence checks. This rejects malformed strings, but does not prove synthesizability.

## 4.2 Attachment-star preservation

For polymer repeat units, wildcard `*` attachment count is preserved:

$$
\#\star(x')=\#\star(x).
$$

## 4.3 Molecular-weight bound

Some historical validators require

$$
0.5\le \frac{MW(x')}{MW(x)}\le 1.5.
$$

## 4.4 Morgan fingerprint Tanimoto similarity

For binary fingerprint vectors $a,b$,

$$
T(a,b)=
\frac{|a\cap b|}{|a|+|b|-|a\cap b|}.
$$

Historical MCMC-style experiments often used

$$
T\ge 0.5.
$$

This is a structural-similarity constraint, not a proof of label preservation.

# 5. Local chemistry-changing operators

## 5.1 Substitution

A token- or atom-level substitution replaces one mutable chemical unit. Different experiment generations used different operator semantics, so the operator must always be named together with its implementation.

A single substitution can be written abstractly as

$$
x'=\operatorname{substitute}(x,i,a\to b).
$$

## 5.2 Deletion

Deletion removes a mutable unit, subject to validity:

$$
x'=\operatorname{delete}(x,i).
$$

Deletion is chemistry-changing and was evaluated as a fixed stress in the canonical architecture work.

## 5.3 Insertion

Insertion exists in historical attack inventories, but the release audit marks single-edit insertion as **not canonically evaluated** in the final fixed-bank/search claims. Version 2 therefore does not silently promote it to a final result.

## 5.4 Rearrangement

Rearrangement appears in historical sequence-model experiments. It belongs to the Transformer-era robustness story, not the final GraphMPNN canonical chemistry-stress result.

# 6. Historical Metropolis-style sequence search

The earlier `ProbabilisticMCMCAttack` chooses among substitution, insertion, and deletion proposals, scores valid candidates, and accepts/rejects with a Metropolis-style rule.

Its total score is approximately

$$
S(x')=C(x')+|f(x')-f(x_0)|,
$$

where $C$ is a coarse chemical-validity score.

For current state $x_t$ and proposal $x'$, it uses

$$
\Delta=S(x')-S(x_t),
$$

$$
p_{acc}
=
\begin{cases}
1,&\Delta>0,\\
\exp(\Delta/T),&\Delta\le 0.
\end{cases}
$$

The implementation is explicitly documented as **Metropolis-style model-guided stochastic search**, not formal Metropolis-Hastings, because it does not include the reverse proposal ratio

$$
\frac{q(x_t\mid x')}{q(x'\mid x_t)}.
$$

# 7. Phase 12B bounded adaptive search

The Phase 12B repair changed the center of gravity of the attack story. Candidates are constrained to be at most three token edits from the **original** source, not merely from the previous state.

## 7.1 Levenshtein recurrence used in Phase 12B

For token sequences $a_{1:m}$ and $b_{1:n}$, define a dynamic-programming table $D$. Boundary conditions:

$$
D(i,0)=i,
\qquad
D(0,j)=j.
$$

Recurrence:

$$
D(i,j)=
\min
\begin{cases}
D(i-1,j)+1,\\
D(i,j-1)+1,\\
D(i-1,j-1)+\mathbf 1[a_i\neq b_j].
\end{cases}
$$

Phase 12B rejects candidates with

$$
D(x_{source},x')>3.
$$

Important later audit nuance: Framework V2 showed that canonicalized SMILES string distance can overstate **chemical** edit count. Therefore Phase 12B’s token-distance threat model and Framework V2’s graph-based edit accounting are related but not interchangeable.

# 8. Random search

Random search repeatedly samples valid neighbors of the source and keeps the candidate with the largest drift.

For evaluated candidates $x'_1,\ldots,x'_Q$,

$$
x^*=\arg\max_k J(x'_k).
$$

Its strength is simplicity and low algorithmic bias; its weakness is inefficient exploitation.

# 9. Greedy search

Greedy search generates a small neighborhood from the current state and moves to the strongest improving candidate.

If $\mathcal N(x_t)$ is the local candidate batch,

$$
x_{t+1}
=
\arg\max_{x'\in\mathcal N(x_t)}J(x')
$$

when the best candidate improves the stored maximum. This is strongly exploitative and can get stuck in local optima.

# 10. Phase 12B Metropolis search

For drift

$$
J_t=|f(x_t)-f(x_0)|,
$$

and candidate drift $J'$, the repaired Phase 12B search uses

$$
\Delta=J'-J_t,
$$

with $T=0.1$ in `run_phase12b_adaptive_search_audit.py`:

$$
p_{acc}=\begin{cases}
1,&\Delta>0,\\
\exp(\Delta/T),&\Delta\le 0.
\end{cases}
$$

Again, this is a search heuristic, not a stationary-distribution MCMC claim.

# 11. Cross-model transfer

A candidate optimized for model $A$ can be rescored by model $B$ without charging the search budget if transfer evaluation is explicitly outside the attack budget. Define

$$
J_A(x')=|f_A(x')-f_A(x)|,
$$

$$
J_B(x')=|f_B(x')-f_B(x)|.
$$

Transfer asks whether high-sensitivity candidates are architecture-specific or shared.

# 12. Framework V2: why the forensic audit mattered

The first Framework V2 developmental benchmark was invalid for three independent reasons:

1. it used a Python hash stub instead of the GraphMPNN;
2. its sources came from train/test instead of validation;
3. query accounting was broken.

Those 6–7 eV mean drifts are therefore forbidden evidence.

The repaired `GraphMPNNPredictor` binds the canonical checkpoint, canonical scaler, strict graph parsing, and auditable model-call accounting. A later `verified_run_1` is the only Framework V2 benchmark that may be cited.

# 13. Exact query accounting

A declared query budget $Q$ must count every target-model candidate evaluation consistently. The repaired contract is:

$$
q_{used}\le Q.
$$

The source prediction can be computed once and explicitly treated as uncharged if the protocol says so. All candidate scoring must flow through one counting path; otherwise a search can silently exceed its budget.

Drift efficiency can be reported as

$$
\eta_{query}=\frac{J(x^*)}{q_{used}}.
$$

# 14. Evolutionary search

Framework V2 includes population-based evolutionary search under the verified benchmark. General structure:

1. maintain a population of valid candidates;
2. score fitness with $J(x')$;
3. select high-fitness candidates;
4. mutate/recombine under the operator and validator;
5. preserve elites;
6. stop at the query budget.

It explores more broadly than greedy search but makes query accounting and duplicate control especially important.

# 15. Operator semantics are part of the threat model

The forensic audit found that Framework V2 atom substitution and Phase 12B token substitution are not identical attacks. Framework V2 can touch aromatic atoms under its operator; the Phase 12B token mutator did not mutate lowercase aromatic tokens. Therefore numerical comparisons across the two threat models must not be presented as if only the search strategy changed.

Similarly, a historical “functional group replacement” operator was renamed conceptually because the implementation behaved more like aliphatic-carbon substitution and could consume multiple independent graph edits per application.

# 16. Historical sequence-model defenses

## 16.1 Random-SMILES augmentation

For equivalent representations $x$ and $x_r$ with the same true label $y$, training can safely include both:

$$
\mathcal L_{aug}
=
\ell(f_\theta(x),y)+\ell(f_\theta(x_r),y).
$$

This is scientifically clean because the graph identity is unchanged.

## 16.2 Representation consistency

An additional consistency objective can be

$$
\mathcal L_{inv}
=
\|f_\theta(x)-f_\theta(x_r)\|_2^2.
$$

## 16.3 Chemistry-changing label-free consistency

For a chemistry-changing candidate $x_c$, the original physical label must not be inherited. A historical regularization approach instead used detached model behavior:

$$
\mathcal L_{cons}
=
\ell
\left(
 f_\theta(x_c),
 \operatorname{stopgrad}(f_\theta(x))
\right).
$$

This can reduce sensitivity but does **not** prove improved physical accuracy.

# 17. Historical two-branch specialization objectives

The project tried to separate representation invariance and chemistry sensitivity in latent space.

Property loss:

$$
\mathcal L_{prop}=\ell(f(x),y).
$$

Representation branch:

$$
\mathcal L_{inv}
=
\|z_{repr}(x)-z_{repr}(x_r)\|_2^2.
$$

Chemistry-sensitivity margin:

$$
\mathcal L_{sens}
=
\max
\left(
0,
 m-\|z_{chem}(x)-z_{chem}(x_c)\|_2
\right).
$$

A diversity penalty discouraged identical branches. The experiment did not establish semantic disentanglement; branch polarization/collapse provided a cheaper route through the objective.

# 18. Why the canonical final model needs less representation defense

Once the GraphMPNN became canonical, equivalent-SMILES invariance was solved structurally. The representation-defense question therefore moved from “how do we regularize a Transformer to ignore serialization?” to “which representation makes serialization irrelevant by construction?”

That architectural pivot is more important than any one historical defense table.

# 19. The oracle problem

For chemistry-changing candidates, define model change

$$
\Delta_G=G(x')-G(x)
$$

and unknown true physical change

$$
\Delta_T=T(x')-T(x).
$$

The physically meaningful adversarial error is

$$
E_G=|\Delta_G-\Delta_T|.
$$

A large $|\Delta_G|$ alone cannot tell whether the model is oversensitive, undersensitive, or accurate.

# 20. Oracle classes

The repository distinguishes conceptually:

- **Matched oracle**: reproduces the same physical property/protocol as the dataset;
- **Calibratable surrogate oracle**: computes a related property and requires empirical calibration;
- **Mismatched oracle**: computes a different physical target and cannot be used directly.

The exact DFT functional/basis/pseudopotential setup that generated the local polyVERSE `bandgap_chain.csv` values is unresolved. Therefore a future matched-oracle claim must begin with calibration rather than assuming PBE or HSE06.

# 21. HPC oracle software architecture

The repository now includes a backend-agnostic HPC framework with implemented contracts for:

- backend interface;
- partial Quantum ESPRESSO and VASP backends;
- structure preparation provenance;
- deterministic calibration/adversarial manifests;
- SLURM array generation;
- backend checks;
- bundle export;
- output ingestion;
- calibration MAE/RMSE/$R^2$/Pearson analysis;
- adversarial $\Delta_T$, model error, and sensitivity analysis;
- uncertainty bookkeeping.

Execution remains externally blocked by missing cluster access and physically justified quantum-chemistry inputs such as pseudopotentials and exact parameter choices.

# 22. Attack → oracle → defense loop

A scientifically complete chemistry-changing loop is:

```text
source polymer x
      |
      v
bounded candidate generation x'
      |
      v
validity + threat-model checks
      |
      v
model prediction drift ΔG
      |
      v
independent physical oracle → ΔT
      |
      v
true response error E = |ΔG - ΔT|
      |
      v
oracle-backed robust training / recalibration
```

The current project reaches the oracle handoff but does not fabricate the missing physical labels.

# 23. Final status table

| Component | Scientific status |
|---|---|
| Equivalent-SMILES control | canonical |
| Single substitution/deletion stress | canonical |
| Phase 12B bounded ≤3 search | canonical |
| 5.961 eV Phase 12 maximum | rejected |
| Framework V2 developmental hash-stub numbers | rejected |
| Framework V2 `verified_run_1` | verified under its own stated operator/budget protocol |
| Sequence-model augmentation/consistency | historical evidence |
| Chemistry-changing drift as physical error | blocked without oracle |
| HPC oracle execution | software prepared, external physical compute blocked |

\newpage

> **Evidence policy.** Statements labelled **Project fact** are grounded in the current `Mrudula-itsjuzme/cyberbio` repository. Sections labelled **Mathematical background** explain standard theory needed to understand the implementation. Historical results are kept only when their status is explicit: canonical, historical, rejected, blocked, or developmental-unverified.

# Scope of this volume

This volume is the evidence ledger. It answers: **what happened, in what order, which claims survived audit, which numbers are canonical, what failed, what was repaired, and what remains externally blocked?**

# 1. Repository lineages

## 1.1 Historical OpenPoly $T_g$ lineage

The earliest materials experiments used a small glass-transition-temperature dataset. This lineage produced useful lessons about sample size, shortcut learning, scaling hygiene, and invalid label inheritance, but it is archived and must not be mixed with the active bandgap pipeline.

## 1.2 Active polyVERSE bandgap lineage

The active dataset is the Ramprasad Group polyVERSE `Other/bandgap_chain.csv` lineage with 4,209 rows. The target is a 1D periodic polymer-chain electronic bandgap in eV. Exact historical DFT functional and basis/pseudopotential details are not recovered from the available local metadata.

# 2. Chronological research evolution

## Phase 1 — Establish the sequence baseline

A small Transformer regressor was used as the controlled sequence-to-property target. This provided a working predictor and an attackable discrete representation.

## Phase 2 — Expose representation sensitivity

Equivalent/Random-SMILES experiments showed that the same molecular graph could produce different Transformer outputs. This reframed the project from generic “adversarial strings” to a specific invariance problem.

## Phase 3 — Test latent semantic specialization

A two-branch Transformer attempted to learn one representation-invariant latent path and one chemistry-sensitive path. Frozen-bank evaluation did not demonstrate the intended semantic specialization. The branches could satisfy the loss through variance polarization rather than interpretable disentanglement.

## Phase 4 — Move robustness supervision to outputs

The project tested output-level consistency and teacher-style regularization. These experiments are historically informative, but chemistry-changing labels had to be reclassified carefully because original-label inheritance is physically unjustified.

## Phases 5–10 — Attribution, transfer, shortcut, and defense audits

The project accumulated evidence on representation attribution, sequence-length shortcuts, frozen candidate banks, cross-attack transfer, and robustness objectives. Some post-hoc adversarial fine-tuning phases exhibited collapse rather than a durable invariance solution.

## Phase 11B — Architecture comparison bug

An architecture comparison initially made the Transformer look far worse because evaluation omitted the padding mask. Mean pooling therefore included corrupted padded states. This result was rejected.

## Phase 11C — Canonical architecture verification

The padding path was repaired, historical checkpoints were re-verified, and the architecture frontier was recomputed under one consistent evaluator. GraphMPNN emerged as the canonical predictor.

## Phase 12 — Graph-native chemistry stress and a second bug

GraphMPNN was stress-tested under chemistry-changing search. An adaptive process accumulated edits relative to previous states, not strictly relative to the original source, producing an invalid 5.961 eV maximum.

## Phase 12B — Bounded-search repair

The search was rebuilt with a strict $\le3$ edit boundary from the original source. The repaired maximum became approximately 3.19 eV under the stated Phase 12B threat model.

## Phases 13–14 — Physical oracle design and feasibility

The project formalized the missing ground truth for chemistry-changing candidates, traced dataset provenance, built calibration manifests, and established that local quantum-chemistry execution was unavailable.

## HPC oracle package — external handoff

A backend-neutral SLURM workflow, QE/VASP scaffolding, manifests, ingestion, and analysis were implemented. Real execution remains outside the current environment.

## Framework V2 — forensic failure, then verified rerun

A developmental generic-attack benchmark was found to use a hash stub, contaminated source split, and invalid query accounting. Those numbers were rejected. The pipeline was repaired with a real hash-validated GraphMPNN wrapper and strict parsing. `verified_run_1` is the only Framework V2 result set that may be cited.

# 3. Canonical architecture table

The repository’s authoritative Phase 11C/12B table is:

### 3.1 Clean model quality and status

| Model | Parameters | Validation MAE | RMSE | $R^2$ | Status |
|---|---:|---:|---:|---:|---|
| Transformer Ordinary | 85,761 | 0.486 | 0.667 | 0.785 | canonical sequence baseline |
| Transformer ArchControl | 90,049 | 0.461 | 0.642 | 0.800 | historical |
| Transformer MixedRobust | 90,049 | 0.441 | 0.612 | 0.819 | historical |
| Transformer Augmented | 90,049 | 0.501 | 0.696 | 0.766 | historical |
| **GraphMPNN Small** | **27,585** | **0.411** | **0.595** | **0.829** | **canonical predictor** |

### 3.2 Stress/invariance metrics from the same canonical table

| Model | Equivalent-SMILES drift (eV) | Substitution drift (eV) | Deletion drift (eV) |
|---|---:|---:|---:|
| Transformer Ordinary | 0.614 | 0.291 | 0.433 |
| Transformer ArchControl | 0.382 | 0.281 | 0.362 |
| Transformer MixedRobust | 0.308 | 0.269 | 0.289 |
| Transformer Augmented | 0.147 | 0.281 | 0.169 |
| **GraphMPNN Small** | **~0** | **0.365** | **0.217** |

GraphMPNN equivalent-SMILES drift is reported as approximately $4.8\times10^{-8}$ eV from floating-point numerical effects.

# 4. What Phase 11C actually proved

Phase 11C did three things simultaneously:

1. recovered the expected historical Transformer metrics after fixing the padding path;
2. compared the sequence and graph architectures under a consistent evaluator;
3. showed that GraphMPNN achieved structural serialization invariance while also improving clean validation MAE with far fewer parameters.

The project therefore changed canonical predictor, not just defense hyperparameters.

# 5. Phase 12B bounded-search evidence

Under the Phase 12B protocol:

- sources were fixed validation molecules;
- search budgets were $Q\in\{10,20,50\}$;
- seeds were 42, 1337, and 7;
- methods were random, greedy, and Metropolis-style;
- every candidate had to remain within at most three token edits of the original source;
- cross-model transfer was measured outside the search budget.

At $Q=50$, the Phase 12B report gives mean GraphMPNN drift approximately:

| Search | Mean GraphMPNN drift (eV) |
|---|---:|
| Random | 0.868 |
| Greedy | 0.967 |
| Metropolis | 0.764 |

Metropolis found the absolute repaired maximum of about 3.19 eV, while greedy had the higher mean in that experiment.

# 6. Why 5.961 eV is invalid

The old Phase 12 stateful search allowed **edit creep**. A candidate could be within a small local step of the previous state while becoming much farther than three edits from the original source. That violated the intended threat model.

Therefore:

$$
5.961\text{ eV}\quad\text{is rejected evidence.}
$$

This is not a “slightly outdated number”; it is tied to a protocol bug.

# 7. Framework V2 developmental audit

The developmental Framework V2 benchmark is also rejected evidence. The forensic audit found:

- placeholder predictor: `float(hash(smiles) % 1000)/100`;
- source split: 25 train, 5 test, 0 validation;
- 482/540 runs exceeded declared query budget;
- the budget counter understated model asks in all 540 runs.

The apparent 6–7 eV average drifts were a hash-stub artifact, not a chemical-model result.

# 8. Framework V2 verified rerun

The repaired benchmark uses:

- canonical GraphMPNN checkpoint validated against the release manifest;
- canonical scaler;
- validation-only source selection;
- strict graph parsing;
- one budgeted scoring path;
- operator fixed when comparing search strategies;
- explicit distinction between operator count and graph edit count.

The verified run reproduced the canonical clean validation metrics exactly before any attack comparison. That preflight is essential: a benchmark is meaningless if it cannot first reproduce the model it claims to attack.

The current repository stores the repaired results under `results/framework_v2/verified_run_1/` and companion tables under `docs/SEARCH_COMPARISON_TABLE.md` and `docs/ATTACK_COMPARISON_TABLE.md`.

# 9. A subtle edit-distance audit lesson

The forensic audit found that SMILES Levenshtein distance is not a reliable chemical edit count after canonicalization. A one-atom substitution can trigger a large string rewrite. Therefore:

- Phase 12B’s token-distance rule defines **that specific threat model**;
- Framework V2 uses graph-aware edit accounting for its own operator protocol;
- the two must not be merged into a single “3 edits” claim without stating the edit definition.

# 10. Clean scaler and unit audit

The Framework V2 audit verified:

- canonical scaler equals a training-split fit to numerical tolerance;
- inverse scaling is applied exactly once;
- prediction drift is computed in eV;
- no scaled-vs-unscaled subtraction is used;
- the GraphMPNN wrapper reproduces the canonical evaluator exactly.

These checks matter because a double inverse transform or unit mismatch can create impressive-looking nonsense without any attack bug.

# 11. GraphDataset strict-mode correction

Historical `GraphDataset(strict=False)` silently advanced to another dataset row if a SMILES failed parsing. That path never became a canonical attack result, but it was a latent evaluation hazard.

The repaired adversarial predictor uses `strict=True`, so malformed candidates raise and are rejected rather than being scored as a neighboring molecule.

# 12. Data leakage and split audits

The repository distinguishes several leakages:

- exact canonical-identity overlap;
- scaffold overlap under random splitting;
- target-scaler fitting leakage;
- attack-candidate overlap;
- source-split contamination in the rejected Framework V2 developmental benchmark.

The lesson is that “no exact duplicate rows” is not enough for materials ML. Structural scaffold sharing can still make evaluation substantially easier.

# 13. Shortcut learning

Sequence length was shown to correlate with bandgap and with model behavior. The project uses this as evidence of partial shortcut reliance, not proof that sequence length fully explains the model.

A one-dimensional length-only linear model was strong enough to serve as a nontrivial baseline in earlier audits, while the Transformer still substantially outperformed it. This makes shortcut analysis a diagnostic rather than a claim that the model “only learned length.”

# 14. Explainability evidence and corrections

The repository contains multiple explainability phases. Version 2 separates developmental observations from the later canonical gradient ablation. Earlier documents that claimed unusually large star-token gradients are not treated as final when the canonical benchmark reports a zero mean boundary-star gradient in its specific ablation.

Attention weights are also not treated as causal explanations. Occlusion, gradient attribution, and shortcut controls are interpreted as complementary diagnostics.

# 15. Defense evidence in proper context

Historical defense experiments showed that Random-SMILES augmentation can reduce representation drift, and chemistry-changing consistency regularization can reduce sensitivity to a perturbation family. However:

- representation invariance became structurally solved by GraphMPNN in the final architecture;
- chemistry-changing consistency is not physical supervision;
- post-hoc adversarial training also produced collapse in some phases;
- final claims therefore emphasize architecture and oracle-backed scientific validity over “defense wins.”

# 16. Bio-Cyber branch halt

The synthetic Bio-Cyber dataset was audited with simple k-mer logistic-regression baselines:

| Features | Accuracy |
|---|---:|
| 1-mer | 50.9% |
| 2-mer | 93.6% |
| 3-mer | 98.9% |

Because a trivial 3-mer linear model nearly solved the classification task, all downstream deep-learning attack/defense claims were halted. Preserving this failure is part of the project’s methodological quality control.

# 17. Physical-oracle provenance lock

The target dataset is verified as polyVERSE `Other/bandgap_chain.csv`, 4,209 rows, 1D periodic polymer-chain bandgap, computational DFT, eV. What remains unknown is the exact historical functional and basis/pseudopotential setup.

Therefore claims such as “the dataset used HSE06” or “the dataset used PBE” are not promoted as exact-dataset facts without direct evidence.

# 18. Oracle calibration logic

A candidate physical oracle must first be tested on a deterministic calibration set. Useful metrics include

$$
\mathrm{MAE},\quad \mathrm{RMSE},\quad R^2,\quad r_{Pearson},\quad \text{bias}.
$$

Only after acceptable alignment should it be used to evaluate adversarial source/candidate deltas.

The project refuses to invent a universal threshold in advance because acceptability depends on the scientific purpose and the property mismatch.

# 19. HPC implementation status

The repository marks the following as implemented: backend contracts, structure-preparation provenance, calibration/adversarial manifests, SLURM generation, backend checks, job export, result ingestion, calibration analysis, adversarial error analysis, and uncertainty bookkeeping.

Quantum ESPRESSO and VASP backends are partial because real physics inputs and external executables/licensed data are required. The blocker is therefore external compute/parameterization.

# 20. Canonical claims ledger

| Claim | Final status |
|---|---|
| Sequence models natively learn chemical equivalence | rejected |
| Branch specialization fixes equivalent-SMILES drift | rejected |
| Post-hoc representation adversarial training guarantees robustness | rejected |
| GNN representation eliminates equivalent-SMILES drift | supported |
| Bounded chemistry edits produce substantial model sensitivity | supported |
| 5.961 eV is a valid bounded result | rejected |
| Phase 11B architecture metrics were accurate | rejected |
| Chemistry-changing prediction drift equals physical error | blocked without oracle |
| Exact dataset functional/basis is known | not established |

# 21. What can and cannot be inferred from 3.19 eV

The repaired bounded maximum says:

> under that GraphMPNN, candidate generator, edit definition, validity checks, search procedure, and query regime, the model prediction could be moved by about 3.19 eV.

It does **not** say:

> the model was wrong by 3.19 eV.

The missing quantity is the true physical response of the edited polymer.

# 22. Reproducibility map

Key current evidence files include:

- `docs/CANONICAL_RESULTS_TABLE.md`
- `docs/CLAIMS_LEDGER.md`
- `docs/RELEASE_AUDIT.md`
- `docs/PHASE11C_CANONICAL_ARCHITECTURE_VERIFICATION.md`
- `docs/PHASE12B_ADAPTIVE_SEARCH_AUDIT.md`
- `docs/PHASE13_INDEPENDENT_BANDGAP_ORACLE.md`
- `docs/PHASE13A_BANDGAP_PROVENANCE_LOCK.md`
- `docs/PHASE14_ORACLE_FEASIBILITY_AND_CALIBRATION.md`
- `docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md`
- `docs/FRAMEWORK_V2_ATTACK_RESULTS.md`
- `results/framework_v2/verified_run_1/`
- `RELEASE_MANIFEST.json`
- `hpc_oracle/`

# 23. Final research position

The final project is not best summarized as “we made an adversarially trained Transformer.” It is better summarized as:

1. we exposed a serialization vulnerability in a polymer sequence predictor;
2. we tested and rejected several inadequate internal/defense fixes;
3. we solved serialization invariance structurally with a graph model;
4. we built bounded chemistry-changing searches and repaired their protocol bugs;
5. we showed that high chemistry sensitivity cannot be called adversarial error without independent physics;
6. we built the software bridge to that physical oracle and stopped at the honest external-compute boundary.
