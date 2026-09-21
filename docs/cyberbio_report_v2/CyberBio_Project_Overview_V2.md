---
title: "CyberBio Project Overview — Version 2"
author: "CyberBio Project Documentation"
date: "September 2026"
---

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
