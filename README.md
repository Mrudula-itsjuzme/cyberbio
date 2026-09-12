# DL Cyberbio: Adversarial Attacks and Defenses in Sequence Models

This repository explores adversarial machine learning vulnerabilities and defenses in deep learning sequence models across two primary domains: **Materials Science** and **Bio-Cybersecurity**.

The repository is structured into two main active sub-projects:

---

## 1. Materials Adversarial (`materials-adversarial/`)
**Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling**

This project focuses on sequence-based deep learning models predicting polymer physical properties directly from 1D representations (PSMILES).

> [!IMPORTANT]
> **Lineage Disambiguation**: The `materials-adversarial` project evolved through two distinct scientific lineages:
> 1. **Historical Exploratory Lineage (OpenPoly Tg / K)**: Exploratory research on polymer glass-transition temperature ($T_g$ in Kelvin) using a small dataset (~247 samples). Key findings included refutation of the initial "length-changing attacks are uniquely dangerous" hypothesis, identification of representation/shortcut learning confounds, and demonstration of model global instability under SMILES randomization. *This lineage is preserved for historical research context in [`materials-adversarial/docs/HISTORICAL_TG_LINEAGE.md`](materials-adversarial/docs/HISTORICAL_TG_LINEAGE.md).*
> 2. **Active Scientific Lineage (polyVERSE Bandgap / eV)**: Current active research focused on polymer bandgap prediction ($E_g$ in eV) using 4,209 high-fidelity DFT polymer records.

### Active Lineage Overview (polyVERSE Bandgap)
*   **Target Property**: Polymer Bandgap ($E_g$ in eV).
*   **Dataset**: 4,209 usable records from polyVERSE, split deterministically (Seed `20260815`) into **2,946 train**, **631 validation**, and **632 sealed test** samples.
*   **Baseline Model**: Multi-head Transformer Encoder Regressor trained to predict Bandgap (eV). Clean Test MAE: **0.4619 eV** ($R^2 = 0.8019$).
*   **Adversarial Framework**: Evaluates model prediction drift under sequence perturbations with strict semantic classification:
    *   *Representation-Preserving Control*: SMILES randomization (canonical molecular equivalence verified).
    *   *Chemistry-Changing Stress Tests*: Single-token substitutions, insertions, deletions, local rearrangements, and Metropolis-style stochastic search.
*   **Paired Robustness Benchmark**: Evaluates clean vs defended models on identical, frozen candidate banks across the sealed 632-sample test set.

For complete active specifications, view [`materials-adversarial/README.md`](materials-adversarial/README.md) and [`materials-adversarial/docs/CANONICAL_PROJECT_STATE.md`](materials-adversarial/docs/CANONICAL_PROJECT_STATE.md).

---

## 2. Bio-Cyber Adversarial (`bio-cyber-adversarial/`)
**Bio-Cybersecurity Adversarial Benchmark**

This project establishes an experimental dataset and baseline pipeline for studying adversarial attacks on synthetic biological sequences.

*   **IMPORTANT**: This is a *synthetic computational benchmark* and does not model real biological systems or real pathogenic motifs, ensuring a safe, controlled environment.
*   **Dataset**: Generates 20,000 synthetic DNA-like sequences (Alphabet: A, C, G, T) with implanted class-specific motifs.
*   **Baseline Model**: Trains a 1D Convolutional Neural Network (CNN) classifier to identify synthetic motifs, including interpretability sanity checks.
*   **Goal**: Serves as a baseline foundation for training adversarial attack agents against sequence-based biological classifiers.

---

## Getting Started
To explore a specific domain, navigate to either `materials-adversarial/` or `bio-cyber-adversarial/` and follow the setup instructions in their respective README files.
