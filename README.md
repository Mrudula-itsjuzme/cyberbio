# DL Cyberbio: Adversarial Attacks and Defenses in Sequence Models

This repository explores adversarial machine learning vulnerabilities and defenses in deep learning sequence models across two primary domains: **Materials Science** and **Bio-Cybersecurity**.

The repository is structured into three main sub-projects:

---

## 1. Materials Adversarial (`materials-adversarial/`)
**Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling**

This project focuses on the adversarial perturbation of PSMILES polymer representations against a Glass Transition Temperature (Tg) regression target model. It uses a probabilistic (MCMC-based) generator to propose scientifically plausible modifications to material sequences.

*   **Phase 1 (Attacks):** Implements a target Transformer model (baseline MAE ~52K) and explores substitution, insertion, deletion, and local rearrangement attacks. The findings demonstrate that length-changing perturbations (insertion/deletion) cause significantly larger prediction drifts than length-preserving ones.
*   **Phase 2 (Defenses):** Implements adversarial training by augmenting the training data with the generated adversarial sequences. The results show a robust reduction in *worst-case* maximum prediction drifts against unseen attacks.
*   **Documentation:** Detailed research decisions, bug logs, and experimental results can be found in [`materials-adversarial/docs/PROJECT_WORKSPACE.md`](materials-adversarial/docs/PROJECT_WORKSPACE.md).
*   **Run It:** See [`materials-adversarial/README.md`](materials-adversarial/README.md) for environment setup and execution instructions.

---

## 2. Bio-Cyber Adversarial (`bio-cyber-adversarial/`)
**Bio-Cybersecurity Adversarial Benchmark**

This project establishes an experimental dataset and baseline pipeline for studying adversarial attacks on synthetic biological sequences.

*   **IMPORTANT:** This is a *synthetic computational benchmark* and does not model real biological systems or real pathogenic motifs, ensuring a safe, controlled environment.
*   **Dataset:** Generates 20,000 synthetic DNA-like sequences (Alphabet: A, C, G, T) with implanted class-specific motifs.
*   **Baseline Model:** Trains a 1D Convolutional Neural Network (CNN) classifier to identify the synthetic motifs. It includes interpretability sanity checks to verify the model hasn't simply learned dataset artifacts.
*   **Goal:** This baseline serves as the foundation for future work training adversarial attack agents against sequence-based biological classifiers.
*   **Run It:** See [`bio-cyber-adversarial/README.md`](bio-cyber-adversarial/README.md) for generation and training instructions.

---

## 3. Project Specifications (`Project/`)

This directory contains the original planning and architecture documents that guided the development of the materials adversarial framework.

*   **`project_spec_two_phase.md`:** The master build specification detailing the 4-week, two-phase implementation plan (Foundation + Probabilistic Attack Engine, followed by Defense and Evaluation) that was used to construct the `materials-adversarial` pipeline.

---

## Getting Started
To explore a specific domain, navigate to either the `materials-adversarial` or `bio-cyber-adversarial` directories and follow the setup instructions in their respective README files.
