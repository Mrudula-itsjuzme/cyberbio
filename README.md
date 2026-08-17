# DL Cyberbio: Adversarial Attacks and Defenses in Sequence Models

This repository explores adversarial machine learning vulnerabilities and defenses in deep learning sequence models across two primary domains: **Materials Science** and **Bio-Cybersecurity**.

The repository is structured into three main sub-projects:

---

## 1. Materials Adversarial (`materials-adversarial/`)
**Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling**

This project focuses on the adversarial perturbation of PSMILES polymer representations against a Glass Transition Temperature (Tg) regression target model. It uses a probabilistic generator to propose modifications to material sequences. 

The project has undergone a rigorous, multi-phase scientific evaluation, which ultimately overturned its initial hypotheses through forensic auditing.

### Experimental Progression & Detailed Results

#### Phase 1: Baseline & Initial Attacks
*   **Target Model:** A small Transformer encoder trained to predict Tg (in Kelvin). 
*   **Data constraints:** Due to dataset deduplication and conflict resolution, only **247 usable samples** were available. The baseline Test MAE on a sealed 6-sample test set was **52.02 K** (Validation MAE: 79.04 K).
*   **The Attacks:** The model was subjected to single-token substitutions, insertions, deletions, and local rearrangements (using RDKit to ensure syntactic validity).
*   **Initial Discovery:** Length-changing perturbations (Insertions/Deletions) caused massive mean prediction drifts (**~30 K**), while length-preserving perturbations (Substitution/Rearrangement) caused minimal drift (**~8 K**). 

#### Phase 2: Defense Ablations & Multi-Seed Replication
*   **Phase 2A (Adversarial Training):** Augmenting the training set with adversarial examples appeared to reduce the Test MAE to 46.20 K and drastically cut the maximum prediction drift for Deletion attacks from 153.9 K down to 81.0 K.
*   **Phase 2B (Controlled Ablation):** An audit discovered that Phase 2A's clean-MAE improvement was a confounder caused by the `TargetScaler` shifting during retraining on augmented data. When the scaler was strictly controlled, **only 1.38 K of the 5.81 K clean improvement survived**.
*   **Phase 2C (Multi-Seed Replication):** Replicated the Phase 2B controls across 5 independent random seeds.
    *   **Finding 1:** There is **no clean-performance benefit** to the adversarial training (+0.81 ± 4.58 K Validation MAE across seeds).
    *   **Finding 2:** Mean/median drift reductions did not reliably replicate. 
    *   **Finding 3:** The only robust defense finding is that adversarial training successfully reduces **worst-case (max) drift** for length-changing attacks (e.g., Insertion max drift dropped from 144.4 K -> 97.7 K consistently across all 5 seeds).

#### Phase 2F: Forensic Architecture Audit (Current Consensus)
A severe adversarial audit was conducted to falsify the Phase 1 and 2 hypotheses, succeeding on all fronts:
*   **Phase 1F Refuted:** Length-preserving controls (shuffling or reversing the tokens) produced drifts of **30.44 K to 32.05 K**, matching or exceeding the insertion/deletion drifts. The model is *globally* unstable to token-level perturbations, not specifically length-sensitive.
*   **The Chemistry Illusion:** A simple 1-parameter length-only linear regression achieved a Validation MAE of **76.77 K**, beating the Transformer's **79.04 K**. The correlation between sequence length and Tg in the dataset is 0.372. The model learned string lengths, not polymer chemistry.
*   **Invalid Target-Preservation:** The assumption that these attacks preserve the true physical Tg is chemically unjustified. Deletions/Insertions change the molecular formula. The only valid Tg-preserving control is SMILES randomization.

#### Phase 3: MCMC Attack Generator & Confound Mitigation
To systematically address the shortcut-learning discovered in Phase 2F, rigorous controls and the final probabilistic attack pipeline were implemented:
*   **Length Baseline Gate & Residualization:** Added a strict length-only linear regression baseline gate. Exploring model training on *length residuals* slightly improved performance (50.62 K vs 52.01 K Test MAE), but the core finding holds: the model remains highly constrained by the small 247-sample dataset size.
*   **SMILES Randomization Control:** Replaced naive shuffling with RDKit's `Chem.MolToSmiles(..., doRandom=True)`. This true label-preserving control caused massive prediction drifts (~32 K mean), confirming the fundamental instability to any token restructuring.
*   **Probabilistic MCMC Generator:** Implemented the Phase 3 Metropolis-Hastings attack generator, successfully proposing and accepting valid sequences with extreme adversarial drift (up to 158 K).
*   **Adversarial Defense:** Training against the MCMC-generated attacks successfully crushed worst-case MCMC drift (155 K -> 114 K) and improved clean performance (48.12 K), yet the inherent fragility to length-preserving randomization remains unsolved.

**Documentation:** Detailed logs of this progression are in [`materials-adversarial/docs/PROJECT_WORKSPACE.md`](materials-adversarial/docs/PROJECT_WORKSPACE.md) and [`materials-adversarial/docs/ARCHITECTURE_FORENSIC.md`](materials-adversarial/docs/ARCHITECTURE_FORENSIC.md).

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
