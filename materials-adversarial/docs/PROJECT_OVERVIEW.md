# Project Overview: Unified Adversarial Learning Framework for Materials Sequence Modelling

> [!IMPORTANT]
> **Explicit Thesis Positioning Statement**:
> *"This work is a proof-of-concept for constrained adversarial robustness in materials sequence modelling, not a claim of complete chemical realism or universally improved predictive performance."*

## 1. Executive Summary

This project develops a **unified adversarial learning framework** for evaluating and enhancing the robustness of sequence-based deep learning models in materials informatics. Focusing on **solid-state polymer bandgap prediction ($E_g$ in eV)** within digital screening pipelines, the framework integrates:
1. **Scientifically Constrained Adversarial Attack Generation**: Formulates mathematical attack objectives that maximize prediction drift subject to chemical validity and structural similarity constraints ($S_{\text{Tanimoto}} \ge 0.5$).
2. **Probabilistic MCMC Search Generator**: Implements a Metropolis-Hastings stochastic search algorithm (`ProbabilisticMCMCAttack`) utilizing bioisosteric functional group proposal operators to efficiently discover strong adversarial sequence candidates.
3. **Chemical Plausibility Validator**: Enforces RDKit syntax parsing, valence correctness, polymer attachment star `*` balance, and molecular weight bounds ($\pm 50\%$).
4. **Closed-Loop Min-Max Defender Training**: Trains the target surrogate model on dynamically generated adversarial candidates, updating parameters via min-max loss optimization ($\lambda=0.5$).
5. **Epistemic Uncertainty Quantification**: Employs Monte Carlo Dropout (MC-Dropout) to measure predictive variance and uncertainty shift under attack ($\Delta \sigma$).

```
                                  UNIFIED ADVERSARIAL FRAMEWORK
                                  
                     ┌────────────────────────────────────────────────────┐
                     │            Polymer PSMILES Input (x)               │
                     └─────────────────────────┬──────────────────────────┘
                                               │
                                               ▼
                     ┌────────────────────────────────────────────────────┐
                     │   TransPolymer Sequence Transformer (f_theta)      │
                     │          Predicts Band Gap E_g (eV)                │
                     └─────────────────────────┬──────────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────┴──────────────────────────────────────────────┐
│                                                                                             │
▼                                                                                             ▼
[ATTACKER MODULE]                                                             [DEFENDER MODULE]
• Probabilistic MCMC Candidate Generator                                      • Closed-Loop Min-Max Training
  - Bioisosteric Functional Group Swaps                                         L = (1-λ) L_clean + λ L_adv
  - Metropolis Acceptance Sampling                                            • Epistemic Uncertainty (MC-Dropout)
• Chemical Plausibility Validator                                               - Predictive Mean & Variance
  - RDKit Parsing + Valence Compliance                                        • Multi-Seed Verification (5 Seeds)
  - Attachment Star '*' Balance                                                 - Baseline vs Defended Drift
  - Tanimoto Similarity (S_Tanimoto >= 0.5)                                     - 20.63% Mean Drift Reduction
  - Molecular Weight Bounds [0.5, 1.5] MW                                       - Pareto Clean RMSE Tradeoff
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
                     ┌────────────────────────────────────────────────────┐
                     │        Publication Figures & Metric Outputs        │
                     └─────────────────────────┬──────────────────────────┘
```

---

## 2. Key Audited Experimental Findings

Evaluated across **5 independent random seeds** ($42, 123, 2026, 777, 999$):

1. **Primary Vulnerability (Representation Drift):** Chemically identical PSMILES serializations produce substantial prediction variation in the Transformer. Randomizing canonical SMILES yields a mean prediction drift of $0.5991 \text{ eV}$ (RandomSplit) and $0.8968 \text{ eV}$ (ScaffoldSplit), exposing severe serialization dependence.
2. **Primary Defense (Scientifically Clean):** Representation-preserving multi-SMILES augmentation drastically reduces this vulnerability while improving structural out-of-distribution (OOD) generalization. On ScaffoldSplit, it reduces representation drift by 54.5% ($0.8968 \rightarrow 0.4079 \text{ eV}$) and improves Clean RMSE ($0.6998 \rightarrow 0.6630 \text{ eV}$). Because the underlying molecule is physically identical, this augmentation safely inherits the original ground-truth $E_g$ label.
3. **Secondary Stress Test (MCMC Attack):** Constrained MCMC explores chemistry-changing neighborhoods and exposes local model sensitivity, discovering perturbations that shift predictions by $\sim 0.22 \text{ eV}$ while strictly passing chemical validity filters.
4. **Secondary Defense Experiment (Robustness Regularization):** Applying a label-free consistency regularizer ($L_{\text{cons}}=\mathcal{L}(f_\theta(x_{\text{MCMC}}), \operatorname{stopgrad}(f_\theta(x)))$) during MCMC training successfully reduces sensitivity to those edits (MCMC drift drops to $0.18 \text{ eV}$). However, because edited structures lack independently computed $E_g$ values, this demonstrates *robustness regularization* rather than evidence of improved physical accuracy.
5. **Future Definitive Experiment:** Resolving the true band gap of chemistry-changing adversarial candidates requires a dedicated physical oracle (e.g., DFT validation) to provide ground-truth $E_g(x_{\text{MCMC}})$ labels.

---

## 3. High-Level Repository Architecture

```
materials-adversarial/
├── src/materials_adv/
│   ├── models/                    <- Specialized Transformer & GraphMPNN architectures
│   ├── attacks/                   <- MCMC, Random, Substitution, Insertion, Deletion attacks
│   ├── domain/chemistry/          <- Plausibility validator, polymer attachment, Tanimoto metric
│   ├── evaluation/                <- Robustness metrics, MAE, RMSE, R^2, uncertainty
│   ├── framework/                 <- Predictor adapters & interface definitions
│   └── experiments/               <- Closed-loop training & evaluation pipeline adapters
├── scripts/
│   ├── run_comprehensive_benchmark_suite.py  <- Executes 5-seed benchmark & ablations
│   ├── generate_publication_plots.py         <- Generates publication PNG figures in outputs/
│   └── reproducibility_audit.py              <- Verifies environment & artifact integrity
├── tests/                         <- 430 unit & integration tests (429 passed, 1 skipped)
├── results/                       <- JSON & CSV benchmark outputs
└── outputs/                       <- Publication figures (PNG format)
```

---

## 4. Document Suite Index

The complete documentation suite comprises 26 detailed technical files located in `docs/`:
- **Domain & Architecture**: `MATERIALS_SCIENCE_BACKGROUND.md`, `TRANSFORMER_ARCHITECTURE.md`, `PROBLEM_STATEMENT.md`, `LITERATURE_REVIEW.md`.
- **Methodology & Theory**: `THREAT_MODEL.md`, `ATTACK_FRAMEWORK.md`, `MCMC_ATTACK.md`, `CHEMICAL_PLAUSIBILITY.md`, `DEFENSE_FRAMEWORK.md`, `UNCERTAINTY.md`, `METRICS.md`.
- **Experiments & Audits**: `EXPERIMENTAL_PROTOCOL.md`, `RESULTS.md`, `ABLATION_STUDIES.md`, `REPRODUCIBILITY.md`, `LIMITATIONS.md`, `FUTURE_WORK.md`, `RESEARCH_CONTRIBUTIONS.md`.
- **Manuscripts & Guides**: `RESEARCH_PAPER_DRAFT.md`, `FINAL_REPORT_DRAFT.md`, `VIVA_GUIDE.md`, `ARCHITECTURE_WALKTHROUGH.md`, `EXPERIMENT_INDEX.md`, `GLOSSARY.md`.
