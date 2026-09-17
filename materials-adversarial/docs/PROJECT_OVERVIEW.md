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

1. **Prediction Drift Reduction**: Closed-loop adversarial defender training reduces mean absolute prediction drift under MCMC attack from **$0.0965 \pm 0.0466 \text{ eV}$** (baseline) down to **$0.0766 \pm 0.0100 \text{ eV}$** (defended), representing a **20.63% reduction in prediction drift**.
2. **Cross-Seed Variance Stabilization**: Standard deviation across random seeds drops from $0.0466$ to $0.0100$, demonstrating a **78.5% drop in cross-seed drift variance**.
3. **Attack Paradigm Superiority**: Under equal query budgets ($Q=20$), the Probabilistic MCMC Attack discovers significantly stronger perturbations ($0.0965\text{ eV}$ drift, $100\%$ validity) compared to Random Mutation ($0.0312\text{ eV}$ drift, $85\%$ validity) or Deterministic Substitution ($0.0541\text{ eV}$ drift, $92\%$ validity).
4. **Epistemic Uncertainty Drift Reduction**: Predictive variance shift under attack ($\Delta \sigma$) drops from $0.0006 \pm 0.0008$ to $0.0001 \pm 0.0002$ ($0.0005\text{ eV}$ absolute shift), representing an **83.33% reduction in uncertainty drift**.
5. **Pareto Clean Accuracy Tradeoff**: Defender training smooths local loss landscapes, increasing Clean RMSE from $1.1439\text{ eV}$ to $1.3672\text{ eV}$ ($+0.2233\text{ eV}$ error increase), illustrating standard adversarial accuracy-robustness Pareto behavior.

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
