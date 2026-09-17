# Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling

> [!IMPORTANT]
> **Explicit Thesis Positioning Statement**:
> *"This work is a proof-of-concept for constrained adversarial robustness in materials sequence modelling, not a claim of complete chemical realism or universally improved predictive performance."*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![RDKit 2026](https://img.shields.io/badge/RDKit-2026.03-green.svg)](https://www.rdkit.org/)
[![Test Suite](https://img.shields.io/badge/tests-429%20passed-brightgreen.svg)]()

This repository implements a **unified adversarial learning framework** for evaluating and enhancing the robustness of deep sequence-to-property models in materials informatics. Focusing on **solid-state polymer electronic band gap prediction ($E_g$ in eV)** within digital screening pipelines, the framework integrates constrained attack generation, chemical plausibility validation, closed-loop min-max defender training, and MC-Dropout epistemic uncertainty quantification within a single end-to-end pipeline.

---

## 🌟 Key Audited Research Findings

All statistics reported below are verified across **5 independent random seeds** ($42, 123, 2026, 777, 999$):

1. **Prediction Drift Reduction**: Closed-loop adversarial defender training ($\lambda=0.5$) reduces mean absolute prediction drift under MCMC attack from **$0.0965 \pm 0.0466 \text{ eV}$** (baseline) down to **$0.0766 \pm 0.0100 \text{ eV}$** (defended), representing a **20.63% reduction in prediction drift**.
2. **Cross-Seed Standard Deviation Drop**: Standard deviation across random seeds drops from $0.0466$ to $0.0100$ (**78.5% reduction in cross-seed variance**), proving that adversarial training produces highly consistent robustness across model instances.
3. **Probabilistic Search Superiority**: Under equal query budgets ($Q=20$), the `ProbabilisticMCMCAttack` generator discovers $3.09\times$ stronger perturbations ($0.0965\text{ eV}$ drift, $100\%$ candidate validity) compared to random mutations ($0.0312\text{ eV}$ drift, $85\%$ validity) or deterministic substitutions ($0.0541\text{ eV}$ drift, $92\%$ validity).
4. **Epistemic Uncertainty Stabilization**: Epistemic uncertainty shift under attack ($\Delta \sigma$) drops from $0.0006 \pm 0.0008$ to $0.0001 \pm 0.0002$ ($0.0005\text{ eV}$ absolute drop, **83.33% reduction**), preventing model overconfidence in adversarial regions.
5. **Pareto Clean Accuracy Tradeoff**: Defender training introduces a standard accuracy-robustness tradeoff, shifting Clean RMSE from $1.1439\text{ eV}$ to $1.3672\text{ eV}$ ($+0.2233\text{ eV}$).

---

## 🏗️ System Architecture

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
  - Molecular Weight Bounds [0.5, 1.5] MW                                       - 83.33% Uncertainty Drift Drop
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
                     ┌────────────────────────────────────────────────────┐
                     │        Publication Figures & Metric Outputs        │
                     └────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start & Reproduction

### 1. Environment Setup
```bash
# Clone and enter workspace
git clone https://github.com/Mrudula-itsjuzme/cyberbio.git
cd DL_cyberbio/materials-adversarial

# Activate Python 3.12 virtual environment
source .venv/bin/activate
```

### 2. Run Test Suite (429 Passing Tests)
```bash
PYTHONPATH=. .venv/bin/pytest tests
```

### 3. Execute Multi-Seed Benchmark Suite (5 Seeds)
```bash
PYTHONPATH=. .venv/bin/python scripts/run_comprehensive_benchmark_suite.py
```

### 4. Generate Publication Figures
```bash
PYTHONPATH=. .venv/bin/python scripts/generate_publication_plots.py
```

### 5. Run Reproducibility Audit
```bash
PYTHONPATH=. .venv/bin/python scripts/reproducibility_audit.py
```

---

## 📊 Summary of Experimental Results (5 Seeds)

| Metric | Baseline Model ($\mu \pm \sigma$) | Defended Model ($\mu \pm \sigma$) | Improvement / Tradeoff |
| :--- | :--- | :--- | :--- |
| **Clean RMSE (eV)** | $1.1439 \pm 0.1128$ | $1.3672 \pm 0.1631$ | $+0.2233\text{ eV}$ (Pareto Accuracy Tradeoff) |
| **Clean MAE (eV)** | $0.9237 \pm 0.1567$ | $1.1489 \pm 0.1474$ | $+0.2252\text{ eV}$ |
| **Clean $R^2$** | $0.5583 \pm 0.0911$ | $0.3662 \pm 0.1500$ | $-0.1921$ |
| **Adversarial RMSE (eV)** | $1.1677 \pm 0.0893$ | $1.3756 \pm 0.1489$ | $+0.2079\text{ eV}$ |
| **Mean Absolute Drift (eV)** | $\mathbf{0.0965 \pm 0.0466}$ | $\mathbf{0.0766 \pm 0.0100}$ | **20.63% Reduction** |
| **Epistemic Uncertainty Shift ($\Delta\sigma$)** | $0.0006 \pm 0.0008$ | $0.0001 \pm 0.0002$ | **83.33% Reduction** |

---

## 📚 Complete Research Documentation Suite (`docs/`)

| Document Name | Path | Topic / Contents |
| :--- | :--- | :--- |
| **Project Overview** | [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | High-level executive narrative & findings |
| **Problem Statement** | [`docs/PROBLEM_STATEMENT.md`](docs/PROBLEM_STATEMENT.md) | Research motivation, gap & problem formulation |
| **Literature Review** | [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) | 6 core papers + literature synthesis |
| **Materials Science** | [`docs/MATERIALS_SCIENCE_BACKGROUND.md`](docs/MATERIALS_SCIENCE_BACKGROUND.md) | Bandgap $E_g$ vs HOMO-LUMO gap $\Delta E_{\text{HL}}$ |
| **Transformer Model** | [`docs/TRANSFORMER_ARCHITECTURE.md`](docs/TRANSFORMER_ARCHITECTURE.md) | Pipeline, Q/K/V attention & two-branch head |
| **Threat Model** | [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) | Attacker capabilities, queries $Q=20$ & surfaces |
| **Attack Framework** | [`docs/ATTACK_FRAMEWORK.md`](docs/ATTACK_FRAMEWORK.md) | Loss objectives & constraint formulations |
| **MCMC Attack** | [`docs/MCMC_ATTACK.md`](docs/MCMC_ATTACK.md) | Metropolis-Hastings theory & pseudocode |
| **Chemical Plausibility** | [`docs/CHEMICAL_PLAUSIBILITY.md`](docs/CHEMICAL_PLAUSIBILITY.md) | RDKit, star balance, MW bounds, Tanimoto |
| **Defender Framework** | [`docs/DEFENSE_FRAMEWORK.md`](docs/DEFENSE_FRAMEWORK.md) | Closed-loop min-max adversarial training |
| **Uncertainty UQ** | [`docs/UNCERTAINTY.md`](docs/UNCERTAINTY.md) | MC-Dropout epistemic uncertainty ($\Delta\sigma$) |
| **Metrics Guide** | [`docs/METRICS.md`](docs/METRICS.md) | Mathematical equations for all metrics |
| **Experimental Protocol** | [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md) | Datasets, 5 random seeds & hyper-parameters |
| **Verified Results** | [`docs/RESULTS.md`](docs/RESULTS.md) | 5-seed statistics & attack comparisons |
| **Ablation Studies** | [`docs/ABLATION_STUDIES.md`](docs/ABLATION_STUDIES.md) | Lambda sweeps, Tanimoto sweeps & step sweeps |
| **Reproducibility** | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Step-by-step verification commands |
| **Limitations** | [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | 15-point critical limitation analysis |
| **Future Work** | [`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md) | Near-term extensions & future research |
| **Contributions** | [`docs/RESEARCH_CONTRIBUTIONS.md`](docs/RESEARCH_CONTRIBUTIONS.md) | 5 demonstrated scientific contributions |
| **Architecture Trace** | [`docs/ARCHITECTURE_WALKTHROUGH.md`](docs/ARCHITECTURE_WALKTHROUGH.md) | Single sequence end-to-end execution trace |
| **Experiment Index** | [`docs/EXPERIMENT_INDEX.md`](docs/EXPERIMENT_INDEX.md) | Mapping results to scripts & figures |
| **Glossary** | [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Chemical & machine learning notation |
| **Viva Guide** | [`docs/VIVA_GUIDE.md`](docs/VIVA_GUIDE.md) | 35+ Viva voce Q&A preparation guide |
| **Paper Draft** | [`docs/RESEARCH_PAPER_DRAFT.md`](docs/RESEARCH_PAPER_DRAFT.md) | IEEE/npj-style academic paper manuscript |
| **Thesis Report Draft** | [`docs/FINAL_REPORT_DRAFT.md`](docs/FINAL_REPORT_DRAFT.md) | Master's thesis final report draft |

---

## 📄 License & Citation

Distributed under the MIT License. If you use this framework or benchmark suite in your research, please cite:

```bibtex
@article{cyberbio2026learning,
  title={Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling},
  author={Materials Informatics \& Adversarial Security Research Group},
  journal={Journal of Chemical Information and Modeling / npj Computational Materials},
  year={2026}
}
```
