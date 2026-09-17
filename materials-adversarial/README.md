# Learning to Attack and Defend: A Unified Adversarial Framework for Robust Materials Sequence Modelling

> [!IMPORTANT]
> **Explicit Thesis Positioning Statement**:
> *"This work is a proof-of-concept for constrained adversarial robustness in materials sequence modelling, not a claim of complete chemical realism or universally improved predictive performance."*

> **Central Thesis**: A polymer Transformer exhibits substantial prediction dependence on the serialization of chemically identical PSMILES. Representation-preserving randomized-SMILES augmentation materially reduces this non-invariance and improves scaffold-split generalization. Chemistry-changing MCMC perturbations provide a complementary stress test of local model sensitivity; consistency training can reduce that sensitivity, but physical accuracy on the edited structures remains unverified without an independent property oracle.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.13](https://img.shields.io/badge/PyTorch-2.13-ee4c2c.svg)](https://pytorch.org/)
[![RDKit 2026](https://img.shields.io/badge/RDKit-2026.03-green.svg)](https://www.rdkit.org/)
[![Test Suite](https://img.shields.io/badge/tests-429%20passed-brightgreen.svg)]()

This repository implements a **unified adversarial learning framework** for evaluating and enhancing the robustness of deep sequence-to-property models in materials informatics. Focusing on **solid-state polymer electronic band gap prediction ($E_g$ in eV)** within digital screening pipelines, the framework integrates constrained attack generation, chemical plausibility validation, label-free consistency regularization for chemistry-changing stress tests, and MC-Dropout epistemic uncertainty quantification within a single end-to-end pipeline.

---

## 🌟 Key Audited Research Findings

All statistics reported below are verified across **5 independent random seeds** ($42, 123, 2026, 777, 999$) via `canonical_benchmark_no_leakage.json`:

1. **Primary Vulnerability (Representation Drift):** Chemically identical PSMILES serializations produce substantial prediction variation in the Transformer. Randomizing canonical SMILES yields a mean prediction drift of $0.5991 \text{ eV}$ (RandomSplit) and $0.8968 \text{ eV}$ (ScaffoldSplit), exposing severe serialization dependence.
2. **Primary Defense (Scientifically Clean):** Representation-preserving multi-SMILES augmentation drastically reduces this vulnerability while improving structural out-of-distribution (OOD) generalization. On ScaffoldSplit, it reduces representation drift by 54.5% ($0.8968 \rightarrow 0.4079 \text{ eV}$) and improves Clean RMSE ($0.6998 \rightarrow 0.6630 \text{ eV}$). Because the underlying molecule is physically identical, this augmentation safely inherits the original ground-truth $E_g$ label.
3. **Secondary Stress Test (MCMC Attack):** Constrained MCMC explores chemistry-changing neighborhoods and exposes local model sensitivity, discovering perturbations that shift predictions by $\sim 0.22 \text{ eV}$ while strictly passing chemical validity filters.
4. **Secondary Defense Experiment (Robustness Regularization):** Applying a label-free consistency regularizer ($L_{\text{cons}}=\mathcal{L}(f_\theta(x_{\text{MCMC}}), \operatorname{stopgrad}(f_\theta(x)))$) during MCMC training successfully reduces sensitivity to those edits (MCMC drift drops to $0.18 \text{ eV}$). However, because edited structures lack independently computed $E_g$ values, this demonstrates *robustness regularization* rather than evidence of improved physical accuracy.
5. **Future Definitive Experiment:** Resolving the true band gap of chemistry-changing adversarial candidates requires a dedicated physical oracle (e.g., DFT validation) to provide ground-truth $E_g(x_{\text{MCMC}})$ labels.

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
• Probabilistic MCMC Candidate Generator                                      • Label-Free Consistency Regularization
  - Bioisosteric Functional Group Swaps                                         L = L_clean + λ L_cons
  - Metropolis Acceptance Sampling                                            • Epistemic Uncertainty (MC-Dropout)
• Chemical Plausibility Validator                                               - Predictive Mean & Variance
  - RDKit Parsing + Valence Compliance                                        • Multi-Seed Verification (5 Seeds)
  - Attachment Star '*' Balance                                                 - Baseline vs Defended Drift
  - Tanimoto Similarity (S_Tanimoto >= 0.5)                                     - 47.4% MCMC Drift Reduction
  - Molecular Weight Bounds [0.5, 1.5] MW                                       - 54.5% Rep. Drift Drop (Scaffold)
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

## 📊 Canonical Benchmark Results (Leakage-Free, 5 Seeds)

### 1. Random Split
| Experiment | Clean RMSE (eV) | Rand-SMILES drift (eV) | MCMC drift (eV) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 0.6007 | 0.5991 | 0.2085 |
| **MCMC-Defended** | 0.6101 | 0.5916 | 0.1624 |
| **Rand-SMILES Aug** | 0.5962 | 0.3257 | 0.1609 |
| **Combined Defense** | 0.6391 | 0.3136 | 0.1097 |

### 2. Scaffold Split
| Experiment | Clean RMSE (eV) | Rand-SMILES drift (eV) | MCMC drift (eV) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 0.6998 | 0.8968 | 0.2218 |
| **MCMC-Defended** | 0.7184 | 0.6710 | 0.1807 |
| **Rand-SMILES Aug** | 0.6630 | 0.4079 | 0.2047 |
| **Combined Defense** | 0.6803 | 0.3776 | 0.1816 |

See `results/experimental_summary.md` and `results/canonical_benchmark_no_leakage.json` for MAE, R², and 95% Confidence Intervals.

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
| **Defender Framework** | [`docs/DEFENSE_FRAMEWORK.md`](docs/DEFENSE_FRAMEWORK.md) | Representation-preserving augmentation and label-free MCMC consistency regularization |
| **Uncertainty UQ** | [`docs/UNCERTAINTY.md`](docs/UNCERTAINTY.md) | MC-Dropout epistemic uncertainty ($\Delta\sigma$) |
| **Metrics Guide** | [`docs/METRICS.md`](docs/METRICS.md) | Mathematical equations for all metrics |
| **Experimental Protocol** | [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md) | Datasets, 5 random seeds & hyper-parameters |
| **Data Splits** | [`docs/scaffold_split.md`](docs/scaffold_split.md) | Scaffold split construction and zero-overlap guarantees |
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
