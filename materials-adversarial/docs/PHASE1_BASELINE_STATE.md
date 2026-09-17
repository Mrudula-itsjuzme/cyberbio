> [!WARNING]
> **Historical Document / Superseded Methodology**
> This file chronicles an earlier development phase of the project (Phases 1-14). Early iterations of the MCMC defense in these logs evaluated closed-loop training using **original-label inheritance** for chemistry-changing edits. As established in the Phase 9 Scientific Audit, this assumption is physically invalid without a DFT oracle.
> The final canonical benchmark (`canonical_benchmark_no_leakage.json` and the final `EXPERIMENTAL_PROTOCOL.md`) explicitly abandons label inheritance in favor of **label-free consistency regularization**, and promotes **Rand-SMILES Augmentation** as the primary, physically sound defense. Please refer to `RESEARCH_CONTRIBUTIONS.md` for the current scientific consensus.

# Phase 1 Baseline State Document

**Project**: Materials Adversarial Framework (`materials-adversarial`)  
**Lineage**: polyVERSE Polymer Bandgap ($E_g$ in eV)  
**Date**: September 12, 2026  
**Status**: Verified Baseline Established

---

## 1. Verified Active Dataset, Split & Configuration

- **Dataset**: polyVERSE polymer dataset (Ramprasad Group)
- **Target Property**: Bandgap ($E_g$)
- **Target Units**: eV
- **Total Usable Records**: 4,209 high-fidelity DFT records
- **Split Strategy**: Deterministic random split (Seed `20260815`), materialized in `data/processed/splits.json`:
  - **Train Split**: 2,946 records (70%)
  - **Validation Split**: 631 records (15%)
  - **Exposed Test Split**: 632 records (15%, `test_sealed: true`)
- **Active Dataset Config**: [`configs/dataset.yaml`](../configs/dataset.yaml)

---

## 2. Tokenizer & Scaler Verification

- **Tokenizer Config**: [`configs/tokenizer.yaml`](../configs/tokenizer.yaml) (Regex-based polymer sequence tokenizer)
- **Vocabulary File**: `data/processed/vocab.json` (36 tokens, including `[*]` attachment points)
- **Target Scaler**: `results/models/transformer_regressor/scaler.json`
  - **Fitted Mean**: `4.474831296673456`
  - **Fitted Std**: `1.4563050416759802`
  - **SHA256 Hash**: `5ef15ad16aee156fbc9b6b7872c1534262e3622022190f17c23fc7e3dab12814`

---

## 3. Checkpoint Provenance & Parameter Count

- **Checkpoint File**: `results/models/transformer_regressor/model.pt`
  - **SHA256 Hash**: `0b1bb3f62ca72a897fd84c8e7f3129f8a7a9d75c1179c10debb63e46113731e4`
- **Metrics Record**: `results/models/transformer_regressor/metrics.json`
- **Total Parameters**: **85,761**
- **Trainable Parameters**: **85,761**
- **Scientific Status**: Valid, high-performing baseline checkpoint ($R^2 = 0.8019$). Kept as the canonical baseline without retraining.

---

## 4. Current Ordinary Transformer Architecture

Defined in `src/materials_adv/models/transformer.py` (`TransformerRegressorModel`):

- **Token Embedding**: $36 \times 64$
- **Positional Embedding**: $256 \times 64$ (Learned absolute positional encodings)
- **Encoder Layers**: 2 Transformer Encoder layers (`nn.TransformerEncoderLayer`)
- **Model Dimension ($d_{\text{model}}$)**: 64
- **Attention Heads**: 4
- **Feedforward Dimension**: 128
- **Activation**: ReLU
- **Encoder Dropout**: 0.1
- **Pooling Strategy**: Masked Mean Pooling over non-PAD tokens
- **Regression Head**: Linear layer ($64 \rightarrow 1$) with inverse target scaling

---

## 5. Recorded Baseline Performance Metrics

Evaluated on clean data using the frozen target scaler:

| Evaluation Split | Sample Count ($N$) | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Coefficient of Determination ($R^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **Validation Split** | 631 | **0.4862 eV** | 0.6674 eV | 0.7846 |
| **Exposed Test Split** | 632 | **0.4619 eV** | 0.6235 eV | 0.8019 |

---

## 6. Exposed Test Split Integrity Confirmation

- **Status**: The 632-sample test split has been repeatedly exposed during Phase 2 development and is **no longer statistically pristine**.
- **Protocol Enforced**: 
  - All model selection, architecture tuning, and hyperparameter decisions must be guided **exclusively by the validation split ($N=631$)**.
  - Test-set metrics are recorded strictly for reference and must not be used for model tuning.

---

## 7. Phase 1 Verification Summary

- [x] Active dataset/split/config verified
- [x] Tokenizer/scaler verified & hashes recorded
- [x] Checkpoint provenance verified & SHA256 recorded
- [x] Parameter count confirmed (85,761)
- [x] Validation (0.4862 eV MAE) and exposed test (0.4619 eV MAE) metrics recorded
- [x] Test split non-pristine status acknowledged
- [x] Ready to proceed to Phase 2 (Two-Head Specialized Model design)
