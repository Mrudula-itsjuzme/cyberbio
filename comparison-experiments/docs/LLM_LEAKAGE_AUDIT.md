# LLM Leakage Audit

**Audit Date:** 2026-09-18
**Audited By:** Autonomous Agent

## Abstract
This document outlines the strict isolation protocols maintained between the LLM generation loop and the sealed target model evaluation environment.

## Ground-Truth Isolation
No direct ground-truth label leakage was identified in the audited prompt construction and data flow. The LLM provider (Mock or OpenAICompatible) is strictly isolated to generating candidate proposals in the form of raw PSMILES sequences. The audit does not prove the absence of all possible indirect leakage or prior-model knowledge embedded within the foundation model's weights. 

## Information Visibility
- **Blind Mode:** Sees only the single `source_sequence`.
- **Objective-aware Mode:** Sees `source_sequence` and text describing the optimization objective (e.g. "Maximize absolute prediction drift").
- **Operator-constrained Mode:** Sees `source_sequence` and explicitly allowed classical mutation operators.
- **Iterative Mode:** Sees `source_sequence`, objective text, and black-box adaptive feedback representing the previous 3 `prediction_drift` scalars. This mode explicitly receives scalar score feedback from the sealed environment and thus operates as a black-box adaptive attack. It must not be compared trivially to non-adaptive (blind) modes without distinguishing this advantage.

## RDKit Validity Isolation
The LLM does NOT invoke `CanonicalValidatorAdapter`. All proposals are blindly passed to the comparison framework's identical validation pipeline. Invalid generations cost token/provider budget but fail instantly in the framework without consuming a target-model prediction query.

## Budget Integrity
A cross-attacker parity test guarantees that evaluating the source prediction costs exactly `1` query across all classical (MCMC, Random, Evolutionary) and LLM attackers identically. `budget.record_query()` enforces budget exhaustion universally.
