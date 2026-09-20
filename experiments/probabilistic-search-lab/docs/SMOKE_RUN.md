# Real Smoke Run

Date: 2026-09-20. Status: engineering validation only, non-canonical.

All methods used frozen test source `poly_3106`, the canonical GraphMPNN
checkpoint and scaler, the shared source-relative validator, seed 7, CPU, and a
target-model budget of 3 including the clean prediction.

| Method | Status | Target queries | Proposals | Valid evaluated | Invalid proposals | Best effect (eV) |
|---|---:|---:|---:|---:|---:|---:|
| Random | WORKS | 3 | 34 | 2 | 0 | 0.0978484 |
| Existing Metropolis-style search | WORKS | 2 | 2 | 0 | 2 | missing |
| Evolutionary | WORKS | 3 | 158 | 2 | 0 | 0.0978484 |
| Thompson | WORKS | 3 | 34 | 2 | 0 | 0.1879907 |
| CEM | WORKS | 3 | 34 | 2 | 0 | 0.2008467 |
| SMC | WORKS | 3 | 34 | 2 | 0 | 0.0978484 |
| Validity-SMC | WORKS | 3 | 34 | 2 | 0 | 0.0978484 |

These values prove wiring and accounting only. They do not support method
ranking: `n_sources=1`, only two candidate queries were available, proposal
budgets differ, and several methods encountered the same small operator bank.
The MCMC missing effects are preserved because no shared-validator-eligible
candidate was evaluated; the unused query is not backfilled.

Raw artifacts are `results/smoke_<method>.json`.
