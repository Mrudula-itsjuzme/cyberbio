# Arena Phase 2 — Frozen Baseline Reference

## Baseline Commit
Frozen baseline (single-seed): `c77e72f` (branch: `experiments/adversarial-arena`)  
Multi-seed milestone: `a121117` (branch: `experiments/adversarial-arena-probabilistic`)

## Baseline Run IDs
- `arena_transformer_seed20260920`
- `arena_transformer_seed20260921`
- `arena_transformer_seed20260922`

## Baseline Attacker Set
- **Training arms**: RandomSMILES, MCMC (Thompson leave-one-out)
- **Held-out (unseen transfer)**: Evolutionary

## Baseline Defender Configuration
- Architecture: TransPolymer
- Optimizer: Adam, LR=0.001, weight_decay=0.0001
- Epochs per round: 1, Batch size: 16
- Hard-example top-k: 4

## Baseline Query Budget
- 25 target-model queries per source per attacker

## Baseline Seeds
- 20260920, 20260921, 20260922 (3 rounds each)

## Baseline Held-Out Evolutionary Setup
- Evolutionary excluded from Thompson training selection
- Evaluated against every D_t and D_{t+1} for transfer calculation

## Baseline Clean Performance (mean ± std across 3 seeds)
| Round | Clean RMSE |
|-------|-----------|
| 0 | 0.4457 ± 0.0982 |
| 1 | 0.5308 ± 0.1252 |
| 2 | 0.5920 ± 0.0640 |

## Baseline Transfer Metrics (Held-out Evolutionary, T_Evo = R(Evo,D_t) - R(Evo,D_{t+1}))
| Round | Transfer Gain (mean ± std) |
|-------|--------------------------|
| 0 | −0.015 ± 0.051 |
| 1 | −0.031 ± 0.091 |

## Integrity Rule
Phase 2 runs MUST NOT alter any artifact under `results/arena/arena_transformer_seed*`.  
All Phase 2 results are written to `results/arena/phase2_{attacker}_{seed}/`.
