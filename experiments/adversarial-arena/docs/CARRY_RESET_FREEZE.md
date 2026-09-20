# Phase 2B Carry/Reset Experiment — Freeze Document

## Provenance

| Item | Value |
|------|-------|
| Branch | `experiments/adversarial-arena-carry-reset` |
| Scientific execution commit | `043c0de5e4870dea25af25c36e2946bf21e16cc5` |
| Config-lock commit | `431613795a3843d9fe0e32fdce09e9cb01d101be` |
| Final analysis/freeze commit | `b6a8efcc7687a41fd55a7b4ad974498c655ec5cd` |

## Experiment Scope

| Metric | Value |
|--------|-------|
| Paired experiments | 9 |
| Treatment arms | 18 |
| Integrity-valid events | 27 |
| Query-matched events | 15 |
| Round-0 control events (matched) | 9 |
| Matched post-init events | 6 |

## Primary Evidence

**Matched-resource post-init causal contrasts: n=1 per method x round.**
No primary finding should be described as replicated across seeds.

## Findings

### Primary Findings (Matched-Resource)

| Method | Verdict |
|--------|---------|
| CEM | INCONCLUSIVE |
| SMC | INCONCLUSIVE |
| Validity-SMC | INCONCLUSIVE |

### Secondary Findings (All Integrity-Valid Events)

| Method | R1 mean G_carry | R2 mean G_carry | Post-init: n=3 mean (std) |
|--------|----------------|----------------|---------------------------|
| CEM | 0.0059 | 0.0205 | 0.0132 (0.0501) |
| SMC | 0.8742 | 0.9095 | 0.8918 (0.2492) |
| Validity-SMC | 0.8742 | 0.9095 | 0.8918 (0.2492) |

**SMC secondary signal: STRONG POSITIVE across all 3 seeds at both R1 and R2.**
**Validity-SMC secondary signal: identical to SMC.**

Post-init per-seed (SMC / Validity-SMC):
- 20260920 = 0.8151
- 20260921 = 0.6899
- 20260922 = 1.1704
- Overall: n=3, mean=0.8918, sample std=0.2492, median=0.8151, min=0.6899, max=1.1704

### Validity Posterior

**NOT_SUPPORTED_IN_PILOT**

Additional validity-posterior benefit beyond SMC was not observed. Validity-SMC produced
identical realized particle trajectories and attack metrics under the tested configuration.

**Mechanistic facts** (directly observed from artifacts):
- Particle states identical across SMC and Validity-SMC
- Drift values identical
- Validity-SMC maintains an extra validity posterior
- Posterior participates in resampling scoring

**Inference** (not directly proven from persisted artifacts):
The validity posterior did not alter realized selection dynamics under this budget.
The exact operational reason is not fully isolated from the persisted artifacts.

## Limitations

- n=3 seeds total
- 12/27 events were query-mismatched (carry overshoot relative to reset)
- Matched post-init analysis is n=1 per method x round — not replicated
- Chemistry-changing drift is a model sensitivity measure, not verified physical property preservation
- No defender co-evolution occurs (shared reference trajectory D_ref is fixed)
- No causal claim from per-query normalization (descriptive secondary only)
- SMC and Validity-SMC produced identical realized particle trajectories and attack metrics under this tested configuration

## Integrity Checks

| Check | Result |
|-------|--------|
| Round-0 control (G_carry_mean == 0) | PASS |
| Carry state continuity | PASS |
| Reset state independence | PASS |
| RNG schedule match | PASS |
| Reference SHA integrity (9 x 3 events) | PASS |
| D0 identity (9 comparisons) | PASS |

## FREEZE STATUS

`FREEZE_APPROVED`

Post-run analysis documents and corrected evidence tables match the V2 aggregation
script (`aggregate_audit_carry_reset_v2.py`) output. Raw scientific artifacts in
`results/arena/` are immutable and unchanged.
