# Phase 2B Carry/Reset Results

This document summarizes the G_carry memory effect using three evidence tiers.

## Evidence Tiers
- **PRIMARY**: Matched-resource post-init evidence.
- **SECONDARY**: All integrity-valid events.
- **SECONDARY DESCRIPTIVE**: Drift per query.

## Data Accounting
- 27 integrity-valid events
- 15 query-matched events
- 9 of those are R0 controls
- 6 query-matched post-init events remain
- Therefore n=1 per method × R1/R2

## Primary vs Secondary Findings

**CEM**
- **Primary**: INCONCLUSIVE
- **Secondary**: Small heterogeneous effect.

**SMC**
- **Primary**: INCONCLUSIVE due to n=1 matched event per post-init round.
- **Secondary**: Strong positive all-event signal across all 3 seeds at both R1 and R2.
  - R1 mean G_carry_mean = 0.8741733649961839
  - R2 mean G_carry_mean = 0.9094665095520716

**SMC Post-init per-seed summary**
- 20260920 = 0.8151
- 20260921 = 0.6899
- 20260922 = 1.1704
- **Across seeds**: n=3, mean = 0.8918, sample std = 0.2492, median = 0.8151, min = 0.6899, max = 1.1704

**Validity-SMC**
- **Primary**: INCONCLUSIVE due to n=1 matched event per post-init round.
- **Secondary**: Same numerical pattern as SMC.
