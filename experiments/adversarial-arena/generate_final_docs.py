import os
import pandas as pd
import numpy as np
import json

# 1. UPDATE CARRY_RESET_RESULTS.md
with open("docs/CARRY_RESET_RESULTS.md", "w") as f:
    f.write("""# Phase 2B Carry/Reset Results

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
""")

# 2. UPDATE CARRY_RESET_RQ_LEDGER.md
with open("docs/CARRY_RESET_RQ_LEDGER.md", "w") as f:
    f.write("""# RQ Ledger (Phase 2B)

- **RQ-CEM: Does CEM carry memory improve search?**
  **Verdict**: INCONCLUSIVE

- **RQ-SMC: Does SMC particle carry memory improve search?**
  **Verdict**: INCONCLUSIVE for primary matched-resource causal claim.
  **Secondary**: STRONG POSITIVE SECONDARY SIGNAL.

- **RQ-VSMC: Does Validity-SMC carry memory improve search?**
  **Verdict**: INCONCLUSIVE for primary matched-resource claim.
  **Secondary**: Same positive all-event signal as SMC.

- **RQ-VPOST: Does the validity posterior provide additional benefit beyond SMC?**
  **Verdict**: NOT_SUPPORTED_IN_PILOT
  (Note: Additional validity-posterior benefit beyond SMC was not observed. Exact causal mechanism is not fully proven from persisted states without explicit resampling logs.)
""")

# 3. UPDATE CARRY_RESET_NEGATIVE_RESULTS.md
with open("docs/CARRY_RESET_NEGATIVE_RESULTS.md", "w") as f:
    f.write("""# Negative and Null Results

- CEM seed 20260920 R1/R2 had negative raw G_carry.
- Other CEM seeds were positive.
- Therefore CEM is heterogeneous, not globally degraded.
- 12/27 events were query-mismatched.
- Only 6 matched post-init events remain.
- SMC and Validity-SMC produced identical realized particle trajectories and attack metrics under the tested configuration.
- Validity-SMC maintained an extra validity posterior, but it produced no observed incremental effect.

## Mechanism Check (SMC vs Validity-SMC)
- **FACT**: Particle states were identical across SMC and Validity-SMC.
- **FACT**: Drifts were identical.
- **FACT**: Validity-SMC has an extra validity posterior.
- **FACT**: The posterior participates in scoring.
- **INFERENCE**: The posterior did not alter realized selection dynamics under this budget.
""")

# 4. UPDATE CARRY_RESET_RUN_VALIDATION.md
with open("docs/CARRY_RESET_RUN_VALIDATION.md", "w") as f:
    f.write("""# Run Validation

- **Round0 control**: PASS
- **Carry integrity**: PASS
- **Reset integrity**: PASS
- **RNG integrity**: PASS
- **Reference SHA integrity**: PASS
- **D0 identity**: PASS

## Limitations
- Query-matched events: 15/27
- Matched post-init events: 6/18 post-init
- Primary replication limitation: n=1 per method × post-init round
""")
