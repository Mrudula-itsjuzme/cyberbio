# Phase 2B Carry/Reset Results

This document summarizes the G_carry memory effect, utilizing rigorous post-run evidence aggregation.

## Data
See `results/final/carry_reset/carry_reset_events.csv`.
Total valid events: 27
Query-matched events: 15

## Analysis
- **CEM**: The memory effect is effectively zero or negative in this configuration. Post-init mean G_carry = 0.013.
- **SMC**: Strong evidence of memory advantage. Post-init mean G_carry = 0.892, with 100% positive fraction across seeds and rounds.
- **Validity-SMC**: Identical to SMC due to a mechanistic implementation detail (see NEGATIVE RESULTS).
