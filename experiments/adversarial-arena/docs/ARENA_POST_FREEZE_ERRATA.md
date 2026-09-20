# Arena Post-Freeze Errata

This document logs textual inconsistencies found and corrected in the frozen Arena documentation (Phase 0 of Probabilistic Integration).

1. **Adaptive Selection Gain Formulation**
   - *Error:* Some frontend tooltips mistakenly described Adaptive Selection Gain as a ratio `R(A_{t+1}, D_t) / R(A_t, D_t)`.
   - *Correction:* Replaced with the correct subtractive formulation: `R(A_{t+1}, D_t) - R(A_t, D_t)`. This ensures consistency with the mathematical definition that positive values indicate absolute robustness drift increase.

2. **Rejection Language**
   - *Error:* `ARENA_TRANSFORMER_PILOT.md` stated that zero-shot transfer was "firmly rejected".
   - *Correction:* Replaced with "not supported in this pilot" because a small-sample multi-seed (n=3) does not justify a strong statistical rejection claim.

3. **Equilibrium Terminology**
   - *Error:* `ARENA_VIVA.md` and `ARENA_MULTI_SEED_ANALYSIS.md` referenced "Nash equilibrium".
   - *Correction:* Replaced with "long-term convergence" and "stationarity" since formal game-theoretic equilibrium properties were not defined for this system.

4. **Training Source Count Discrepancy**
   - *Error:* `ARENA_MULTI_SEED_ANALYSIS.md` and `ARENA_PILOT_CONFIG_LOCK.md` incorrectly listed "10 training sources".
   - *Correction:* Audited `run_pilot.py`, which explicitly hardcodes `df[df.split=="train"].head(100)`. Corrected the documentation to state 100 training sources and 10 validation sources.
