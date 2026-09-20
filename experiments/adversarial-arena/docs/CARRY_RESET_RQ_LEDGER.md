# RQ Ledger (Phase 2B)

- **RQ-CEM: Does CEM carry memory improve search?**
  **Verdict: NOT_SUPPORTED_IN_PILOT**. CEM shows negative/zero effect, failing to meaningfully exploit carry state in this low-budget configuration.

- **RQ-SMC: Does SMC particle carry memory improve search?**
  **Verdict: SUPPORTED_IN_PILOT**. SMC shows consistent positive G_carry across all seeds/rounds.

- **RQ-VSMC: Does Validity-SMC carry memory improve search?**
  **Verdict: SUPPORTED_IN_PILOT**. Produces identical results to SMC due to lack of resampling, but positive G_carry is still observed over Reset.

- **RQ-VPOST: Does the validity posterior provide additional benefit beyond SMC?**
  **Verdict: NOT_SUPPORTED_IN_PILOT**. Mechanistic null result: validity posterior does not alter trajectory under budget=25.
