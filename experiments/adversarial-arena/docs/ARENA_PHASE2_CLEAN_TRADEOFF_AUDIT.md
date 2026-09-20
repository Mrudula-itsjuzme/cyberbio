# Clean-Robust Tradeoff Audit

Expected up to 6 classifiable rounds per method (R1 and R2 for 3 seeds). R0 is always NA due to missing D0 clean RMSE (evaluations not persisted before first training cycle).

## SMC Missing Rows Investigation

For SMC, only 4 classifiable updates exist. Investigation of raw artifacts confirms:
- SMC Seed 20260920 completed R0, R1, R2. (yields 2 classifiable updates: R1, R2)
- SMC Seed 20260921 completed R0, R1 and then terminated/exited cleanly as `RUN_COMPLETE`. (yields 1 classifiable update: R1)
- SMC Seed 20260922 completed R0, R1 and then terminated/exited cleanly as `RUN_COMPLETE`. (yields 1 classifiable update: R1)

Total classifiable SMC rounds = 2 + 1 + 1 = 4.

There is no parsing bug. The missing 2 classifications are because those rounds do not exist in the empirical trajectory artifacts.

## D0 Clean Metric Provenance
Initial clean metric D0 was evaluated in memory but not persisted. 
Therefore: Round 0 clean-performance delta is NOT_AVAILABLE. 
Clean/robust tradeoff begins only once consecutive persisted defender checkpoints exist.
