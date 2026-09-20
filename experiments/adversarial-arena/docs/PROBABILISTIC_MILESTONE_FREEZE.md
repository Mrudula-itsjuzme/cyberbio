# Probabilistic Arena Integration Milestone

**Status**: FROZEN
**Date**: 2026-09-20

## Summary
The internal search probabilistic attackers (CEM, SMC, and Validity-SMC) have been successfully integrated into the Arena architecture using wrapper adapters (`arena.adapters`). 
The algorithms themselves were not rewritten, ensuring the exact preservation of their original implementations from the `probabilistic-search-lab`. 

## Key Integrity Constraints Satisfied
1. **Central Query Accounting**: All internal predictions and validity checks increment the outer Arena query count securely. Internal searches are forced to evaluate constraints via the `_ArenaValidator` wrapper.
2. **Deterministic Validator Supremacy**: Empirical Validity estimators only guide sampling; they never overrule the underlying deterministic RDKit parser.
3. **Internal Metric Export**: Sub-agent search metrics (CEM family distributions, SMC Effective Sample Size, and Validity Beta posteriors) are safely bubbled up into `AttackerState` without polluting the external arena abstraction. 
4. **App Visualization**: The React frontend (`main.jsx`) has been patched to view curricula-specific runs and render internal metrics tracking across rounds.

## Experiments
The 3-round, 3-seed runs across `baseline`, `probabilistic`, and `mixed` curricula are currently running via `scripts/run_probabilistic_experiments.sh`. Once complete, the data will be used to verify the differences in convergence between curriculum types.

## Null Results Kept Out
Contextual Thompson has been kept out of this iteration, as specified in the pilot review.
