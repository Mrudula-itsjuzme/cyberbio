> [!NOTE]
> **Status**: HISTORICAL

# Phase 7: Search Method Audit

## The Anomaly
In Phase 7, we observed a surprising phenomenon during adaptive adversarial search: **Random Search significantly outperformed Greedy Search and Metropolis-Hastings MCMC**. Initially, this was incorrectly attributed to a highly non-convex loss landscape where greedy approaches were getting trapped in poor local optima.

## The Methodological Flaw
A methodological audit revealed that this behavior was an artifact of the `attack_budget=1` (max_changes=1) constraint combined with how stateful search algorithms track their "current" state.

1. **Stateful Traps**: Greedy Search and Metropolis MCMC update their internal "current state" whenever they accept a candidate. Under `max_changes=1`, any further modifications to that new state would result in a candidate that is 2-hops away from the original sequence. 
2. **Rejection Cascade**: Because the attack budget strictly enforces a maximum of 1 edit relative to the *original* sequence, all subsequent proposals generated from the 1-hop state are rejected as invalid (budget exceeded).
3. **Algorithm Halting**: 
    - **Greedy Search** evaluates all 1-hop neighbors of the new state. Since all of them are rejected (they are 2-hops from the original), it finds no valid improvements and terminates prematurely.
    - **Metropolis MCMC** continues to propose candidates from the 1-hop state, but they are consistently rejected for exceeding the budget, rendering the rest of the search budget useless.
4. **Random Search's Advantage**: Random Search samples candidates by modifying the original sequence at each step. By keeping the original sequence cached (a 0-hop revert), it never gets "trapped" in a 1-hop state. Every proposal is exactly 1-hop away from the original, allowing it to efficiently explore the neighborhood and fully utilize the search budget.

## Oracle Verification
To confirm this, we enumerated the true 1-hop neighborhood for a set of validation samples (the "Oracle") and compared the search algorithms' performance against it at a budget of 50 queries:

*   **Random Search**: Recovered ~85% of the optimal Oracle drift and found the exact Oracle best candidate ~60% of the time.
*   **Greedy Search**: Recovered only ~40% of the optimal Oracle drift and found the exact Oracle best candidate <10% of the time, due to halting after the first accepted move.
*   **Metropolis MCMC**: Recovered ~55% of the optimal Oracle drift and found the exact Oracle best candidate ~12% of the time.

The multi-seed stability check confirmed that Random Search's high performance is consistent across different random seeds (mean drift standard deviation is minimal across 5 runs).

## Conclusion and Recommendations
The superiority of Random Search in Phase 7 is a direct consequence of the 1-hop budget constraint acting as an algorithmic trap for stateful searches, not a fundamental property of the chemical loss landscape.

**Phase 8 Recommendation**:
To properly evaluate adversarial robustness and enable multi-step gradient-free optimization (which more closely resembles real-world adaptive attacks), we must **introduce a multi-edit adaptive budget (e.g., `attack_budget=3` or `5`)**. This will allow Greedy and MCMC algorithms to traverse the landscape legitimately without hitting an immediate mathematical dead end.
