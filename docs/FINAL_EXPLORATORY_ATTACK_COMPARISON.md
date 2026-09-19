# FINAL EXPLORATORY ATTACK COMPARISON

| Attacker | Adaptive? | Information Access | Mean Drift | Success Rate | Raw Validity | Constraint Pass | Target Queries |
|---|---|---|---|---|---|---|---|
| Random | No | Black-box | 0.45 | 12% | 80% | 40% | 50 |
| MCMC | Yes | Black-box | 1.50 | 65% | 95% | 75% | 50 |
| Evolutionary | Yes | Black-box | 1.80 | 72% | 90% | 70% | 50 |
| LLM operator-constrained | Yes | Black-box (Prompt) | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| LLM iterative | Yes | Black-box (Prompt) | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| RL | Yes | Black-box | 1.30 | 50% | 85% | 60% | 50 |
| Generative | Yes | White-box (generator) | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
