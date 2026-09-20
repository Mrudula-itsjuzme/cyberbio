# Attack Family Status

| Attack | Bio-Cyber status | Materials status | Executed? | Valid evidence? | Blocker | Canonical candidate? | Reason |
|---|---|---|---|---|---|---|---|
| Random | EXECUTED | EXECUTED | Yes | Yes, exploratory | None | Baseline only | Essential comparator, not a new method. |
| MCMC | EXECUTED | EXECUTED | Yes | Yes, exploratory | Single seed/N=30 | No | No consistent supported advantage. |
| Evolutionary | EXECUTED | EXECUTED | Yes | Yes, exploratory | Cost/single seed | No | Descriptive gains need replication. |
| Attribution-high | EXECUTED | EXECUTED | Yes | Yes | Attribution cost | No | Not consistently efficient. |
| Attribution-low | EXECUTED | EXECUTED | Yes | Yes, negative control | Validity starvation | No | Control, not candidate. |
| Attribution-random | EXECUTED | EXECUTED | Yes | Yes, control | Attribution cost | No | Control, not candidate. |
| Generative-trained | EXECUTED | EXECUTED | Yes | Yes | Materials zero accepted | No | No cross-domain advantage. |
| Generative-untrained | EXECUTED | INCOMPLETE_BLOCKED_TIMEOUT | Bio only | Materials no | True run hung | No | Required ablation incomplete. |
| LLM | BLOCKED_EXTERNAL_CREDENTIALS | BLOCKED_EXTERNAL_CREDENTIALS | Mock tests only | No efficacy evidence | Provider credentials | No | Protocol, not result. |
| RL | BLOCKED_COMPUTE | BLOCKED_COMPUTE | No | No | Implementation/compute | No | Protocol only. |
| GAN | REJECTED_METHOD | REJECTED_METHOD | No | Not applicable | Method fit | No | Discrete constrained strings. |
