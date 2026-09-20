# Why These Methods

| Method | Why use it / alternative | Assumption and targeted failure | Cost | Invalid if / decision |
|---|---|---|---|---|
| Random | Calibrates every adaptive method; simpler than MCMC | Valid edits sampled uniformly are informative | Low | Budget or validator differs; keep baseline. |
| MCMC | Retains locally useful states unlike independent random search | Local drift landscape has navigable neighborhoods | Sequential queries, mixing | Asymmetric proposals ignored or chain incomparable; keep exploratory. |
| Evolutionary | Population can escape one local trajectory | Recombination/selection preserve useful edits | Many proposals, population bookkeeping | Small budgets cannot form generations; keep exploratory. |
| Attribution | Tests whether model salience improves targeting | Occlusion score identifies causally useful positions | One query per token plus attack | Attribution cost omitted or controls absent; keep with controls. |
| Generator | Tests learned syntax prior rather than blind edits | Training distribution supports valid local proposals | Training plus autoregressive inference | Leakage, invalid control, or zero accepted candidates; do not promote. |
| LLM | Could inject broad structural priors | Prompted model can propose constraint-compatible strings | API tokens, latency, nondeterminism | LLM self-validates or mock data is treated as evidence; blocked. |
| RL | Could learn adaptive action sequences | Reward and state transfer across sources | High sample/compute cost | Reward unvalidated, no held-out policy test; blocked, noncanonical. |
| GAN | Could learn a proposal distribution | Stable adversarial learning over discrete outputs | High and unstable | Existing model mislabeled or validity poor; rejected. |

RandomSMILES alone cannot answer chemistry-changing search questions. A GNN replacement changes the target model rather than the attacker. Direct gradient attacks break the black-box comparison contract. DFT is valuable as a downstream physical oracle but too costly and unavailable for the core query loop. The validator, not an LLM, is authoritative because validity must be deterministic, inspectable, and identical across proposal families.
