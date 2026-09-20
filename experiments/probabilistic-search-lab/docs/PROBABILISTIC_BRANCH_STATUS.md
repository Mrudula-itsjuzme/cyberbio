# Probabilistic Branch Status

Engineering status after the first real smoke run. Performance columns remain
unknown until matched-budget multi-source/multi-seed experiments.

| Method | Implemented? | Runs? | Valid? | Better than Random? | Better than MCMC? | Query Efficient? | Compute Acceptable? | Novelty Status | Final Decision |
|---|---|---|---|---|---|---|---|---|---|
| Random | yes | yes | smoke only | unknown | unknown | unknown | yes at smoke scale | established | REFERENCE_BASELINE |
| Existing Metropolis-style | adapter | yes | zero eligible in smoke | unknown | unknown | unknown | yes at smoke scale | established | KEEP_EXPERIMENTAL |
| Evolutionary | adapter | yes | smoke only | unknown | unknown | unknown | yes at smoke scale | established | REFERENCE_BASELINE |
| Thompson | yes | yes | smoke only | unknown | unknown | unknown | yes at smoke scale | established method + application | KEEP_EXPERIMENTAL |
| CEM | basic | yes | smoke only | unknown | unknown | unknown | yes at smoke scale | established method + application | KEEP_EXPERIMENTAL |
| SMC | basic | yes | smoke only | unknown | unknown | unknown | yes at smoke scale | established method + application | KEEP_EXPERIMENTAL |
| Validity-SMC | basic | yes | smoke only | unknown | unknown | unknown | yes at smoke scale | established combination | KEEP_EXPERIMENTAL |
| BO / Validity-BO | no | no | unknown | unknown | unknown | unknown | unknown | established | BLOCKED |
| Uncertainty search | no | no | unknown | unknown | unknown | unknown | unknown | established combination | BLOCKED |
| GFlowNet | no | no | unknown | unknown | unknown | unknown | unknown | established | BLOCKED |
