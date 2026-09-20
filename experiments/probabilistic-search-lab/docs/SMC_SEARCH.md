# Sequential Monte Carlo Search

Status: basic implementation and real smoke run complete.

Particles start at the source, propagate through canonical edit families, pass
the shared source-relative validation gate, and receive log weights from model
effect minus operator edit cost. Weights are normalized with a max-shift,
effective sample size is computed as `1 / sum(w_i^2)`, and multinomial
resampling occurs below the configured ESS fraction. ESS, entropy, resampling,
unique-particle count, and target queries are logged at every step.

The current rejuvenation kernel is the next canonical local edit. Tempering and
alternative resampling kernels remain ablations, not implicit defaults.
