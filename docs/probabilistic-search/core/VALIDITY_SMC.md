# Validity-Calibrated SMC

Status: empirical validity-weighted implementation and real smoke run complete.

The feasibility proxy is a Beta-Bernoulli model per canonical edit family. It is
updated exclusively from proposal validity labels; target-model effects never
enter its training data. Posterior mean validity is included in particle log
weights. The full posterior is written to each result.

This smoke implementation tests the accounting and feedback path. It is not yet
a contextual validity model: sequence length, local context, and descriptors are
planned ablations and require held-out calibration checks.
