# Cross-Entropy Search

Status: basic implementation and real smoke run complete.

The current implementation learns a categorical distribution over the two
canonical edit families. Each iteration samples proposals, applies the shared
validator, scores only eligible candidates, selects the top `rho` fraction, and
fits the family distribution to the elites. Smoothing and an exploration floor
limit immediate collapse. Family probabilities and entropy are logged.

This is the first controlled implementation, not the full planned factorization
over family, position, and replacement token. That extension should happen only
after candidate-position identities are made stable across canonical operators.
CEM is established; no novelty claim is made.
