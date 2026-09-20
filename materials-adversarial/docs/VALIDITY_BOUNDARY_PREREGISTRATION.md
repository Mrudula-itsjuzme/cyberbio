# VALIDITY-BOUNDARY SEARCH PREREGISTRATION

## Motivation
Existing observations suggest:
- Validity-SMC can exploit learned validity statistics.
- Many naive proposals die before target evaluation.
- Context derived only from the clean source fails to predict useful attack families.
- Useful information may therefore be LOCAL and SEARCH-CONDITIONAL rather than static/source-global.

## Hypotheses
- **H1**: Prediction drift increases as accepted candidates approach particular validity/plausibility boundaries.
- **H2**: The maximum-drift candidate distribution differs from the general valid-candidate distribution.
- **H3**: An online boundary-aware sampler improves drift per target-model query relative to ordinary SMC.
- **H4**: Boundary targeting may increase invalid-proposal cost even when target-query efficiency improves.
- **H5**: Any benefit may be source-specific and should not be assumed universal.

## Boundary Definition
We operationalize "boundary proximity" using a local feasibility probe without spending target-model queries.
For a candidate `x`, we sample `k=10` cheap neighboring proposals using the base edit families.
We calculate local validity probability: `p_valid_local(x) = valid_neighbors / total_neighbors`.
Boundary uncertainty is defined as:
`u_valid(x) = 4 * p_valid_local(x) * (1 - p_valid_local(x))`
This value peaks at 1.0 when exactly half the neighbors are valid, representing the edge of the feasibility manifold.

## Methods & Accounting
- **Methods**: Random, MCMC, CEM, SMC, Validity-SMC, Boundary-SMC (with ablations for validity vs. boundary preference).
- **Target Queries**: Matched strictly at 10, 25, 50 queries. Target-model evaluations are explicitly separated from validator queries and local neighbor probes.
- **Seeds**: Paired sources and seeds (e.g., 42, 123, 456).

Status: WRITTEN PRIOR TO BOUNDARY-SMC FINAL EVALUATION.
