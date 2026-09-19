# FAILURE ANALYSIS

## Categories
- **Invalid syntax**: Mutations that produce unparseable strings.
- **Constraint failure**: Mutations generating structurally invalid molecules (e.g. failing RDKit checks).
- **Duplicate collapse**: MCMC converging on the same representation.
- **No improvement**: Search getting stuck in local optima.
- **Budget exhaustion**: Failing to find a successful adversarial example within 50 queries.
- **Provider failure**: LLM API key unavailable.
- **Representation instability**: Small edits leading to large drift.
- **Defense overfitting**: Defenses transferring poorly across distinct attack distributions.
