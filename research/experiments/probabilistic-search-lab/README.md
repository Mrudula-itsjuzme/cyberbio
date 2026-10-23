# Probabilistic Search Lab

Experimental, non-canonical benchmark branch. Canonical research assets are
read-only dependencies. See `docs/INTERFACE_AUDIT.md` before adding adapters.

The shared entry point is `run_probabilistic_benchmark.py`. Random, the existing
Metropolis-style search, evolutionary search, Thompson, CEM, SMC, and
validity-SMC use the same canonical assets and accounting gateway. Unsupported
methods return an explicit machine-readable status and never manufacture metrics.
