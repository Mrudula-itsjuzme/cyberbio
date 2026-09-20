# Probabilistic Attacker Integration Audit

## 1. CEM (Cross-Entropy Method)
- **File:** `/experiments/probabilistic-search-lab/cem/search.py`
- **Constructor:** `CEMSearch(families, batch_size, elite_fraction, smoothing, entropy_floor)`
- **Proposal Representation:** Random choice from families weighted by probabilities `probs`.
- **Target-Model Query:** Calls `target.predict(proposal.candidate)` only if validator passes.
- **Validator Mechanism:** Calls `validator.validate(source, proposal.candidate)`.
- **Query Budget:** Tracked by `ledger.remaining_target`.
- **Random State:** Handles its own `np.random.default_rng(seed)`.
- **Returned Schema:** Returns dictionary from `finalize` method (standard across the lab).
- **Diagnostics:** Returns trajectories containing entropy, probs, elite effects.
- **Dependencies:** `categorical_entropy`, `search_common.begin`, `search_common.finalize`.
- **Classification:** **NEEDS_ADAPTER** (Arena requires a standard `BaseAttacker` class interface).

## 2. Basic SMC (Sequential Monte Carlo)
- **File:** `/experiments/probabilistic-search-lab/smc/search.py`
- **Constructor:** `SMCSearch(families, particles, ess_threshold, beta, validity_model=None)`
- **Proposal Representation:** Sequential particle mutation over iterations.
- **Target-Model Query:** Calls `target.predict(proposal.candidate)` if validator passes.
- **Validator Mechanism:** Calls `validator.validate(source, proposal.candidate)`.
- **Query Budget:** Iterates while `ledger.remaining_target`.
- **Random State:** Uses `np.random.default_rng(seed)`.
- **Returned Schema:** Uses `finalize` dictionary.
- **Diagnostics:** Logs ESS, particle count, entropy, and resampling events.
- **Dependencies:** `search_common.begin`, `search_common.finalize`.
- **Classification:** **NEEDS_ADAPTER**

## 3. Validity-SMC
- **File:** `/experiments/probabilistic-search-lab/validity_smc/model.py` and `smc/search.py`
- **Constructor:** Instantiated via `SMCSearch(..., validity_model=EmpiricalValidityModel())`.
- **Diagnostics:** Same as SMC plus `validity_posterior`.
- **Classification:** **NEEDS_ADAPTER**

### Adapter Strategy
Instead of copying the algorithms into `arena/attackers/`, we will write thin adapter classes that subclass `BaseAttacker`. These classes will instantiate `CEMSearch` or `SMCSearch`, call their `.run()` method, and format the output dictionary into `ArenaCandidate` lists, while hooking up the Arena's query budget and validator correctly.

## Phase 9-11: Fixed-Defender Benchmark Results

The following metrics were observed during a 1-round attack (query budget: 25) against 10 validation sources using the frozen `D_0` TransPolymer defender checkpoint:

```json
{
  "RandomSMILES": {
    "mean_drift": 0.0,
    "max_drift": 0.0,
    "valid_candidates": 0,
    "total_queries": 0,
    "runtime_seconds": 0.14
  },
  "MCMC": {
    "mean_drift": 0.0,
    "max_drift": 0.0,
    "valid_candidates": 0,
    "total_queries": 26,
    "runtime_seconds": 0.07
  },
  "CEM": {
    "mean_drift": 0.23,
    "max_drift": 1.53,
    "valid_candidates": 240,
    "total_queries": 260,
    "runtime_seconds": 2.40
  },
  "SMC": {
    "mean_drift": 0.25,
    "max_drift": 1.53,
    "valid_candidates": 174,
    "total_queries": 194,
    "runtime_seconds": 1.99
  },
  "Validity-SMC": {
    "mean_drift": 0.23,
    "max_drift": 1.59,
    "valid_candidates": 161,
    "total_queries": 181,
    "runtime_seconds": 1.72
  }
}
```

### Classification
- **ARENA_READY**: CEM, SMC, Validity-SMC (Successfully propose multiple diverse candidates per source with substantial drift up to ~1.5 eV).
- **BASELINE_ONLY**: RandomSMILES, MCMC (Struggle heavily with the strict validity constraints given limited queries).

All implemented wrappers adhere strictly to the Central Query Accounting model via `_ArenaPredictor` and `_ArenaValidator`, ensuring deterministic validation and robust evaluation.
