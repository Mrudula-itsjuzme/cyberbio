> [!WARNING]
> **Historical Document / Superseded Methodology**
> This file chronicles an earlier development phase of the project (Phases 1-14). Early iterations of the MCMC defense in these logs evaluated closed-loop training using **original-label inheritance** for chemistry-changing edits. As established in the Phase 9 Scientific Audit, this assumption is physically invalid without a DFT oracle.
> The final canonical benchmark (`canonical_benchmark_no_leakage.json` and the final `EXPERIMENTAL_PROTOCOL.md`) explicitly abandons label inheritance in favor of **label-free consistency regularization**, and promotes **Rand-SMILES Augmentation** as the primary, physically sound defense. Please refer to `RESEARCH_CONTRIBUTIONS.md` for the current scientific consensus.

> [!NOTE]
> **Status**: HISTORICAL

# Phase 6 — Attack Family Expansion

## Research boundary

Phase 6 compares qualitatively different black-box search policies under the
same realized model-query budget. It does not add an LLM, GAN, gradient access,
or Transformer-internal access. Every candidate score passes only through
`PredictorProtocol.predict`.

The benchmark deliberately separates:

- **Proposal operators:** the existing substitution, insertion, deletion, and
  local rearrangement implementations.
- **Search policies:** random walk/search, greedy ascent, beam search,
  Metropolis-style stochastic search, and evolutionary population search.

These proposal operators can change polymer chemistry and the target property.
They are adversarial stress tests, not label-preserving examples. RDKit parsing
and the project plausibility checks control candidate eligibility; they do not
prove synthesizability or preservation of bandgap.

## Query and perturbation controls

The canonical run uses 25 seeded validation molecules, 12 candidate model
queries per molecule and strategy, and a maximum token edit distance of three
from the original representation. The original prediction is one additional
shared-cost query and is reported separately. Invalid, implausible, duplicate,
and over-budget proposals do not consume candidate model queries, but remain in
the raw proposal trace.

All 125 sample-strategy runs exhausted exactly 12 candidate queries. Therefore
the canonical comparison has equal nominal and realized budgets. A preliminary
30-query run did not meet this condition and is preserved as
`results/attack_efficiency_nominal30_unequal_realized/`; it is not used to rank
strategies.

The Metropolis implementation is a Metropolis-style optimization heuristic. It
is not called formal Metropolis-Hastings because the composite, state-dependent
proposal distribution has no computed forward/reverse proposal correction.

## Canonical observed results

| Search policy | Mean best drift (eV) | Drift/query | Valid / proposed | Mean proposals | Mean edit distance |
|---|---:|---:|---:|---:|---:|
| Random | 1.0186 | 0.0849 | 0.1297 | 144.68 | 2.76 |
| Greedy | 1.3377 | 0.1115 | 0.1422 | 156.12 | 2.76 |
| Beam | 1.2134 | 0.1011 | 0.4002 | 32.28 | 2.28 |
| Metropolis-style | 1.2560 | 0.1047 | 0.1337 | 144.52 | 2.76 |
| Evolutionary | 1.2310 | 0.1026 | 0.4024 | 33.40 | 2.28 |

Greedy accepted 0.2365 of eligible proposals as improvements. The
Metropolis-style policy accepted 0.4766. Acceptance rate is not defined for
random, beam, or evolutionary selection because forcing those mechanisms into
the same denominator would be misleading.

Paired mean best-drift differences versus random search were:

| Search policy | Delta (eV) | Paired bootstrap 95% interval |
|---|---:|---:|
| Greedy | +0.3191 | [+0.1068, +0.5658] |
| Beam | +0.1948 | [+0.0124, +0.3965] |
| Metropolis-style | +0.2374 | [+0.0442, +0.4412] |
| Evolutionary | +0.2124 | [-0.0317, +0.4712] |

The intervals use 2,000 paired resamples over only 25 molecules. They quantify
sampling uncertainty for this run; they are not multiple-comparison-corrected
hypothesis tests and must not be reported as universal significance.

## Interpretation

### Observed

- Greedy search achieved the largest mean best drift and best drift per query
  at the fixed 12-query budget.
- Beam and evolutionary search required far fewer raw proposals to obtain 12
  eligible, distinct candidates and ended at smaller mean edit distances.
- All policies found large prediction changes among RDKit-valid, plausibility-
  eligible strings, but those changes cannot be assigned the original labels.
- Search path changes candidate validity: the shared operator set does not imply
  identical proposal distributions after policies move to different states.

### Supported

- Search policy matters beyond the choice of atomic edit operator for this
  checkpoint, query budget, perturbation bound, and validation sample.
- Greedy and Metropolis-style search are more query-efficient than the random
  baseline in this bounded experiment.
- Population/beam state management improves proposal-level eligibility, although
  its best-drift advantage is partly entangled with a smaller realized edit size.

### Speculative or unverified

- Any strategy is superior at budgets above 12, on a sealed test split, on other
  model seeds, or against defended models.
- RDKit-valid and heuristic-plausible candidates are experimentally realizable.
- Larger prediction drift corresponds to a larger real bandgap error.
- Evolutionary search is worse than greedy search; 25 examples are insufficient
  for that general ranking.

## Artifacts

The canonical immutable run is `results/attack_efficiency/`:

- `proposal_trace.jsonl`: every proposal, rejection, query, drift, and decision.
- `per_example_results.csv`: paired terminal metrics and seeds.
- `query_efficiency_curves.csv`: best-so-far drift after every model query.
- `attack_efficiency.csv`: aggregate efficiency matrix.
- `summary.json`: headline results and comparison guardrails.
- `reproducibility.json`: checkpoint hash, runtime, seed, and git state.
- `config_snapshot.json`: exact dataset, model, attack, and Phase 6 settings.

Reproduce into a new directory from `materials-adversarial/`:

```bash
.venv/bin/python scripts/run_attack_efficiency.py \
  --output-dir results/attack_efficiency_<new-run-id>
```

Existing output directories are never overwritten.
