# materials-adversarial

**Learning to Attack and Defend: A Unified Adversarial Framework for Robust
Materials Sequence Modelling**

Phase 1 (attack side) scaffold. Adversarial perturbation of PSMILES polymer
representations against a Tg-regression target model.

> **Status: the dataset is not present. Nothing has been trained. No results exist.**
> See [`docs/PROJECT_WORKSPACE.md`](docs/PROJECT_WORKSPACE.md) — the living source
> of truth for status, decisions, blockers and next actions.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[chem,dev]"
./scripts/run_tests.sh -q
```

`torch` is deliberately **not** installed: model hyperparameters cannot be chosen
until the dataset is audited, and this machine has no GPU. When needed:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The `run_tests.sh` wrapper exists because a system-wide ROS 2 install exports a
global `PYTHONPATH` whose pytest plugins crash collection. See Problem #6 in the
workspace doc.

## Current blocker

The OpenPoly dataset must be supplied before any dataset-dependent work:

- **Paper:** Wang et al., *"OpenPoly: A Polymer Database Empowering Benchmarking
  and Multi-property Predictions"*, Chinese Journal of Polymer Science (2025),
  DOI `10.1007/s10118-025-3402-y`
- **Data:** https://github.com/WangGroupFDU/Openpoly_benchmark →
  `data/final_polymer_properties_fromliterature.csv`

Place it in `data/raw/`, then:

```bash
.venv/bin/python scripts/audit_dataset.py --path data/raw --output results/audit_report.json
```

The audit **discovers** schema rather than assuming it, proposes candidate
columns with evidence for human confirmation, and never auto-writes configs.

## What works today

| Component | Notes |
|---|---|
| PSMILES tokenizer | Chemical tokens, exact round-trip, 176 tests |
| Attacks | Deletion, insertion, reordering complete; substitution mechanics complete |
| Validity pipeline | Representation (RDKit) + plausibility heuristics, kept separate |
| Attack records | JSONL, signed drift, Phase 2+ fields reserved |
| Metrics | MAE/RMSE/R², configurable attack-success criterion |
| Audit tooling | Schema discovery, Tg unit detection, conflict analysis |

```bash
# Exercise the full pipeline with no model (ConstantPredictor test double):
PYTHONPATH="" .venv/bin/python -c "
import numpy as np
from materials_adv.attacks.deletion import DeletionAttack
from materials_adv.attacks.generator import AttackGenerator, ConstantPredictor
gen = AttackGenerator([DeletionAttack(np.random.default_rng(0))],
                      predictor=ConstantPredictor(350.0), seed=0)
for r in gen.run([('p1', '[*]CC(=O)O[*]')], n_variants=3):
    print(r.adversarial_psmiles, r.validity_status, r.prediction_drift)
"
```

## Design commitments

- **Token-level attacks, never character-level.** A character edit splits `Cl`
  and corrupts `[C@@H]` — a string bug that would masquerade as chemistry.
- **Strict UNCHECKED validity.** Syntactic well-formedness never implies chemical
  validity. Without a successful RDKit parse, candidates are `UNCHECKED`, not
  valid. No result may claim scientific plausibility from syntax alone.
- **Signed drift.** `abs()` is applied at reporting time; direction of Tg shift
  is a real finding.
- **Protected positions.** Edits that guarantee invalidity (unmatched parens,
  broken ring closures, removed attachment points) are excluded by default, so
  attacks measure model robustness rather than the validity filter.
- **PENDING stubs raise.** Blocked code fails loudly instead of returning a
  plausible wrong answer.
- **No fabricated data.** Ever.

## Scope

**In (Phase 1):** dataset audit, preprocessing, baseline Tg model, attack
generation, validity filtering, drift evaluation (Experiments 1–5).

**Out:** adversarial training, defense, MCMC, probabilistic attacks, uncertainty
estimation, external-dataset validation (Experiments 6–11).
