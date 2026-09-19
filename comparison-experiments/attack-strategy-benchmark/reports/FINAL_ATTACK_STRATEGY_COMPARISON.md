# Attack Strategy Comparison — Phase 4.1 (Corrected)

*Branch: `experiments/attack-strategy-comparison`*
*Canonical CyberBio results on `main` are FROZEN and unmodified.*

---

## 1. Scope and Design

This optional comparison study evaluates six attack conditions across two domains
(Bio-Cyber and Materials) under a unified, fair evaluation framework.

**Domains:**
- **Bio-Cyber** — binary DNA sequence classifier (CNN_Distance), task: label-flip
- **Materials** — polymer Tg regression (TransformerRegressor), task: maximise |ΔTg|

**Attack conditions (identical in both domains):**
| Condition | Strategy |
|---|---|
| Random | Uniform random mutation, accept if valid |
| MCMC | Metropolis-Hastings on drift objective |
| Evolutionary | Population selection on drift fitness |
| Attribution-High | Occlusion attribution → target highest-sensitivity position |
| Attribution-Low | Occlusion attribution → target lowest-sensitivity position |
| Attribution-Random | Occlusion attribution → target random position (control) |

**Source pools:** N = 30 deterministically sampled sources per domain (seed = 42,
frozen manifests).

---

## 2. Budget Definitions

Three distinct counters are tracked per example:

| Counter | Definition |
|---|---|
| `proposal_attempts` | Every mutation call, including failed ones; hidden retries are surfaced |
| `valid_candidates` | Proposals that pass RDKit SMILES parse + canonicalisation + tokenisation |
| `attack_model_queries` | Model evaluations on valid candidates only |

**Primary budget: `attack_model_queries` ∈ {5, 20, 50}**

**Proposal cap: `MAX_PROPOSALS = 10 × query_budget`**
This cap is applied uniformly to all six conditions.

**Attribution queries** are counted separately:
```
attribution_queries  = len(tokenize(source))   # actual token count, NOT 256
total_model_queries  = attribution_queries + attack_model_queries
```

Two efficiency views are reported:
- **Search-only**: comparison at equal `attack_model_queries`
- **End-to-end**: comparison at equal `total_model_queries`

---

## 3. Bio-Cyber Results

Bio-Cyber uses ACGT-alphabet sequences of length 150. All mutations are syntactically
valid (no external validity oracle), so `valid_candidates == proposal_attempts` for
non-attribution attacks.

Attribution overhead: `attribution_queries = 150` per source.

| Condition | Budget 50 mean drift | Budget 50 success rate |
|---|---|---|
| (see `bio_cyber/aggregated_results.csv`) | | |

*Full paired statistics in `results/bio_cyber/statistical_results.json`.*

---

## 4. Materials Results

### 4.1 Model

- Checkpoint: `materials-adversarial/results/models/transformer_regressor/model.pt`
- Architecture: TransformerRegressor, `vocab_size=37` (36 tokens + padding index 0)
- Tokeniser: historical `char2idx = {c: i+1 for i, c in enumerate(vocab)}`, 1-indexed
- Status: `VERIFIED` (see `results/materials/model_compatibility.json`)

### 4.2 SMILES parse validity

Candidates are checked with `Chem.MolFromSmiles` → canonicalisation via
`Chem.MolToSmiles(mol, canonical=True)` → tokenisation against the historical vocab.

**This is RDKit SMILES parse / molecular graph validity, not physical plausibility.**
Physical plausibility would require a separate DFT or force-field oracle not present here.

RDKit proposal validity rates (across conditions, budget = 50):

*(see `results/materials/proposal_validity_summary.csv` for exact per-condition rates)*

Key observation: token-level mutations drawn from the 36-token vocabulary alphabet
produce invalid SMILES at rates that depend on the structural neighbourhood of the
source string. Validity rates range from ~0.23 to ~0.56 across conditions and budgets.

### 4.3 Observed drift (search-only budget)

From `results/materials/aggregated_results.csv` (budget = 50):

| Condition | Mean |ΔTg| | Proposal attempts | RDKit validity rate |
|---|---|---|---|
| Random | 0.365 | 106.1 | 0.491 |
| MCMC | 0.525 | 94.9 | 0.552 |
| Evolutionary | 0.566 | 97.4 | 0.532 |
| Attribution-High | 0.297 | 182.6 | 0.534 |
| Attribution-Low | 0.101 | 351.9 | 0.236 |
| Attribution-Random | 0.308 | 208.2 | 0.294 |

### 4.4 Attribution overhead

Attribution queries = actual tokenised source length, not the padding length of 256.

| Statistic | Value |
|---|---|
| Mean attribution queries | 34.4 |
| Median | 34.0 |
| Min | 8 |
| Max | 90 |

*(Values computed from the 30-source pool; exact numbers in `statistical_results.json`.)*

### 4.5 Paired statistics (Wilcoxon signed-rank, N = 30, seed 42, 1 000 bootstrap resamples)

**No McNemar test is applied.** `success = NaN` for all Materials rows because no
canonical regression threshold exists in the Materials project.

Exact p-values, effect sizes, and 95 % bootstrap CIs are in
`results/materials/statistical_results.json`.

Interpretation guide:
- Where `p ≥ 0.05`, report: *"No statistically significant paired difference was detected at N = 30."*
- Observed direction and magnitude are reported descriptively regardless of p-value.

### 4.6 Termination reasons

Allowed values: `QUERY_BUDGET_EXHAUSTED`, `PROPOSAL_CAP_REACHED`, `NO_VALID_CANDIDATE`.

Distribution across conditions is in `results/materials/aggregated_results.csv`
(`frac_*` columns).

---

## 5. Cross-Domain Comparison

**Shared metrics (directly comparable):**
- `attack_model_queries` — model evaluations at search phase
- `total_model_queries` — includes attribution overhead
- `proposal_attempts`, `runtime_seconds`, `edit_distance`

**Domain-specific metrics (not interchangeable):**
| Domain | Validity oracle | Success metric |
|---|---|---|
| Bio-Cyber | ACGT alphabet (always passes) | Label flip (`success ∈ {0,1}`) |
| Materials | RDKit SMILES parse + canonicalisation | NaN (no threshold) |

These validity systems are **not scientifically equivalent** and are not collapsed
into a single shared column in the cross-domain table.

See `results/cross_domain/attack_strategy_comparison.csv` for the unified table with
properly NaN-masked domain-specific columns.

---

## 6. Figures

| Figure | Content |
|---|---|
| `materials_drift_vs_budget.png` | Mean |ΔTg| vs attack model query budget, per condition |
| `materials_model_queries_vs_attack.png` | Attack model query consumption per condition and budget |
| `materials_total_queries_vs_attack.png` | Total model queries (attack + attribution) |
| `materials_proposal_overhead.png` | Mean proposal attempts per condition and budget |
| `materials_rdkit_proposal_validity.png` | RDKit SMILES parse validity rate of proposals |
| `materials_runtime_vs_attack.png` | Wall-clock runtime per condition |
| `materials_attribution_comparison.png` | Search-only drift vs end-to-end total query cost for attribution conditions |
| `cross_domain_model_query_cost.png` | Attack model query cost, both domains |
| `cross_domain_total_query_cost.png` | Total model query cost (including attribution), both domains |

No placeholder figures are present. Molecular similarity was not computed using a
chemical fingerprint oracle, so **no similarity figure is included**.

---

## 7. Defense Transfer

`MATERIALS_DEFENSE_TRANSFER = NOT_RUN`

The canonical 46-token defended model has not been independently verified for
vocabulary / preprocessing compatibility. It is not evaluated here.

---

## 8. Limitations

- N = 30 per domain limits statistical power; treat all p-values as exploratory.
- Materials attack success relies solely on continuous |ΔTg| drift; no regression
  threshold has been canonically established.
- Attribution guidance targets a fixed position throughout the search phase; adaptive
  recomputation after each accepted move was not implemented.
- Evolutionary search under budget = 5 has insufficient budget to run meaningful
  generations; termination by proposal cap is expected.

---

## 9. Reproducibility

All random state is derived deterministically per `(source_id, attack_condition, budget)`:
```python
seed_str  = f"{source_id}_{attack_condition}_{budget}_42"
rng_seed  = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
```

No shared global RNG stream is used across conditions.
