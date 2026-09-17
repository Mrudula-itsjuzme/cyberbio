# Framework V2 Benchmark Forensic Audit

**Scope:** `results/developmental_run_1/` (frozen copy at
`results/framework_v2/developmental_run_1/`) and
`docs/FRAMEWORK_V2_ATTACK_RESULTS.md`.
**Branch:** `framework-v2-new-attacks` @ `8fd9d85`.
**Auditor artefacts:** `results/framework_v2/audit/` +
`results/framework_v2/audit_top20_candidates.csv`.
**Reproduce:** `.venv/bin/python scripts/audit_framework_v2_benchmark.py`
**Verdict: `MULTIPLE_PROTOCOL_ERRORS`.** No Framework V2 attack number may be cited.

---

## 0. Headline: the 6–7 eV mean drift is an artefact of a placeholder predictor

`scripts/benchmark_v2.py` never loaded a model. It scored candidates with:

```python
class SafeGraphPredictor:
    def predict(self, candidate) -> float:
        return float(hash(candidate.identifier) % 1000) / 100.0
```

`hash()` here is CPython's **process-local, salt-randomised** string hash. The value is
uniform on `[0, 10)`, has no relationship to the GraphMPNN, and is not even reproducible
between two runs of the same script (`PYTHONHASHSEED`). The reported benchmark is the
expected value of maximising `|f(cand) − f(src)|` over two independent uniform draws and
climbing towards `10 × 2/3 ≈ 6.67 eV` as `Q` grows:

| | reported (stub) | this audit (real GraphMPNN) |
|---|---|---|
| Evolutionary + SimpleSubst, Q10 | 6.438 | 0.887 |
| Evolutionary + SimpleSubst, Q20 | 6.767 | 1.059 |
| Evolutionary + SimpleSubst, Q50 | 6.950 | 1.379 |
| Greedy + SimpleSubst, Q50 max | 9.59 | 2.928 |
| Motif Q50 mean | 5.937 | 0.433 |

The two distributions are separated by a factor of ~5. `max drift = 9.59 eV` is itself
impossible for a real model whose entire prediction range is ~2.1–8.1 eV.

The mock is not a subtle numerical defect: it is the absence of the pipeline being
studied. Everything downstream is therefore void, and the three protocol defects below
would each independently invalidate the run.

---

## 1. Source split audit — **`SOURCE_SPLIT_BUG`**

`data/processed/splits.json` (sha256 `4a9d2825…0ccd02`): train **2946**, validation
**631**, test **632**, `test_sealed = true`.

The 30 sources actually used by the developmental run are **not** validation-only:

| split of origin | count |
|---|---|
| train | **25** |
| test | **5** |
| validation | 0 |

Exact fingerprint of the mechanism: `random.seed(42); pool = list(range(4209));
random.shuffle(pool); first 30 with >3 atoms` reproduces the recorded 30 SMILES **in
order**. The run sampled from all of `processed.csv`.

Note the on-disk script now reads `splits.get("val")`, which *would* be validation-only —
so `benchmark_v2.py` was edited after the artefacts were produced, and the recorded
artefacts were never regenerated. A stale, contaminated run is being presented as the
V2 result.

**Action taken:** per protocol, sources were rebuilt from the canonical validation split
(`random.seed(42)` over `splits["val"]`, first 30 with >3 atoms; all 30 confirmed in
`val`, 0 in train, 0 in test). Every forensic number below uses those rebuilt sources.
Evidence: `results/framework_v2/audit/source_split_audit.csv`.

---

## 2. Model / checkpoint audit — **PASS**

| field | value |
|---|---|
| model class | `materials_adv.models.graph_predictor.GraphPredictor` |
| checkpoint path | `results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt` |
| checkpoint sha256 | `3fd339c5ebd0d700a05868a0d53ee752c61f378155245855252797c01bcbb81b` |
| `RELEASE_MANIFEST.json` sha | `3fd339c5ebd0…bb81b` — **matches** |
| node_dim / hidden_dim / num_layers | 7 / 64 / 3 |
| parameters | **27 585** (expected 27 585) |
| target scaler | `results/models/transformer_regressor/scaler.json`, sha `5ef15ad1…b12814` — **matches manifest** |

The canonical model and scaler are intact. Note that `results/models/` contains no
GraphMPNN at all — the only surviving GraphMPNN checkpoint is the Phase 11C copy — which
is consistent with the V2 benchmark having no model binding to begin with.

---

## 3. Predictor equivalence — **`INVALID_MODEL_PIPELINE`**

Twenty unchanged validation polymers, scored by (A) the canonical Phase 11C evaluator
(`run_phase11c_canonical_verification.eval_model_canonical(is_graph=True)`) and (B) the
Framework-V2 predictor wrapper:

**Framework V2 shipped no predictor wrapper at all.** The library had an abstract
`Predictor` interface and nothing implementing it, so the benchmark invented a stub.

| comparison | mean abs diff | max abs diff |
|---|---|---|
| canonical vs stub (**what ran**) | **3.077 eV** | **6.014 eV** |
| canonical vs `GraphMPNNPredictor` (added by this audit) | 0.000000 | 0.000000 |

`src/materials_adv/framework/predictors.py` now provides `GraphMPNNPredictor`, which
binds the canonical checkpoint, the canonical scaler, and counts forward passes so that
query accounting can be audited. Bit-identical to the canonical evaluator.

---

## 4. Validation reproduction — **PASS**

Through the V2 wrapper over the full validation split (n = 631):

| metric | value | canonical |
|---|---|---|
| MAE | **0.411190 eV** | 0.41119 |
| RMSE | 0.595369 eV | — |
| R² | 0.828639 | — |

The canonical 0.411 eV validation MAE reproduces exactly, so the model, scaler, graph
featurisation and batching paths are correct. The failure is confined to the benchmark.

---

## 5. Scaler / unit forensics — **PASS (for the real pipeline)**

Five source/candidate pairs traced end-to-end in
`results/framework_v2/audit/scaler_forensics.csv`: raw model output → one
`inverse_transform` → eV, for both source and candidate, then
`drift = abs(candidate_eV − source_eV)`.

* Inverse scaling is applied **exactly once** (traced drift vs library drift differ by
  ≤ 5.9e-07 eV, i.e. float32 noise).
* **No double inverse scaling.** Applying `inverse_transform` twice maps a 7.24 eV source
  to 15.01 eV; the pipeline does this nowhere. (Detected-case ratio if it did: ×1.456,
  the scaler std.)
* **No scaled-vs-unscaled subtraction.** Subtracting in scaled space returns ≈0.687× the
  eV drift, and the quantity is unitless.
* **Correct scaler statistics.** The canonical scaler equals a scaler refit on the train
  split to 1e-9 (mean 4.474831, std 1.456305), i.e. it was fit on train, not on all data.
  An all-data fit would shift the mean by 0.011 and the drift scale by ~0.7%.

The developmental V2 benchmark bypassed the scaler entirely because it bypassed the model.

---

## 6. Prediction range audit

Over 540 reconstructed V2 candidates:

| | min | median | mean | max |
|---|---|---|---|---|
| source predictions (eV) | 2.337 | 5.218 | 5.005 | 7.237 |
| candidate predictions (eV) | 2.086 | 4.574 | 4.546 | 8.113 |

Canonical training target range `[0.071, 9.835]`; validation target range `[0.912, 9.680]`.

**All 540 candidates are `IN_TRAIN_RANGE`.** Zero outside-train-range, zero extreme
extrapolation. The developmental run's 9.59 eV max "drift" is therefore not merely large,
it is outside the range any prediction (or target) in the dataset could produce.
No predictions were clipped.

---

## 7. Edit-budget audit

`Budget.validate_edit_distance` counts entries in `Candidate.provenance` — i.e. **operator
applications**, not chemical edit distance. It is a metadata check, and it is not an
independent measure.

Independent measure used here: `element_change_count + |atom_count_delta|`, computed from
the graphs. This is invariant under RDKit canonicalisation, unlike SMILES string distance.

| operator | independent edits per candidate | >3 |
|---|---|---|
| SimpleSubst | 1×220, 2×42, 3×8 (max **3**) | **0** |
| Motif | 0×45, 1×150, 2×51, 3×23, **4×1** | 1 (0.37%) |

So the stored counter does not *equal* the chemistry for either operator, but for the
substitution operator the `≤3` budget is genuinely respected — `>3 = 0` **passes**.

**Methodological warning (important).** A naive re-audit using SMILES Levenshtein distance
would report **301/540 (55.7%) budget violations with distances up to 28**, and would
"reproduce" an edit-creep narrative. That is a false positive: every V2 candidate is
re-canonicalised by `Chem.MolToSmiles`, and canonicalisation alone inflates a *single*
atom substitution to a string distance of 10–16 (max 44 observed). Canonicalisation
changes strings, not chemistry. Any future edit-budget claim must be graph-based.

---

## 8. Chemistry validity audit

Per-candidate, independent of stored metadata (RDKit parse, sanitisation, valence,
component count, wildcard/attachment preservation, graph identity):

| check | result |
|---|---|
| parse + sanitize + valence | 540 / 540 |
| single connected component | 540 / 540 |
| wildcard count preserved (attachment semantics) | 540 / 540 |
| graph differs from source | 495 / 540 |
| **chemically identical no-ops** | **45** |

All 45 "invalid" candidates are no-ops whose canonical graph equals the source. Their
measured drift is exactly 0.0 eV (the model is deterministic), so they contribute no
artefact drift — but they consumed 45 of the query budget while producing nothing.

Latent defect, not triggered here: `GraphDataset.__getitem__` **silently substitutes a
neighbouring molecule** (index `idx+1`) when a SMILES fails to parse, instead of raising.
Unparseable candidates would be scored as if they were valid chemistry.

---

## 9. Atom / graph size audit

| quantity | mean | max |
|---|---|---|
| atom delta | +0.026 | **+3** |
| heavy-atom delta | +0.026 | +3 |
| formal-charge delta | 0.000 | 0 |
| components | 1 | 1 |

Simple substitution does not grow molecules (mean delta 0.000). Only the motif operator
grows anything at all (mean +0.052, one candidate +3 atoms from a CF3 insertion). No
charge changes. No flagged structural blow-ups.

---

## 10. `FunctionalGroupReplacementAttack` — **not functional-group replacement**

Mechanism: `Chem.ReplaceSubstructs` with SMARTS `[C;X4;h1,h2,h3]` against replacements
`O, N, F, C#N, C(F)(F)F`.

Observed: 3 of 5 replacements are single atoms. Products are overwhelmingly **single-atom
substitutions of an aliphatic carbon**, no disconnected fragments (components ≡ 1 — the
extra nitrile/CF3 atoms are bonded to the attachment atom by `ReplaceSubstructs`), and some
products have **zero element change** (bond/ring rearrangements). One application can cost
up to **4 independent edits** (max graph-change proxy 7).

**Recommendation: rename to `AliphaticCarbonSubstitutionAttack`** (or
`AtomMotifSubstitutionAttack`). Do not use "functional-group replacement" as a scientific
claim. `docs/PAPER_DRAFT.md` §8 already labels it `IMPLEMENTED_NOT_EVALUATED`, which is
correct, but the operator name over-promises relative to its implementation.

---

## 11. Motif budget complexity

One motif application ≠ one atom substitution. `operator_edit_count = 1` (provenance)
while `independent_edit_count ∈ {0,1,2,3,4}` and the graph-change proxy reaches 7.
Both numbers must be reported whenever motif operators are used, and the motif operator's
budget must be stated in *operator applications*, not "edits". Direct numeric comparison
of SimpleSubst against Motif would conflate a strategy effect with an operator-capacity
effect.

---

## 12. Query accounting — **`QUERY_ACCOUNTING_INVALID`**

Independently instrumented over all 540 (strategy × operator × Q × source) runs:

| check | result |
|---|---|
| runs where `predictor` asks exceed declared `Q` | **482 / 540** |
| runs where the budget counter understates asks | **540 / 540** |
| overshoot range | **+5 to +60 asks** |
| duplicate valid proposals | 774 |
| invalid proposals | 45 |

Scoring occurs outside the budget counter in four distinct ways:

1. every `AttackObjective.evaluate` calls `predictor.predict` **twice** (candidate *and*
   source) while consuming **one** query — a systematic 2× under-count;
2. `GenericGreedySearch` calls `evaluate(...)` on the current candidate *after* the budget
   loop, with no budget check;
3. `benchmark_v2.py` recomputes drift as `predict(res)` and `predict(source)` outside the
   counter;
4. `EvolutionarySearch` repopulation calls `operator.apply` / `validator.is_valid` with no
   accounting, so generation cost is invisible.

Fixing this requires a single budgeted scoring entry point (a counting predictor wrapper
such as `GraphMPNNPredictor`), not four independent call sites.

---

## 13. Provenance of "Random Q50 ≈ 6.70 eV"

The claim **does** have an artefact — and the artefact is invalid.

| field | value |
|---|---|
| file | `results/framework_v2/developmental_run_1/attack_results.csv` |
| sha256 | `4554a40ce8b8b8d2c9e70b38f1fd4567cf26c383d9a81cfe58763dd174712525` |
| column | `Mean Drift` (Strategy = Random, Operator = SimpleSubst, Q = 50) |
| value | **6.7067** |
| source count | 30 (25 train + 5 test) |
| aggregation | unweighted mean of the per-source max `\|drift\|` |

Invalid because `Mean Drift` was computed with the hash stub and because 30/30 sources are
out-of-split. **Remove the claim** from every table and doc.

---

## 14. Phase 12B bridge test (the key experiment)

Identical sources (first 10 validation polymers), identical model (canonical GraphMPNN),
identical `Q = 10`, identical edit budget `3`, identical `seed = 42`. Operator semantics
differ by design, as documented. Canonical pipeline = the unbound Phase 12B methods
(`mutate_smiles` / `check_validity` / `generate_valid_neighbor` / `search_random`) bound to
a harness backed by the real model, i.e. the exact canonical algorithm.

| | Phase 12B canonical | Framework V2 rebuilt |
|---|---|---|
| mean drift | **0.713 eV** | **0.589 eV** |
| max drift | **1.555 eV** | **1.072 eV** |
| mean token edit distance | 1.0 | 8.3 (canonicalisation-inflated) |
| mean independent edits | 1.0 | 1.0 |

Ratio V2/Phase12B = **0.83**. With the real model, a V2 random search over atom-level
substitutions lands in the *same regime* as the canonical Phase 12B pipeline. The 6–7 eV
spread is entirely an artefact of the stub predictor; there is no remaining anomaly to
explain. Evidence: `results/framework_v2/audit/phase12b_bridge.csv`.

---

## 15. One-edit sanity check

10 validation polymers, every valid one-atom substitution enumerated (424 candidates,
all with independent edit count 1):

| mean | median | p95 | max |
|---|---|---|---|
| **0.213 eV** | 0.104 eV | 0.839 eV | **1.547 eV** |

Zero candidates above the canonical bounded maximum of 3.19 eV. The largest one-edit drift
is an aromatic `C→S` re-typing of a chlorinated ring in
`[*]Oc1c(Cl)cc(NC(=O)NCCCCCCNC([*])=O)cc1Cl` (4.720 → 3.173 eV). Its string edit distance
is 11 for a one-atom edit — the canonicalisation artefact in section 7, visible directly.
Note 23% of enumerated one-edit sites are **aromatic** atoms, which the Phase 12B token
mutator cannot reach at all. Evidence: `results/framework_v2/audit/one_edit_sanity.csv`.

---

## 16. Top-20 forensic table

`results/framework_v2/audit_top20_candidates.csv` (sha256 `d2450a2f…a8d11b`), required
schema plus diagnostic columns.

* 20/20 valid, single-component, attachment-preserving, all `IN_TRAIN_RANGE`.
* All 20 are `SimpleSubst` (12 Evolutionary, 5 Greedy, 3 Random); max independent edit
  count **3** — the budget holds.
* Independent edit counts: 1 edit ×5, 2 edits ×9, 3 edits ×6; **0/20 exceed 3.19 eV**.
* Max drift **2.928 eV** (Greedy/Evolutionary, Q50), i.e. below the canonical bounded
  maximum.
* `site_was_aromatic` is populated for the 4 single-atom-mappable rows only: multi-edit
  candidates are not atom-mappable by re-enumeration, so **NaN means "multi-edit", not
  "missing evidence"**.

---

## 17. Operator semantics: V2 SimpleSubst vs canonical Phase 12B

| axis | V2 `AtomSubstitutionAttack` | Phase 12B `mutate_smiles` |
|---|---|---|
| granularity | RDKit **atom**-level (`RWMol.ReplaceAtom`) | SMILES **token**-level (chemical tokenizer) |
| allowed atoms | C, N, O, S, F | C, N, O, S, F, Cl, Br, I, P, B, Si |
| replacement set | C, N, O, S, F | C, N, O, S, F, Cl, Br, I |
| wildcards | implicitly protected (`GetAtomicNum() > 1` excludes `[*]`); count preserved | bracket atoms are single tokens, never allowed |
| bonds | untouched; sanitisation may re-perceive aromaticity | untouched |
| attachment protection | none beyond `[*]` itself — **43/416 sites are adjacent to an attachment atom** | none explicit |
| aromaticity | **mutable** (~11% of realised substitutions re-type an aromatic atom) | **immutable** (lowercase `c`/`n` are not allowed tokens) |
| charge | mutable (no guard) | immutable (bracket atoms) |
| validation | `SanitizeMol` per mutation | RDKit parse + canonical ≠ original + graph build + token edit ≤ 3 |
| edit unit | provenance entries | token Levenshtein from the **original** |

Empirical: over 10 validation polymers, V2 exposes **204 mutable heavy atoms including 47
aromatic atoms**; Phase 12B exposes **142 mutable token positions**, all aliphatic.

**These are not the same threat model.** V2 can re-type aromatic and charged atoms and can
edit backbone atoms adjacent to the polymer attachment point; Phase 12B cannot. Their
numeric strengths must never be compared directly.

---

## 18. Targeted "100% success" — rename required

Definition in code: fraction of sources where `sign(f(x_adv) − f(x_src))` matches the
requested direction. That is **direction only** — no threshold, no effect size, no
robustness meaning. Correct name: **`DIRECTIONAL_SUCCESS_RATE`**.

It is not evidence of attack strength: with the stub predictor and a search maximising
`|drift|`, directional success is near-guaranteed (SimpleSubst reported 1.00;
Motif 0.833). No generic "attack success" claim may be derived from it. The
`Mean Signed Change` column (+4.58 / −5.00 eV) is likewise stub-valued.

---

## 19. Scaffold operator status

`ScaffoldPreservingAttack` requires a `core_scaffold_smiles` argument, is absent from the
developmental run, and has never been structurally verified against the original scaffold
with substructure matching. **Status: `IMPLEMENTED_NOT_EVALUATED`.** Must stay out of all
scientific tables.

**Framework isolation caveat.** `materials_adv.framework.{interfaces,objectives,budget}`
are genuinely dependency-free (verified in a subprocess: `rdkit` never enters
`sys.modules`), but `materials_adv.attacks.search.evolutionary` imports
`RDKitValidityChecker` at **module level**, so importing the "generic" search layer drags
RDKit in regardless of the validator passed in. Both behaviours are pinned by regression
tests. The framework is domain-agnostic in its interfaces, not yet in its import graph.

---

## 20. Paper safety

Searched `docs/` and the root `.tex` files for `6.949`, `6.766`, `6.438`, `6.70`,
`5.937`, `83.3%`, `100%`.

* The unverified V2 numbers appear **only** in `docs/FRAMEWORK_V2_ATTACK_RESULTS.md` (and
  its frozen copy). That file now carries a `DEVELOPMENTAL_UNVERIFIED` banner.
* `docs/PAPER_DRAFT.md` §8 and `docs/VIVA_DEFENSE_PACK.md` mention Framework V2 but already
  label the new families `IMPLEMENTED_NOT_EVALUATED` / `PROTOCOL_ONLY` and cite **no V2
  numbers**. No numerical claim needs retracting from the paper — only the operator
  naming (§10) and any implicit "we evaluated V2" framing.
* The `100%` hits elsewhere (`ARCHITECTURE_FORENSIC.md`, `PAIRED_BENCHMARK_AUDIT.md`) are
  unrelated length-preserving/scaler-frozen checks, not V2 claims.

---

## 21. Final verdict

```
MULTIPLE_PROTOCOL_ERRORS
```

Independently fatal defects: `INVALID_SOURCE_SPLIT` (25/30 train, 5/30 test),
`INVALID_MODEL_PIPELINE` (hash stub, no predictor bound), `QUERY_ACCOUNTING_INVALID`
(482/540 runs over budget, 540/540 under-counted). Clean: model checkpoint & scaler
integrity, predictor equivalence of the new wrapper, validation MAE reproduction,
unit/scaling, chemistry validity, `≤3`-edit budget for the substitution operator.

**No Framework V2 attack result may enter the paper.** Framework V2 remains a
*scaffolding* contribution (`docs/PAPER_DRAFT.md` §8 framing is the correct one).

Per protocol, no new `results/framework_v2/verified_run_1/` benchmark was executed, since
the audit does not pass.

---

## 22. Exact next action

Rebuild the benchmark before any V2 number is produced again:

1. replace `SafeGraphPredictor` with `GraphMPNNPredictor` (done: it now exists);
2. route **all** scoring through a single budgeted entry point so `predict` calls are
   exactly `Q` (fixes the 2× objective under-count and the greedy post-loop call);
3. fix `GraphDataset.__getitem__` to raise on unparseable SMILES instead of silently
   substituting a neighbour;
4. replace `Budget.validate_edit_distance` with a graph-based measure, and report
   `operator_edit_count` **and** graph-level change for motif operators;
5. rename `FunctionalGroupReplacementAttack`;
6. re-run on validation-only sources into `results/framework_v2/verified_run_1/`, with
   random / greedy / metropolis / evolutionary compared under *identical* operator, model,
   budget and query schedule, and operators compared separately from strategies.

---

## 23. Verification

```
pytest                               385 passed, 1 skipped
  tests/test_framework_v2_audit.py    24 passed   (new)
python -m compileall src scripts tests       exit 0
git diff --check                             exit 0
```

Regression coverage added: predictor == canonical evaluator · checkpoint/scaler hashes ·
27 585 params · validation MAE ≈ 0.41119 · single inverse transform · drift unit = eV ·
validation-only source selection · full-dataset-shuffle fingerprint of the recorded run ·
`≤3`-edit budget for the substitution operator · provenance-counter ≠ edit distance ·
string-distance false positive · 2× objective query under-count · budget never over-counts ·
forward-call accounting · validity/attachment/component gates · motif operator is atom-level ·
threat-model difference vs Phase 12B · directional-success is sign-only ·
framework RDKit isolation · top-20 table schema.

---

## Appendix: artefact hashes

```
results/developmental_run_1/attack_results.csv      4554a40ce5b8b8d2c9e70b38f1fd4567cf26c383d9a81cfe58763dd174712525
results/developmental_run_1/benchmark_sources.csv    8d801a1987282eb8eb9e1121d57e63c48adc2acad569bd0a62a46b0e9fb7dbd1
results/developmental_run_1/targeted_results.csv     563f5822fa23f8158478e9f93c3804ce3f1ca371551050bf388c8efa328bc6d2
results/framework_v2/audit_top20_candidates.csv      d2450a2f27a5381ab1e66fadd7f3e27cdc656b9dc195dd6815bd46e074a8d11b
(GraphMPNN checkpoint / scaler hashes as in section 2)
```
