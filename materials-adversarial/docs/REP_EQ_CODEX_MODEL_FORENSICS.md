# Representation-equivalence model forensics

## Finding

The canonical sequence research target is the repository's original
`TransformerRegressorModel`, a two-layer PyTorch Transformer encoder trained to
predict polyVERSE polymer bandgap in eV.  It is not a TransPolymer checkpoint,
and it is not GraphMPNN.  Later project phases promoted GraphMPNN for a separate
architecture comparison, but that does not replace the frozen sequence model
whose representation sensitivity is under study here.

## Evidence chain

| Question | Evidence |
|---|---|
| model class | `src/materials_adv/models/transformer.py`, class `TransformerRegressorModel` |
| frozen checkpoint | `results/models/transformer_regressor/model.pt`, SHA-256 `0b1bb3f62ca72a897fd84c8e7f3129f8a7a9d75c1179c10debb63e46113731e4` |
| scaler | `results/models/transformer_regressor/scaler.json`, SHA-256 `5ef15ad16aee156fbc9b6b7872c1534262e3622022190f17c23fc7e3dab12814` |
| architecture/config | `configs/model.yaml`: 64 dimensions, 2 layers, 4 heads, 128 feed-forward dimensions, mean pooling, 256 positions |
| loader/inference | `src/materials_adv/models/regression.py`; `scripts/run_representation_attribution.py`; `scripts/run_phase11c_canonical_verification.py` |
| tokenizer | `src/materials_adv/data/tokenizer.py:tokenize`; the historical predictor calls this function directly |
| frozen vocabulary | `data/processed/vocab.json`, 36 tokens, SHA-256 `3dd4c162c5d84138d7bf0ea4fb04a0cb81b1b9e9ee705a0e0318a3214d8935bf` |
| target | `configs/dataset.yaml` and `data/processed/processed.csv`: `Bandgap`, eV |
| dataset | polyVERSE `data/raw/bandgap_chain.csv`, materialized as `data/processed/processed.csv` |
| split | `data/processed/splits.json`, SHA-256 `4a9d28252d058c0c2e39cf70eb10dd2f79ad57df4e842c3f917cdfe9830ccd02`: train 2,946, validation 631, sealed test 632 |
| historical metric | `results/models/transformer_regressor/metrics.json`: test MAE 0.4619163 eV, RMSE 0.6235057 eV, R2 0.8019493 |
| history | commit `66385a8` introduced the baseline sequence model; later Phase 11C files explicitly load this checkpoint as `Transformer_Ordinary` alongside, rather than as, GraphMPNN |

The words "TransPolymer" in narrative documents describe the architecture
family and motivation.  No TransPolymer pretrained checkpoint or compatible
TransPolymer model loader is used by the canonical result path.

## Checkpoint reproduction

The new audit loaded the checkpoint with the same class, architecture, scaler,
tokenizer and vocabulary.  Predictions for all 631 historical validation rows
were compared with `results/representation_attribution/baseline_predictions.csv`.
Maximum absolute difference was `3.1885e-7 eV`, passing the predeclared absolute
tolerance of `1e-6 eV`.  Details are frozen in
`results/representation_equivalence_sequence/model_manifest.json`.

## Existing Phase 5 artifact classification

The pre-existing experiment was inspected in the user's original dirty
worktree and left untouched.  Its technically correct core is retained only as
`GRAPH_MODEL_PERMUTATION_INVARIANCE_CONTROL`.

| Artifact | Class | Reason |
|---|---|---|
| `scripts/representation_equivalence_audit.py` | WRONG_MODEL | runs `GraphMPNNPredictor`, samples incorrectly when no split column, canonicalizes away polymer attachment conventions, and hard-codes a post-hoc 0.1 threshold |
| `scripts/representation_equivalence_audit2.py` | WRONG_MODEL | same wrong target; GraphMPNN-only rerun |
| `docs/REPRESENTATION_EQUIVALENCE_FEASIBILITY.md` | PARTIAL | valid description of the graph control, invalid as a conclusion about the sequence target |
| `docs/REPRESENTATION_EQUIVALENCE_RESEARCH_GATE.md` | INVALID | "mathematically impossible" and STOP claims overgeneralize the graph control |
| `results/representation_equivalence/cohort_manifest.json` | PARTIAL | only 15 sources and lacks source ids, labels, rows, split and seed |
| `equivalent_variants.csv`, `equivalence_validation.csv` | VALID_CONTROL | graph-equivalent randomized serializations, but only for the GraphMPNN control and missing required validation metadata |
| `predictions.csv` | VALID_CONTROL | real GraphMPNN predictions, but no source prediction or required pair fields |
| `source_robustness.csv`, `error_vs_invariance.csv`, `top_failures.csv` | VALID_CONTROL | reusable graph-control calculations; incomplete schema and only 15 sources |
| `family_results.csv` | VALID_CONTROL | graph-control family summary only |
| eight PNG files | VALID_CONTROL | real figures from the graph control; filenames and claims must not be used for the sequence result |

The reusable pieces are RDKit graph-identity checking and source-level drift
aggregation.  The wrong-model scripts, post-hoc threshold, missing split lock,
and overbroad conclusions are not reused.
