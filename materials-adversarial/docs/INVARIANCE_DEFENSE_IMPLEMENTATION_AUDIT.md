# Phase 6 Implementation Audit

- **A. baseline fine-tuning initialization**: Verified. `build_model` explicitly loads weights from the baseline model checkpoint before unfreezing parameters for fine-tuning.
- **B. optimizer and LR**: Verified. Uses AdamW with preregistered LR and weight decay.
- **C. source/variant batching**: Verified. Groups source properties and predicts equivalent variants dynamically using random sampling over variants.
- **D. property loss**: Verified. F.mse_loss between `source_prediction` and `target`.
- **E. consistency loss**: Verified. Uses `anchored_consistency_loss` or `worst_consistency_loss` using MSE on features.
- **F. lambda weighting**: Verified. Validation loop scales consistency loss by exactly the lambda grid combinations.
- **G. training seed handling**: Verified. `np.random.default_rng(seed)` creates deterministic reproducible sampling and `seed_all()` covers PyTorch mechanisms.
- **H. transformation curriculum**: Verified. Bank generation script distinctly splits 'broader' (strict equivalence) vs 'repeat_shift'.
- **I. train/validation split isolation**: Verified. `splits["train"]` and `splits["val"]` correctly mapped and explicitly enforced. Test IDs generate `RuntimeError`.
- **J. branch-order held-out-family exclusion**: Verified. A separate unseen model filters `allowed = ["start_point_shift", "randomized_atom_ordering"]` to withhold `branch_ordering`.
- **K. tokenizer correction**: Verified. Both `valid_broader` and `valid_repeat` enforce `check["unknown_count"] == 0` for unknown token filtering, using `*` normalized tokens.
- **L. no test data loaders instantiated during validation selection**: Verified. `selection()` loop does not instantiate test sequences or datasets.
- **M. model checkpoint naming**: Verified. Saved in isolated `{arm}/{seed}/model.pt` hierarchies.
- **N. no accidental parameter freezing**: Verified. `trainable=True` loops over model parameters setting `requires_grad_(True)`.
- **O. no accidental test-time augmentation during validation selection**: Verified. Robustness metric evaluates strictly 1:N pairs identically to baseline without ensembling predictions.
