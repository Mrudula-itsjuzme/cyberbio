# Transformer input-pipeline audit

The frozen wrapper uses the canonical chemical tokenizer, maps vocabulary tokens to IDs 1-36, and reserves ID 0 for padding. Unknown tokens also map to 0. This conflation is a real input-pipeline defect, although the corrected strict audit contains no unknown IDs.

The model uses a learned 256-position embedding, two Transformer encoder layers, a boolean `src_key_padding_mask`, and masked mean pooling. Padding outputs are excluded from pooling. No input is silently truncated by the active wrapper: sequences above 256 raise `ValueError`. Observed maxima were 123 source tokens and 127 variant tokens; truncation count was zero.

Ten-repeat inference was exactly deterministic (`0 eV` maximum repeat variation). Alone, pair, same-length, mixed-length and reordered-batch controls agreed within `7.15e-7 eV`. Therefore neither dropout, padding, masking nor batch composition explains the observed eV-scale drift.

Token-position displacement has a small positive descriptive association with strict drift (`Spearman rho=0.2125`). Together with learned absolute positional embeddings, distinct token orders, and preserved chemistry, the supported mechanism is sequence-order/positional-encoding sensitivity. This is not a causal decomposition.
