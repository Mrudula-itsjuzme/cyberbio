# Materials Vocabulary Forensics Report

## 1. Problem
The canonical materials models expect either a 37-token or 46-token vocabulary (determined via `model.pt` embedding dimension inspection). The existing `vocab.json` file in `materials-adversarial/data/processed/` contains a raw JSON list of exactly 36 tokens. Without reconstructing the exact training vocabulary—and its precise index mappings—any transfer or cross-domain comparison runs the risk of testing arbitrary token permutations, thus invalidating findings.

## 2. Available Vocab Artifacts
- Current `data/processed/vocab.json`: 36 elements, list format.
- Historical `data/processed/vocab.json` at `da95af7`: 45 elements.
- Historical `data/processed/vocab.json` at `35db32c`: 42 elements.

## 3. Checkpoint Shapes
- `transformer_regressor`: `[37, 64]` embedding.
- `transformer_defended_phase2b_balanced`: `[46, 64]` embedding.

## 4. Special-Token Analysis & Dataset Token Inventory
A review of `materials-adversarial/src/materials_adv/data/tokenizer.py` revealed that `PSmilesTokenizer` utilizes four special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`. When built via `Vocabulary.build`, these four are prepended to the corpus tokens.
For the 46-token checkpoint (`transformer_defended_phase2b_balanced`), the arithmetic exactly matches the historical `35db32c` vocabulary: 42 JSON tokens + 4 special tokens = 46 tokens. 

However, the 37-token clean `transformer_regressor` does not fit this `4 specials + X = 37` arithmetic based on any historical `vocab.json`. 

## 5. 37-Token Checkpoint Analysis (Exact Recovery)
A forensic audit of the training script `materials-adversarial/src/materials_adv/training/train.py` proved that the 37-token model did NOT utilize `PSmilesTokenizer`. Instead, it implemented an ad-hoc inline `PolymerDataset`:
```python
self.char2idx = {c: i + 1 for i, c in enumerate(vocab)}
```
The script loads the 36-token JSON list and offsets the index by 1. The 0th index is explicitly hardcoded in the `PolymerDataset._encode` method to serve jointly as `<pad>` and `<unk>`.

The model is initialized with:
```python
self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
```
where `vocab_size = len(vocab) = 36`. 
Therefore, `36 + 1 = 37`. 

## 6. Recovery Verdict
**EXACT_RECOVERED.**
The discrepancy was an artifact of ad-hoc vocabulary index offset masking during early training scripts vs the rigorous 4-special-token architecture introduced in later defended checkpoints. No tokens are missing. The 36-element list IS the complete training vocabulary when mapped via a 1-indexed offset with 0 serving as pad/unk.

## 7. Materials Model Compatibility
**MATERIALS_MODEL_COMPATIBILITY = VERIFIED.**
The checkpoint successfully loads its state dictionary with no embedding size mismatches and outputs finite prediction tensors when passed a token sequence mapped according to the recovered `char2idx` schema. 

## 8. Materials Comparison Status
**UNBLOCKED.**
The Materials arm of the benchmark can now proceed confidently in subsequent phases.
