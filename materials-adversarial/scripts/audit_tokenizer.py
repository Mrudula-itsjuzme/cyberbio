import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from materials_adv.models.transformer import TransformerRegressorModel
from src.materials_adv.data.tokenizer import tokenize
from materials_adv.data.scaler import TargetScaler
from scripts.run_invariance_defense import load_test_banks, build_model, clean_metrics, load_inputs

ROOT = Path("/home/mrudula/Downloads/DL_cyberbio/.phase5b-worktree/materials-adversarial")
OUT = ROOT / "results/invariance_defense"

def predict_custom(model, texts, vocab, simulate_historical):
    mapping = {token: index + 1 for index, token in enumerate(vocab)}
    ids_list = []
    for text in texts:
        if simulate_historical:
            tokens = tokenize(text, on_unknown="unk")
        else:
            from src.materials_adv.robustness.equivalence import normalize_attachment_tokens
            tokens = tokenize(normalize_attachment_tokens(text), on_unknown="raise")
        
        values = [mapping.get(token, 0) for token in tokens]
        ids_list.append(values)
        
    width = max(len(t) for t in ids_list)
    ids = torch.zeros((len(texts), width), dtype=torch.long)
    mask = torch.ones((len(texts), width), dtype=torch.bool)
    for row, values in enumerate(ids_list):
        ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
        mask[row, : len(values)] = False
        
    model.eval()
    values_out = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            batch_ids = ids[start:start+64]
            batch_mask = mask[start:start+64]
            values_out.extend(model(batch_ids, padding_mask=batch_mask).numpy())
    return np.asarray(values_out, dtype=float)

def evaluate_custom(model, variants, vocab, scaler, simulate_historical):
    rows = []
    variant_col = "variant"
    source_col = "source"
    
    for source_id, group in variants.groupby("source_id", sort=False):
        src_text = str(group[source_col].iloc[0])
        var_texts = group[variant_col].astype(str).tolist()
        
        if simulate_historical:
            src_text = src_text.replace("[*]", "*")
            var_texts = [t.replace("[*]", "*") for t in var_texts]
            
        texts = [src_text, *var_texts]
        values = scaler.inverse_transform(predict_custom(model, texts, vocab, simulate_historical))
        
        source_val = values[0]
        max_drift = np.max(np.abs(values[1:] - source_val))
        rows.append({"source_id": source_id, "worst_case": max_drift})
        
    return pd.DataFrame(rows)["worst_case"].mean()

def run():
    frame, splits, vocab, scaler = load_inputs()
    repeat_bank, broad_bank = load_test_banks()
    
    baseline = build_model(vocab, trainable=False)
    
    defended = build_model(vocab, trainable=False)
    defended.load_state_dict(torch.load(OUT / "models/broader_worst/20260815/model.pt", map_location="cpu", weights_only=True))

    print("Historical Tokenizer + Baseline Model:")
    rep_h = evaluate_custom(baseline, repeat_bank, vocab, scaler, True)
    brd_h = evaluate_custom(baseline, broad_bank, vocab, scaler, True)
    print(f"Repeat/Shift worst mean: {rep_h:.4f}")
    print(f"Broader worst mean: {brd_h:.4f}")
    
    print("\nCorrected Tokenizer + Baseline Model:")
    rep_c = evaluate_custom(baseline, repeat_bank, vocab, scaler, False)
    brd_c = evaluate_custom(baseline, broad_bank, vocab, scaler, False)
    print(f"Repeat/Shift worst mean: {rep_c:.4f}")
    print(f"Broader worst mean: {brd_c:.4f}")

    print("\nCorrected Tokenizer + Defended Model:")
    rep_d = evaluate_custom(defended, repeat_bank, vocab, scaler, False)
    brd_d = evaluate_custom(defended, broad_bank, vocab, scaler, False)
    print(f"Repeat/Shift worst mean: {rep_d:.4f}")
    print(f"Broader worst mean: {brd_d:.4f}")

if __name__ == "__main__":
    run()
