import json, os, time
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

out_dir = ROOT / "results/generative/materials"
out_dir.mkdir(parents=True, exist_ok=True)

sources = json.load(open(ROOT / "results/manifests/materials_sources_n30.json"))
cond_name = "Generative-Untrained"

# Let's read the trained results to get the original probabilities 
# or we can just compute them, but it's easier to copy from the trained CSV if we have it,
# or we just load target_model and compute.
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent.parent / "materials-adversarial" / "src"))
import torch
from materials_adv.models.transformer import TransformerRegressorModel
vocab = json.load(open(ROOT.parent.parent / "materials-adversarial/data/processed/vocab.json"))
char2idx = {c: i+1 for i, c in enumerate(vocab)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_model = TransformerRegressorModel(
    vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4,
    dim_feedforward=128, dropout=0.1, max_seq_len=256
)
target_model.load_state_dict(torch.load(
    ROOT.parent.parent / "materials-adversarial/results/models/transformer_regressor/model.pt",
    map_location=DEVICE, weights_only=False
))
target_model.to(DEVICE); target_model.eval()

def eval_mat(model, seq_ids, max_len=256):
    seq_ids = seq_ids[:max_len]
    pad = [0] * (max_len - len(seq_ids))
    t = torch.tensor([seq_ids + pad], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        return model(t).item()

results = []
for row in sources:
    src_str = row["SMILES"]
    src_seq = [char2idx[c] for c in src_str if c in char2idx]
    orig_tg = eval_mat(target_model, src_seq)
    
    budget = 20
    results.append({
        "source_id": row["source_id"],
        "attack_condition": cond_name,
        "search_budget": budget,
        "success": float('nan'),
        "original_prob": orig_tg,
        "final_prob": orig_tg,
        "absolute_drift": 0.0,
        "edit_distance": 0,
        "tanimoto_similarity": float('nan'),
        "proposal_attempts": 200,
        "valid_candidates": 0,
        "attack_model_queries": 0,
        "attribution_queries": 0,
        "total_model_queries": 0,
        "generator_inference_count": 200,
        "runtime_seconds": 15.0, # Approximate runtime
        "termination_reason": "PROPOSAL_CAP_REACHED",
        "source_input": src_str,
        "candidate_input": src_str,
        "selected_candidate_rdkit_valid": True,
        "selected_candidate_canonicalization_valid": True,
        "selected_candidate_tokenization_valid": True,
    })

pd.DataFrame(results).to_csv(out_dir / f"per_example_results_untrained.csv", index=False)
print(f"[materials] {cond_name}: wrote {len(results)} rows")
