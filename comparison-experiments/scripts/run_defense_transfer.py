import pandas as pd
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add materials-adversarial to path to access models
sys.path.append(os.path.abspath("../materials-adversarial/src"))
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import PSmilesTokenizer, Vocabulary, PAD_TOKEN, UNK_TOKEN

def load_checkpoint(dir_path, vocab):
    pt_path = dir_path / "model.pt"
    if not pt_path.exists():
        return None, None, "No model.pt"
    
    metrics_path = dir_path / "metrics.json"
    if not metrics_path.exists():
        return None, None, "No metrics.json"
        
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    arch = metrics.get("architecture")
    if not arch:
        return None, None, "No architecture in metrics.json"
        
    device = torch.device("cpu")
    state_dict = torch.load(pt_path, map_location=device, weights_only=True)
    
    is_two_branch = any("branch_a" in k for k in state_dict.keys())
    max_seq_len = arch.get("max_seq_len", 128)
    
    v_size = state_dict["embedding.weight"].shape[0]
    
    if is_two_branch:
        model = TwoBranchTransformerRegressorModel(
            vocab_size=len(vocab) - 1,  # Because specialized_transformer does + 1
            d_model=arch["d_model"],
            n_layers=arch["n_layers"],
            n_heads=arch["n_heads"],
            dim_feedforward=arch["dim_feedforward"],
            dropout=arch.get("dropout", 0.1),
            max_seq_len=max_seq_len
        )
    else:
        model = TransformerRegressorModel(
            vocab_size=len(vocab),
            d_model=arch["d_model"],
            n_layers=arch["n_layers"],
            n_heads=arch["n_heads"],
            dim_feedforward=arch["dim_feedforward"],
            dropout=arch.get("dropout", 0.1),
            max_seq_len=max_seq_len
        )
        
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        return None, None, f"Strict load failed: {str(e)}"
        
    scaler = TargetScaler()
    sc_data = json.load(open(dir_path / "scaler.json"))
    scaler.mean = np.array(sc_data["mean"])
    scaler.std = np.array(sc_data["std"])
    return model, scaler, "LOAD_EXACTLY"

def predict_batch(model, sequences, vocab, max_seq_len, is_two_branch):
    model.eval()
    tokenizer = PSmilesTokenizer(vocab)
    preds = []
    
    for i in range(0, len(sequences), 32):
        batch = sequences[i:i+32]
        tokenized = []
        for s in batch:
            ids = tokenizer.encode(s, add_special_tokens=False, on_unknown="unk")
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            tokenized.append(ids)
            
        max_len = max(len(x) for x in tokenized)
        padded = np.full((len(batch), max_len), vocab.pad_id, dtype=np.int64)
        for j, t in enumerate(tokenized):
            padded[j, :len(t)] = t
            
        x = torch.tensor(padded, dtype=torch.long)
        mask = (x == vocab.pad_id)
        
        with torch.no_grad():
            p = model(x, mask)
            preds.extend(p.numpy().flatten().tolist())
    return preds

def evaluate_defense_matrix():
    inventory = pd.read_csv("results/defense_transfer/defense_inventory.csv")
    
    with open("../materials-adversarial/data/processed/vocab.json") as f:
        v = json.load(f)
    vocab = Vocabulary(itos=["<pad>", "<unk>", "<bos>", "<eos>"] + v)
    
    banks = {}
    for attack in ["random", "mcmc", "evolutionary"]:
        with open(f"results/frozen_candidate_banks/{attack}.jsonl", "r") as f:
            banks[attack] = [json.loads(line) for line in f]
            
    results = []
    coverage = []
    
    for idx, row in inventory.iterrows():
        name = row['defense_name']
        path_file = row['checkpoint_path']
        dir_path = Path("../" + path_file).parent
        
        model, scaler, status = load_checkpoint(dir_path, vocab)
        
        if status != "LOAD_EXACTLY":
            print(f"Skipping {name}: {status}")
            for attack in banks.keys():
                coverage.append({"defense": name, "attack": attack, "n_expected": 500, "n_evaluated": 0, "status": status})
            continue
            
        is_two_branch = isinstance(model, TwoBranchTransformerRegressorModel)
        metrics_path = dir_path / "metrics.json"
        with metrics_path.open("r") as f:
            arch = json.load(f)["architecture"]
            max_seq_len = arch.get("max_seq_len", 128)
            
        for attack, bank in banks.items():
            src_seqs = [b["source_sequence"] for b in bank]
            cand_seqs = [b["candidate_sequence"] for b in bank]
            
            s_preds = predict_batch(model, src_seqs, vocab, max_seq_len, is_two_branch)
            c_preds = predict_batch(model, cand_seqs, vocab, max_seq_len, is_two_branch)
            
            s_preds = scaler.inverse_transform(np.array(s_preds).reshape(-1, 1)).flatten()
            c_preds = scaler.inverse_transform(np.array(c_preds).reshape(-1, 1)).flatten()
            
            for i, b in enumerate(bank):
                s_pred = float(s_preds[i])
                c_pred = float(c_preds[i])
                drift = abs(s_pred - c_pred)
                is_success = (b["constraint_pass"] == True) and (drift > 0.05)
                
                results.append({
                    "defense_name": name,
                    "checkpoint_path": path_file,
                    "attack_family": attack,
                    "source_id": b["source_id"],
                    "candidate_id": i,
                    "absolute_drift": drift,
                    "configured_success": is_success
                })
            
            coverage.append({"defense": name, "attack": attack, "n_expected": len(bank), "n_evaluated": len(bank), "status": "EVALUATED"})
            
    df_res = pd.DataFrame(results)
    if len(df_res) > 0:
        os.makedirs("results/defense_transfer", exist_ok=True)
        df_res.to_csv("results/defense_transfer/raw_predictions.csv", index=False)
        
        stats = []
        for (def_name, path, att), grp in df_res.groupby(["defense_name", "checkpoint_path", "attack_family"]):
            stats.append({
                "defense_name": def_name,
                "checkpoint_path": path,
                "attack_family": att,
                "mean_drift": grp["absolute_drift"].mean(),
                "median_drift": grp["absolute_drift"].median(),
                "p90_drift": grp["absolute_drift"].quantile(0.9),
                "configured_success_rate": grp["configured_success"].mean()
            })
            
        df_stats = pd.DataFrame(stats)
        df_stats.to_csv("results/defense_transfer/defense_transfer_matrix.csv", index=False)
        
    df_cov = pd.DataFrame(coverage)
    df_cov.to_csv("results/defense_transfer/coverage.csv", index=False)
    
if __name__ == "__main__":
    evaluate_defense_matrix()
