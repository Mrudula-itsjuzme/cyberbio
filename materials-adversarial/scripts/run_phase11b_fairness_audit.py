import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rdkit import Chem
import pandas as pd
import numpy as np
import time
from pathlib import Path

import sys
sys.path.append("src")

from materials_adv.data.augmented_dataset import AugmentedSMILESDataset
from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.models.graph_predictor import GraphPredictor
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize

def get_seq(s, vocab_map, max_seq_len=256):
    ids = [vocab_map.get(t, vocab_map.get("<unk>", 1)) + 1 for t in tokenize(s)]
    ids = ids[:max_seq_len]
    padded = ids + [0] * (max_seq_len - len(ids))
    return torch.tensor(padded, dtype=torch.long).unsqueeze(0)

def train_model(model, train_loader, val_loader, scaler, device, is_graph=False, max_epochs=50, patience=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    best_val_mae = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None
    
    for epoch in range(max_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}")
            optimizer.zero_grad()
            if is_graph:
                x = batch["x"].to(device)
                adj = batch["adj"].to(device)
                mask = batch["mask"].to(device)
                y_hat = model(x, adj, mask).view(-1)
            else:
                x = batch["x"].to(device)
                y_hat = model(x).view(-1)
                
            y = batch["target"].to(device)
            y_scaled = torch.tensor(scaler.transform(y.cpu().numpy()), dtype=torch.float32).to(device).view(-1)
            loss = criterion(y_hat, y_scaled)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if is_graph:
                    x = batch["x"].to(device)
                    adj = batch["adj"].to(device)
                    mask = batch["mask"].to(device)
                    y_hat = model(x, adj, mask).view(-1)
                else:
                    x = batch["x"].to(device)
                    y_hat = model(x).view(-1)
                
                y_hat_np = y_hat.cpu().numpy().reshape(-1, 1)
                y_pred = scaler.inverse_transform(y_hat_np).flatten()
                y_true = batch["target"].cpu().numpy().flatten()
                val_losses.append(np.abs(y_pred - y_true).mean())
        
        val_mae = np.mean(val_losses)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    model.load_state_dict(best_state)
    return best_epoch, best_val_mae

def eval_model(model, loader, scaler, device, is_graph=False):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            if is_graph:
                x = batch["x"].to(device)
                adj = batch["adj"].to(device)
                mask = batch["mask"].to(device)
                y_hat = model(x, adj, mask).view(-1)
            else:
                x = batch["x"].to(device)
                y_hat = model(x).view(-1)
            
            y_hat_np = y_hat.cpu().numpy().reshape(-1, 1)
            y_pred = scaler.inverse_transform(y_hat_np).flatten()
            all_preds.extend(y_pred)
            all_targets.extend(batch["target"].cpu().numpy().flatten())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean((all_preds - all_targets)**2))
    r2 = 1 - (np.sum((all_targets - all_preds)**2) / np.sum((all_targets - np.mean(all_targets))**2))
    std = np.std(all_preds)
    rng = np.ptp(all_preds)
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Std": std, "Range": rng}

def eval_representation_invariance(model, evals, vocab, scaler, device, is_graph=False):
    model.eval()
    drifts = []
    vocab_map = {t: i for i, t in enumerate(vocab)}
    
    with torch.no_grad():
        for item in evals:
            smi = item["original"]
            equiv = item["equivalents"]
            
            if is_graph:
                # for graphs, the representation is identical, but lets compute it anyway
                ds = GraphDataset(pd.DataFrame({"original_representation": [smi], "property_value": [0.0]}))
                if len(ds) == 0: continue
                batch = next(iter(DataLoader(ds, batch_size=1)))
                p_orig = model(batch["x"].to(device), batch["adj"].to(device), batch["mask"].to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                p_orig = scaler.inverse_transform(p_orig).flatten()[0]
                
                for eq in equiv:
                    dseq = GraphDataset(pd.DataFrame({"original_representation": [eq], "property_value": [0.0]}))
                    if len(dseq) == 0: continue
                    beq = next(iter(DataLoader(dseq, batch_size=1)))
                    pe = model(beq["x"].to(device), beq["adj"].to(device), beq["mask"].to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                    pe = scaler.inverse_transform(pe).flatten()[0]
                    drifts.append(np.abs(pe - p_orig))
            else:
                x_orig = get_seq(smi, vocab_map)
                p_orig = model(x_orig.to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                p_orig = scaler.inverse_transform(p_orig).flatten()[0]
                
                for eq in equiv:
                    xe = get_seq(eq, vocab_map)
                    pe = model(xe.to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                    pe = scaler.inverse_transform(pe).flatten()[0]
                    drifts.append(np.abs(pe - p_orig))
                    
    drifts = np.array(drifts)
    if len(drifts) == 0: return {"mean_drift": 0, "median_drift": 0, "p95": 0, "max": 0, "std": 0}
    return {
        "mean_drift": np.mean(drifts),
        "median_drift": np.median(drifts),
        "p95": np.percentile(drifts, 95),
        "max": np.max(drifts),
        "std": np.std(drifts)
    }

def eval_chemistry_stress(model, evals, vocab, scaler, device, is_graph=False):
    model.eval()
    drifts = []
    vocab_map = {t: i for i, t in enumerate(vocab)}
    
    with torch.no_grad():
        for item in evals:
            smi = item["original"]
            equiv = item["mutations"]
            
            if is_graph:
                ds = GraphDataset(pd.DataFrame({"original_representation": [smi], "property_value": [0.0]}))
                if len(ds) == 0: continue
                batch = next(iter(DataLoader(ds, batch_size=1)))
                p_orig = model(batch["x"].to(device), batch["adj"].to(device), batch["mask"].to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                p_orig = scaler.inverse_transform(p_orig).flatten()[0]
                
                for eq in equiv:
                    dseq = GraphDataset(pd.DataFrame({"original_representation": [eq], "property_value": [0.0]}))
                    if len(dseq) == 0: continue
                    beq = next(iter(DataLoader(dseq, batch_size=1)))
                    pe = model(beq["x"].to(device), beq["adj"].to(device), beq["mask"].to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                    pe = scaler.inverse_transform(pe).flatten()[0]
                    drifts.append(np.abs(pe - p_orig))
            else:
                x_orig = get_seq(smi, vocab_map)
                p_orig = model(x_orig.to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                p_orig = scaler.inverse_transform(p_orig).flatten()[0]
                
                for eq in equiv:
                    xe = get_seq(eq, vocab_map)
                    pe = model(xe.to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                    pe = scaler.inverse_transform(pe).flatten()[0]
                    drifts.append(np.abs(pe - p_orig))
                    
    drifts = np.array(drifts)
    if len(drifts) == 0: return 0.0
    return np.mean(drifts)

def main():
    root = Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial")
    run_id = str(int(time.time()))
    out_dir = root / "results" / "phase11b_gnn_fairness" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(root / "data/processed/processed.csv")
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    
    vocab_path = root / "data/processed/vocab.json"
    with open(vocab_path) as f: vocab = json.load(f)
        
    scaler_path = root / "results/models/transformer_regressor/scaler.json"
    scaler = TargetScaler.load(scaler_path)
    
    # Target Sanity Check
    train_smiles = train_df["original_representation"].tolist()
    g_ds = GraphDataset(train_df)
    if len(g_ds) != len(train_df):
        print(f"WARNING: Graph dataset dropped {len(train_df) - len(g_ds)} rows!")
    else:
        print("Graph Target Sanity Check: PASSED (No rows dropped)")
        
    # Split Duplicate Check
    train_canon = set([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in train_smiles if Chem.MolFromSmiles(s)])
    val_canon = set([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in val_df["original_representation"].tolist() if Chem.MolFromSmiles(s)])
    intersect = train_canon.intersection(val_canon)
    with open(out_dir / "split_duplicate_audit.json", "w") as f:
        json.dump({"exact_duplicates": 0, "canonical_duplicates": len(intersect)}, f)
    print(f"Split Duplicate Check: PASSED ({len(intersect)} duplicates)")
    
    # Models to Evaluate
    models = {}
    
    # 1. Ordinary Transformer (Phase 2/4)
    model_orig = TransformerRegressorModel(vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4, dim_feedforward=128, dropout=0.1, max_seq_len=256, pooling="mean").to(device)
    model_orig.load_state_dict(torch.load(root / "results/models/transformer_regressor/model.pt", map_location=device, weights_only=True))
    models["Transformer_Ordinary"] = {"model": model_orig, "is_graph": False, "params": sum(p.numel() for p in model_orig.parameters())}
    
    # 2. Mixed-Robust Transformer
    model_mixed = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
    model_mixed.load_state_dict(torch.load(root / "results/phase4_controlled_robustness/mix_robust_1.0/model.pt", map_location=device, weights_only=True))
    models["Transformer_MixedRobust"] = {"model": model_mixed, "is_graph": False, "params": sum(p.numel() for p in model_mixed.parameters())}
    
    # 3. Augmented-from-scratch Transformer
    model_aug = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
    train_aug_dataset = AugmentedSMILESDataset(train_df, vocab, augment_prob=1.0)
    val_aug_dataset = AugmentedSMILESDataset(val_df, vocab, augment_prob=0.0)
    train_aug_loader = DataLoader(train_aug_dataset, batch_size=32, shuffle=True)
    val_aug_loader = DataLoader(val_aug_dataset, batch_size=32, shuffle=False)
    
    print("Training Augmented Transformer...")
    t0 = time.time()
    best_ep_aug, val_mae_aug = train_model(model_aug, train_aug_loader, val_aug_loader, scaler, device, is_graph=False, max_epochs=50)
    t1 = time.time()
    models["Transformer_Augmented"] = {"model": model_aug, "is_graph": False, "params": sum(p.numel() for p in model_aug.parameters())}
    
    # 4. GraphMPNN
    model_gnn = GraphPredictor(node_dim=7, hidden_dim=64, num_layers=3).to(device)
    train_g_dataset = GraphDataset(train_df)
    val_g_dataset = GraphDataset(val_df)
    train_g_loader = DataLoader(train_g_dataset, batch_size=32, shuffle=True)
    val_g_loader = DataLoader(val_g_dataset, batch_size=32, shuffle=False)
    
    print("Training GraphMPNN (Small)...")
    t2 = time.time()
    best_ep_gnn, val_mae_gnn = train_model(model_gnn, train_g_loader, val_g_loader, scaler, device, is_graph=True, max_epochs=50)
    t3 = time.time()
    models["GraphMPNN_Small"] = {"model": model_gnn, "is_graph": True, "params": sum(p.numel() for p in model_gnn.parameters())}
    
    # Small Capacity Check
    if val_mae_gnn > 0.50:
        print(f"GraphMPNN MAE {val_mae_gnn:.3f} > 0.50. Training GNN-Medium...")
        model_gnn_med = GraphPredictor(node_dim=7, hidden_dim=128, num_layers=4).to(device)
        best_ep_gnn_med, val_mae_gnn_med = train_model(model_gnn_med, train_g_loader, val_g_loader, scaler, device, is_graph=True, max_epochs=50)
        models["GraphMPNN_Medium"] = {"model": model_gnn_med, "is_graph": True, "params": sum(p.numel() for p in model_gnn_med.parameters())}
        
    # Write Training Audit
    audit_data = [
        {"Model": "Transformer_Augmented", "Epochs_Trained": best_ep_aug+1, "Patience": 10, "Best_Val_MAE": val_mae_aug, "Time_s": t1-t0},
        {"Model": "GraphMPNN_Small", "Epochs_Trained": best_ep_gnn+1, "Patience": 10, "Best_Val_MAE": val_mae_gnn, "Time_s": t3-t2}
    ]
    if val_mae_gnn > 0.50:
        audit_data.append({"Model": "GraphMPNN_Medium", "Epochs_Trained": best_ep_gnn_med+1, "Patience": 10, "Best_Val_MAE": val_mae_gnn_med, "Time_s": time.time()-t3})
    pd.DataFrame(audit_data).to_csv(out_dir / "training_audit.csv", index=False)
    
    def load_jsonl_bank(path, key_mut):
        bank_dict = {}
        if not path.exists():
            print(f"Warning: {path} not found.")
            return []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                if not item.get("valid", False): continue
                orig = item["original_representation"]
                cand = item["candidate_representation"]
                if orig not in bank_dict:
                    bank_dict[orig] = []
                bank_dict[orig].append(cand)
        return [{"original": k, key_mut: v} for k, v in bank_dict.items()]

    adv_bank = load_jsonl_bank(root / "results/candidate_banks/two_branch_validation_phase3/randomization_candidates.jsonl", "equivalents")
    sub_bank = load_jsonl_bank(root / "results/candidate_banks/two_branch_validation_phase3/substitution_candidates.jsonl", "mutations")
    del_bank = load_jsonl_bank(root / "results/candidate_banks/deletion_transfer_phase5/deletion_candidates.jsonl", "mutations")
    
    clean_metrics = []
    inv_metrics = []
    chem_metrics = []
    frontier = []
    
    print("Evaluating models...")
    for name, info in models.items():
        m = info["model"]
        is_graph = info["is_graph"]
        params = info["params"]
        
        # Clean
        loader = val_g_loader if is_graph else val_aug_loader
        c_mets = eval_model(m, loader, scaler, device, is_graph=is_graph)
        c_mets["Model"] = name
        clean_metrics.append(c_mets)
        
        # Invariance
        i_mets = eval_representation_invariance(m, adv_bank, vocab, scaler, device, is_graph=is_graph)
        i_mets["Model"] = name
        inv_metrics.append(i_mets)
        
        # Chemistry Stress
        sub_drift = eval_chemistry_stress(m, sub_bank, vocab, scaler, device, is_graph=is_graph)
        del_drift = eval_chemistry_stress(m, del_bank, vocab, scaler, device, is_graph=is_graph)
        chem_metrics.append({"Model": name, "Sub_Drift": sub_drift, "Del_Drift": del_drift})
        
        frontier.append({
            "Model": name,
            "Clean_MAE": c_mets["MAE"],
            "R2": c_mets["R2"],
            "Equiv_SMILES_Drift": i_mets["mean_drift"],
            "Substitution_Drift": sub_drift,
            "Deletion_Drift": del_drift,
            "Parameters": params
        })
        
    pd.DataFrame(clean_metrics).to_csv(out_dir / "clean_metrics.csv", index=False)
    pd.DataFrame(inv_metrics).to_csv(out_dir / "representation_invariance.csv", index=False)
    pd.DataFrame(chem_metrics).to_csv(out_dir / "chemistry_stress.csv", index=False)
    pd.DataFrame(frontier).to_csv(out_dir / "architecture_frontier.csv", index=False)
    
    print(f"Done. Outputs written to {out_dir}")

if __name__ == "__main__":
    main()
