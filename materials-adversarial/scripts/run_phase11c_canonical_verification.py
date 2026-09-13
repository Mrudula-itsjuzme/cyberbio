import json
import sys
import time
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rdkit import Chem

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.models.transformer import TransformerRegressorModel
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel
from materials_adv.data.augmented_dataset import AugmentedSMILESDataset
from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.models.graph_predictor import GraphPredictor
from materials_adv.utils.config import load_config

def get_file_hash(path: Path) -> str:
    if not path.exists(): return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def encode_batch(smiles_list, vocab, max_len):
    # This matches the historical eval_phase2b_clean.py encoding identically
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}
    ids_list, masks_list = [], []
    for s in smiles_list:
        ids = [char2idx.get(t, 0) for t in tokenize(s)]
        mask = [False] * len(ids)
        while len(ids) < max_len:
            ids.append(0)
            mask.append(True)
        ids_list.append(ids[:max_len])
        masks_list.append(mask[:max_len])
    return torch.tensor(ids_list, dtype=torch.long), torch.tensor(masks_list, dtype=torch.bool)

def eval_model_canonical(model, df, vocab, scaler, device, is_graph=False):
    model.eval()
    all_preds, all_targets = [], []
    
    if is_graph:
        ds = GraphDataset(df)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                adj = batch["adj"].to(device)
                mask = batch["mask"].to(device)
                y_hat = model(x, adj, mask).view(-1).cpu().numpy().reshape(-1, 1)
                y_pred = scaler.inverse_transform(y_hat).flatten()
                all_preds.extend(y_pred)
                all_targets.extend(batch["target"].numpy().flatten())
    else:
        # Batch inference exactly reproducing Phase 2B masking behavior
        max_len = 256
        batch_size = 32
        smiles = df["original_representation"].tolist()
        targets = df["property_value"].astype(float).tolist()
        
        with torch.no_grad():
            for i in range(0, len(smiles), batch_size):
                smi_batch = smiles[i:i+batch_size]
                y_batch = targets[i:i+batch_size]
                x_ids, x_masks = encode_batch(smi_batch, vocab, max_len)
                
                out = model(x_ids.to(device), padding_mask=x_masks.to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                y_pred = scaler.inverse_transform(out).flatten()
                all_preds.extend(y_pred)
                all_targets.extend(y_batch)
                
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    err = all_preds - all_targets
    ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
    ss_res = np.sum(err**2)
    
    if not np.all(np.isfinite(all_preds)):
        raise ValueError("Model predicted non-finite values.")
        
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
        "Std": np.std(all_preds),
        "Range": np.max(all_preds) - np.min(all_preds),
        "Predictions": all_preds,
        "Targets": all_targets
    }

def train_retrain_model(model, train_df, val_df, vocab, scaler, device, is_graph=False, max_epochs=50, save_path=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    
    if is_graph:
        train_ds = GraphDataset(train_df)
        val_ds = GraphDataset(val_df)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    else:
        train_ds = AugmentedSMILESDataset(train_df, vocab, augment_prob=1.0)
        val_ds = AugmentedSMILESDataset(val_df, vocab, augment_prob=0.0)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
    best_val_mae = float('inf')
    best_state = None
    patience = 10
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if is_graph:
                y_hat = model(batch["x"].to(device), batch["adj"].to(device), batch["mask"].to(device)).view(-1)
            else:
                y_hat = model(batch["x"].to(device), padding_mask=(batch["x"]==0).to(device)).view(-1)
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
                    y_hat = model(batch["x"].to(device), batch["adj"].to(device), batch["mask"].to(device)).view(-1)
                else:
                    y_hat = model(batch["x"].to(device), padding_mask=(batch["x"]==0).to(device)).view(-1)
                y_hat_np = y_hat.cpu().numpy().reshape(-1, 1)
                y_pred = scaler.inverse_transform(y_hat_np).flatten()
                y_true = batch["target"].cpu().numpy().flatten()
                val_losses.append(np.abs(y_pred - y_true).mean())
        val_mae = np.mean(val_losses)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience: break
            
    model.load_state_dict(best_state)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)
        
    return model

def load_jsonl_bank(path, key_mut):
    bank_dict = {}
    if not path.exists(): return []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if not item.get("valid", False): continue
            orig = item["original_representation"]
            cand = item["candidate_representation"]
            if orig not in bank_dict: bank_dict[orig] = []
            bank_dict[orig].append(cand)
    return [{"original": k, key_mut: v} for k, v in bank_dict.items()]

def eval_drift(model, evals, vocab, scaler, device, key_mut, is_graph=False):
    drifts = []
    
    with torch.no_grad():
        for item in evals:
            smi = item["original"]
            equiv = item[key_mut]
            
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
                x_orig, m_orig = encode_batch([smi], vocab, 256)
                p_orig = model(x_orig.to(device), padding_mask=m_orig.to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                p_orig = scaler.inverse_transform(p_orig).flatten()[0]
                
                for eq in equiv:
                    xe, me = encode_batch([eq], vocab, 256)
                    pe = model(xe.to(device), padding_mask=me.to(device)).view(-1).cpu().numpy().reshape(-1, 1)
                    pe = scaler.inverse_transform(pe).flatten()[0]
                    drifts.append(np.abs(pe - p_orig))
                    
    drifts = np.array(drifts)
    if len(drifts) == 0: return {"mean_drift": 0, "median_drift": 0, "p95": 0, "max": 0}
    return {
        "mean_drift": np.mean(drifts),
        "median_drift": np.median(drifts),
        "p95": np.percentile(drifts, 95),
        "max": np.max(drifts)
    }

def main():
    root = Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial")
    run_id = str(int(time.time()))
    out_dir = root / "results" / "phase11c_canonical_verification" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(root / "data/processed/processed.csv")
    splits = json.loads((root / "data/processed/splits.json").read_text())
    train_df = df.iloc[splits["train"]].copy().reset_index(drop=True)
    val_df = df.iloc[splits["val"]].copy().reset_index(drop=True)
    
    with open(root / "data/processed/vocab.json") as f: vocab = json.load(f)
    scaler = TargetScaler.load(root / "results/models/transformer_regressor/scaler.json")
    
    # 1. Gather Models
    models = {}
    
    # A. Ordinary Transformer
    path_orig = root / "results/models/transformer_regressor/model.pt"
    m_orig = TransformerRegressorModel(vocab_size=len(vocab), d_model=64, n_layers=2, n_heads=4, dim_feedforward=128, dropout=0.1, max_seq_len=256, pooling="mean").to(device)
    m_orig.load_state_dict(torch.load(path_orig, map_location=device, weights_only=True))
    models["Transformer_Ordinary"] = {"model": m_orig, "is_graph": False, "path": path_orig}
    
    # B. Architecture Control
    path_ctrl = root / "results/models/specialized_control/model.pt"
    m_ctrl = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
    m_ctrl.load_state_dict(torch.load(path_ctrl, map_location=device, weights_only=True))
    models["Transformer_ArchControl"] = {"model": m_ctrl, "is_graph": False, "path": path_ctrl}
    
    # C. Mixed-Robust Transformer
    path_mix = root / "results/phase4_controlled_robustness/mix_robust_0.1/model.pt"
    m_mix = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
    m_mix.load_state_dict(torch.load(path_mix, map_location=device, weights_only=True))
    models["Transformer_MixedRobust"] = {"model": m_mix, "is_graph": False, "path": path_mix}
    
    # Since Phase 11B models were not saved to disk, train them correctly now.
    path_aug = out_dir / "models" / "transformer_augmented" / "model.pt"
    if not path_aug.exists():
        print("Replicating Phase 11C Augmented Transformer (not persisted historically)...")
        m_aug = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
        m_aug = train_retrain_model(m_aug, train_df, val_df, vocab, scaler, device, is_graph=False, max_epochs=50, save_path=path_aug)
    else:
        m_aug = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
        m_aug.load_state_dict(torch.load(path_aug, map_location=device, weights_only=True))
    models["Transformer_Augmented"] = {"model": m_aug, "is_graph": False, "path": path_aug}
        
    path_gnn = out_dir / "models" / "graph_mpnn_small" / "model.pt"
    if not path_gnn.exists():
        print("Replicating Phase 11C GraphMPNN Small (not persisted historically)...")
        m_gnn = GraphPredictor(node_dim=7, hidden_dim=64, num_layers=3).to(device)
        m_gnn = train_retrain_model(m_gnn, train_df, val_df, vocab, scaler, device, is_graph=True, max_epochs=50, save_path=path_gnn)
    else:
        m_gnn = GraphPredictor(node_dim=7, hidden_dim=64, num_layers=3).to(device)
        m_gnn.load_state_dict(torch.load(path_gnn, map_location=device, weights_only=True))
    models["GraphMPNN_Small"] = {"model": m_gnn, "is_graph": True, "path": path_gnn}
    
    # Save checkpoint manifest
    manifest = {}
    for name, m_info in models.items():
        manifest[name] = {
            "path": str(m_info["path"]),
            "hash": get_file_hash(m_info["path"]),
            "params": sum(p.numel() for p in m_info["model"].parameters())
        }
    with open(out_dir / "checkpoint_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Executing Canonical Evaluation...")
    clean_metrics = []
    cross_preds = {"source_id": val_df["polymer_id"].tolist(), "target_eV": val_df["property_value"].tolist()}
    
    for name, m_info in models.items():
        m = m_info["model"]
        is_graph = m_info["is_graph"]
        mets = eval_model_canonical(m, val_df, vocab, scaler, device, is_graph=is_graph)
        mets["Model"] = name
        
        # Verify predictions match validation length perfectly
        assert len(mets["Predictions"]) == len(val_df), f"{name} predicted length mismatch!"
        cross_preds[name + "_pred"] = mets["Predictions"]
        
        # Historical assertion
        if name == "Transformer_Ordinary":
            assert abs(mets["MAE"] - 0.486) < 0.01, f"Ordinary MAE reproduced {mets['MAE']}, expected ~0.486!"
        if name == "Transformer_ArchControl":
            assert abs(mets["MAE"] - 0.461) < 0.01, f"Control MAE reproduced {mets['MAE']}, expected ~0.461!"
        if name == "Transformer_MixedRobust":
            assert abs(mets["MAE"] - 0.441) < 0.01, f"Mixed MAE reproduced {mets['MAE']}, expected ~0.441!"
            
        clean_metrics.append({k:v for k,v in mets.items() if k not in ["Predictions", "Targets"]})
        
    pd.DataFrame(clean_metrics).to_csv(out_dir / "canonical_clean_metrics.csv", index=False)
    
    cross_df = pd.DataFrame(cross_preds)
    cross_df.head(20).to_csv(out_dir / "cross_model_predictions.csv", index=False)
    
    # Graph Duplicate Audit
    print("Auditing Graph Duplicates...")
    train_canon = set([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in train_df["original_representation"].tolist() if Chem.MolFromSmiles(s)])
    val_canon = set([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in val_df["original_representation"].tolist() if Chem.MolFromSmiles(s)])
    intersect = train_canon.intersection(val_canon)
    with open(out_dir / "split_duplicate_audit.json", "w") as f:
        json.dump({"canonical_duplicates_leaked": len(intersect)}, f, indent=2)
        
    print("Evaluating Fairness & Stress...")
    adv_bank = load_jsonl_bank(root / "results/candidate_banks/two_branch_validation_phase3/randomization_candidates.jsonl", "equivalents")
    sub_bank = load_jsonl_bank(root / "results/candidate_banks/two_branch_validation_phase3/substitution_candidates.jsonl", "mutations")
    del_bank = load_jsonl_bank(root / "results/candidate_banks/deletion_transfer_phase5/deletion_candidates.jsonl", "mutations")
    
    inv_metrics = []
    stress_metrics = []
    
    for name, m_info in models.items():
        m = m_info["model"]
        is_graph = m_info["is_graph"]
        
        im = eval_drift(m, adv_bank, vocab, scaler, device, "equivalents", is_graph)
        im["Model"] = name
        inv_metrics.append(im)
        
        sm = eval_drift(m, sub_bank, vocab, scaler, device, "mutations", is_graph)
        dm = eval_drift(m, del_bank, vocab, scaler, device, "mutations", is_graph)
        stress_metrics.append({
            "Model": name,
            "Sub_Mean": sm["mean_drift"],
            "Del_Mean": dm["mean_drift"]
        })
        
    pd.DataFrame(inv_metrics).to_csv(out_dir / "representation_invariance.csv", index=False)
    pd.DataFrame(stress_metrics).to_csv(out_dir / "chemistry_stress.csv", index=False)
    
    # Final Table
    frontier = []
    for c_met, i_met, s_met in zip(clean_metrics, inv_metrics, stress_metrics):
        name = c_met["Model"]
        frontier.append({
            "Model": name,
            "Clean_MAE": c_met["MAE"],
            "RMSE": c_met["RMSE"],
            "R2": c_met["R2"],
            "Prediction_Std": c_met["Std"],
            "Equiv_SMILES_Drift": i_met["mean_drift"],
            "Substitution_Stress_Drift": s_met["Sub_Mean"],
            "Deletion_Stress_Drift": s_met["Del_Mean"],
            "Parameters": manifest[name]["params"]
        })
    pd.DataFrame(frontier).to_csv(out_dir / "architecture_frontier.csv", index=False)
    print(f"Done. Outputs written to {out_dir}")

if __name__ == "__main__":
    main()
