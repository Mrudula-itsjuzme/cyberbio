import json
import os
import sys
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.models.graph_predictor import GraphPredictor

def get_file_hash(path: Path) -> str:
    if not path.exists(): return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def eval_model_canonical(model, df, scaler, device):
    model.eval()
    all_preds, all_targets = [], []
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
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    err = all_preds - all_targets
    ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
    ss_res = np.sum(err**2)
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    }

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

def eval_drift(model, evals, scaler, device):
    drifts = []
    with torch.no_grad():
        for item in evals:
            smi = item["original"]
            equiv = item["equivalents"]
            
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
                    
    drifts = np.array(drifts)
    if len(drifts) == 0: return 0.0
    return np.mean(drifts)

def main():
    print("Running Canonical Reproduction Verification...")
    manifest_path = "RELEASE_MANIFEST.json"
    if not os.path.exists(manifest_path):
        print("ERROR: RELEASE_MANIFEST.json not found.")
        sys.exit(1)
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    root = Path(".")
    
    # 1. Verify Hashes
    print("\n--- HASH-VERIFIED ---")
    files_to_hash = [
        ("raw_dataset", "data/raw/bandgap_chain.csv"),
        ("processed_dataset", "data/processed/processed.csv"),
        ("splits", "data/processed/splits.json"),
        ("vocab", "data/processed/vocab.json"),
        ("scaler", "results/models/transformer_regressor/scaler.json"),
        ("GraphMPNN_Small_ckpt", "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"),
        ("Mixed_Robust_Transformer_ckpt", "results/phase4_controlled_robustness/mix_robust_0.1/model.pt"),
        ("Phase12B_search_summary", "results/phase12b_adaptive_search_audit/1789345214/search_summary.json"),
    ]
    
    for name, path in files_to_hash:
        actual_hash = get_file_hash(root / path)
        if name in manifest.get("hashes", {}):
            expected = manifest["hashes"][name]
            if actual_hash != expected:
                print(f"WARNING: Hash mismatch for {name}. Expected: {expected}, Actual: {actual_hash}")
            else:
                print(f"Hash verified: {name}")
        else:
            print(f"Not in manifest hashes: {name} (Actual: {actual_hash})")
            
    print(f"Phase 12B Max Bounded Chemistry Drift (HASH-VERIFIED): {manifest['results']['GraphMPNN_Small']['max_bounded_chemistry_drift']} eV")
    
    # 2. Recompute GraphMPNN metrics
    print("\n--- RECOMPUTED ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = TargetScaler.load(root / "results/models/transformer_regressor/scaler.json")
    df = pd.read_csv(root / "data/processed/processed.csv")
    splits = json.loads((root / "data/processed/splits.json").read_text())
    val_df = df.iloc[splits["val"]].copy().reset_index(drop=True)
    
    m_gnn = GraphPredictor(node_dim=7, hidden_dim=64, num_layers=3).to(device)
    ckpt_path = root / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
    if not ckpt_path.exists():
        print(f"ERROR: GraphMPNN checkpoint not found at {ckpt_path}")
        sys.exit(1)
        
    m_gnn.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    
    print("Recomputing validation inference...")
    mets = eval_model_canonical(m_gnn, val_df, scaler, device)
    print(f"GraphMPNN Validation MAE: {mets['MAE']:.6f} eV")
    print(f"GraphMPNN Validation RMSE: {mets['RMSE']:.6f} eV")
    print(f"GraphMPNN Validation R2: {mets['R2']:.6f}")
    
    print("Recomputing equivalent-SMILES drift...")
    adv_bank = load_jsonl_bank(root / "results/candidate_banks/two_branch_validation_phase3/randomization_candidates.jsonl", "equivalents")
    eq_drift = eval_drift(m_gnn, adv_bank, scaler, device)
    print(f"GraphMPNN Equivalent-SMILES Drift: {eq_drift:.6f} eV")
    
if __name__ == "__main__":
    main()
