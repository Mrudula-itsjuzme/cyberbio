import os
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rdkit import Chem

from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel
from materials_adv.models.graph_predictor import GraphPredictor
from materials_adv.data.augmented_dataset import AugmentedSMILESDataset
from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize

def eval_model(model, dataloader, scaler, device, is_graph=False):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in dataloader:
            if is_graph:
                x = batch["x"].to(device)
                adj = batch["adj"].to(device)
                mask = batch["mask"].to(device)
                out = model(x, adj, mask).cpu().numpy().flatten()
            else:
                x = batch["x"].to(device)
                out = model(x).cpu().numpy().flatten()
                
            preds.extend(scaler.inverse_transform(out))
            trues.extend(batch["target"].numpy().flatten())
    
    preds = np.array(preds)
    trues = np.array(trues)
    
    mae = np.mean(np.abs(preds - trues))
    ss_res = np.sum((trues - preds)**2)
    ss_tot = np.sum((trues - np.mean(trues))**2)
    r2 = 1 - ss_res/(ss_tot + 1e-8)
    
    return {"val_mae": float(mae), "val_r2": float(r2), "std": float(np.std(preds)), "range": float(np.ptp(preds))}

def representation_robustness(model, smis, vocab, scaler, device, is_graph=False):
    model.eval()
    preds_orig = []
    preds_rand = []
    
    with torch.no_grad():
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue
            smi_rand = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            
            if is_graph:
                # Graph dataset logic for single SMILES
                from materials_adv.data.graph_dataset import get_node_features, get_edge_features
                
                def get_graph(s, max_nodes=256):
                    m = Chem.MolFromSmiles(s)
                    n_atoms = m.GetNumAtoms()
                    x = np.zeros((max_nodes, 7), dtype=np.float32)
                    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
                    actual_nodes = min(n_atoms, max_nodes)
                    for i, atom in enumerate(m.GetAtoms()):
                        if i >= actual_nodes: break
                        x[i] = get_node_features(atom)
                    for bond in m.GetBonds():
                        i = bond.GetBeginAtomIdx()
                        j = bond.GetEndAtomIdx()
                        if i < actual_nodes and j < actual_nodes:
                            adj[i, j] = 1.0; adj[j, i] = 1.0
                    for i in range(actual_nodes): adj[i, i] = 1.0
                    mask = np.zeros((max_nodes,), dtype=bool)
                    mask[:actual_nodes] = True
                    return torch.tensor(x).unsqueeze(0), torch.tensor(adj).unsqueeze(0), torch.tensor(mask).unsqueeze(0)

                x1, a1, m1 = get_graph(smi)
                x2, a2, m2 = get_graph(smi_rand)
                p1 = model(x1.to(device), a1.to(device), m1.to(device)).cpu().numpy().flatten()
                p2 = model(x2.to(device), a2.to(device), m2.to(device)).cpu().numpy().flatten()
            else:
                vocab_map = {t: i for i, t in enumerate(vocab)}
                def get_seq(s, max_seq_len=256):
                    ids = [vocab_map.get(t, vocab_map.get("<unk>", 1)) + 1 for t in tokenize(s)]
                    ids = ids[:max_seq_len]
                    padded = ids + [0] * (max_seq_len - len(ids))
                    return torch.tensor(padded, dtype=torch.long).unsqueeze(0)

                t1 = get_seq(smi)
                t2 = get_seq(smi_rand)
                p1 = model(t1.to(device)).cpu().numpy().flatten()
                p2 = model(t2.to(device)).cpu().numpy().flatten()
                
            preds_orig.extend(scaler.inverse_transform(p1))
            preds_rand.extend(scaler.inverse_transform(p2))
            
    drifts = np.abs(np.array(preds_orig) - np.array(preds_rand))
    return float(np.mean(drifts)), float(np.median(drifts)), float(np.percentile(drifts, 95))

def train_model(model, train_loader, val_loader, scaler, device, is_graph=False, max_epochs=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    best_state = None
    start_time = time.time()
    
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
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
            
        metrics = eval_model(model, val_loader, scaler, device, is_graph=is_graph)
        if metrics["val_mae"] < best_val_mae:
            best_val_mae = metrics["val_mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    return time.time() - start_time

def main():
    root = Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = str(int(time.time()))
    out_dir = root / "results" / "phase11_architecture_comparison" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(root / "data/processed/processed.csv")
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    
    vocab_path = root / "data/processed/vocab.json"
    with open(vocab_path) as f: vocab = json.load(f)
        
    scaler_path = root / "results/models/transformer_regressor/scaler.json"
    scaler = TargetScaler.load(scaler_path)
    
    # MODEL A
    model_a = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
    d0_ckpt = root / "results/phase4_controlled_robustness/mix_robust_0.1/model.pt"
    model_a.load_state_dict(torch.load(d0_ckpt, map_location=device, weights_only=True))
    
    # MODEL B
    model_b = TwoBranchTransformerRegressorModel(vocab_size=len(vocab)).to(device)
    train_aug_dataset = AugmentedSMILESDataset(train_df, vocab, augment_prob=0.5)
    val_seq_dataset = AugmentedSMILESDataset(val_df, vocab, augment_prob=0.0)
    train_aug_loader = DataLoader(train_aug_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_seq_loader = DataLoader(val_seq_dataset, batch_size=64, shuffle=False)
    print("Training Model B...")
    train_model(model_b, train_aug_loader, val_seq_loader, scaler, device, is_graph=False, max_epochs=10)
    
    # MODEL C
    model_c = GraphPredictor().to(device)
    train_graph_dataset = GraphDataset(train_df)
    val_graph_dataset = GraphDataset(val_df)
    train_graph_loader = DataLoader(train_graph_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_graph_loader = DataLoader(val_graph_dataset, batch_size=64, shuffle=False)
    print("Training Model C...")
    train_model(model_c, train_graph_loader, val_graph_loader, scaler, device, is_graph=True, max_epochs=10)
    
    metrics_a = eval_model(model_a, val_seq_loader, scaler, device, is_graph=False)
    metrics_b = eval_model(model_b, val_seq_loader, scaler, device, is_graph=False)
    metrics_c = eval_model(model_c, val_graph_loader, scaler, device, is_graph=True)
    
    pd.DataFrame([{"model": "A_BaselineSeq", **metrics_a}, {"model": "B_AugmentedSeq", **metrics_b}, {"model": "C_GraphMPNN", **metrics_c}]).to_csv(out_dir / "clean_metrics.csv", index=False)
    
    # Representation Invariance Eval
    eval_smiles = val_df["original_representation"].tolist()[:100]
    drift_a = representation_robustness(model_a, eval_smiles, vocab, scaler, device, is_graph=False)
    drift_b = representation_robustness(model_b, eval_smiles, vocab, scaler, device, is_graph=False)
    drift_c = representation_robustness(model_c, eval_smiles, vocab, scaler, device, is_graph=True)
    
    pd.DataFrame([
        {"model": "A_BaselineSeq", "mean_drift": drift_a[0], "median_drift": drift_a[1], "p95": drift_a[2]},
        {"model": "B_AugmentedSeq", "mean_drift": drift_b[0], "median_drift": drift_b[1], "p95": drift_b[2]},
        {"model": "C_GraphMPNN", "mean_drift": drift_c[0], "median_drift": drift_c[1], "p95": drift_c[2]}
    ]).to_csv(out_dir / "representation_bank_metrics.csv", index=False)

    print(f"Results written to {out_dir}")

if __name__ == "__main__":
    main()
