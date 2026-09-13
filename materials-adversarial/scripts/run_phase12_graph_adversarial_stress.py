import json
import sys
import time
import random
from pathlib import Path
import hashlib
from collections import defaultdict
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.data.augmented_dataset import AugmentedSMILESDataset
from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.models.graph_predictor import GraphPredictor
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel

def get_file_hash(path: Path) -> str:
    if not path.exists(): return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def encode_batch(smiles_list, vocab, max_len):
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

class Phase12Evaluator:
    def __init__(self, root: Path, device: torch.device):
        self.root = root
        self.device = device
        self.out_dir = root / "results/phase12_graph_adversarial" / str(int(time.time()))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.root / "data/processed/vocab.json") as f: self.vocab = json.load(f)
        self.scaler = TargetScaler.load(self.root / "results/models/transformer_regressor/scaler.json")
        
        self.load_models()
        self.load_datasets()
        
    def load_models(self):
        print("Loading GraphMPNN Canonical...")
        # Resolve latest Phase11C GraphMPNN dynamically or assume fixed for this run
        # We will use the exact phase 11C path identified earlier
        self.path_gnn = self.root / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
        self.m_gnn = GraphPredictor(node_dim=7, hidden_dim=64, num_layers=3).to(self.device)
        self.m_gnn.load_state_dict(torch.load(self.path_gnn, map_location=self.device, weights_only=True))
        self.m_gnn.eval()
        
        print("Loading Transformer Reference...")
        self.path_tx = self.root / "results/phase4_controlled_robustness/mix_robust_0.1/model.pt"
        self.m_tx = TwoBranchTransformerRegressorModel(vocab_size=len(self.vocab)).to(self.device)
        self.m_tx.load_state_dict(torch.load(self.path_tx, map_location=self.device, weights_only=True))
        self.m_tx.eval()
        
        manifest = {
            "GraphMPNN": {"path": str(self.path_gnn), "hash": get_file_hash(self.path_gnn)},
            "Transformer_MixedRobust": {"path": str(self.path_tx), "hash": get_file_hash(self.path_tx)},
            "Scaler": {"hash": get_file_hash(self.root / "results/models/transformer_regressor/scaler.json")}
        }
        with open(self.out_dir / "checkpoint_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
    def load_datasets(self):
        splits = json.loads((self.root / "data/processed/splits.json").read_text())
        df = pd.read_csv(self.root / "data/processed/processed.csv")
        self.val_df = df.iloc[splits["val"]].copy().reset_index(drop=True)
        
    def predict_gnn(self, smiles_list):
        if not smiles_list: return []
        ds = GraphDataset(pd.DataFrame({"original_representation": smiles_list, "property_value": [0.0]*len(smiles_list)}))
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                y_hat = self.m_gnn(batch["x"].to(self.device), batch["adj"].to(self.device), batch["mask"].to(self.device))
                p = self.scaler.inverse_transform(y_hat.view(-1).cpu().numpy().reshape(-1, 1)).flatten()
                preds.extend(p)
        return preds
        
    def predict_tx(self, smiles_list):
        if not smiles_list: return []
        preds = []
        with torch.no_grad():
            for i in range(0, len(smiles_list), 32):
                batch_smi = smiles_list[i:i+32]
                x_ids, x_masks = encode_batch(batch_smi, self.vocab, 256)
                y_hat = self.m_tx(x_ids.to(self.device), padding_mask=x_masks.to(self.device))
                p = self.scaler.inverse_transform(y_hat.view(-1).cpu().numpy().reshape(-1, 1)).flatten()
                preds.extend(p)
        return preds


    def load_jsonl_bank(self, path, key_mut):
        bank_dict = {}
        if not Path(path).exists(): return []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                if not item.get("valid", False): continue
                orig = item["original_representation"]
                cand = item["candidate_representation"]
                if orig not in bank_dict: bank_dict[orig] = []
                bank_dict[orig].append(cand)
        return [{"original": k, key_mut: v} for k, v in bank_dict.items()]

    def analyze_graph_diff(self, orig_smi, cand_smi):
        orig_mol = Chem.MolFromSmiles(orig_smi)
        cand_mol = Chem.MolFromSmiles(cand_smi)
        if not orig_mol or not cand_mol: return None
        
        orig_canon = Chem.MolToSmiles(orig_mol)
        cand_canon = Chem.MolToSmiles(cand_mol)
        if orig_canon == cand_canon:
            return None # Same graph
            
        nodes_diff = cand_mol.GetNumAtoms() - orig_mol.GetNumAtoms()
        edges_diff = cand_mol.GetNumBonds() - orig_mol.GetNumBonds()
        
        # lightweight features
        orig_degree = np.mean([a.GetDegree() for a in orig_mol.GetAtoms()])
        cand_degree = np.mean([a.GetDegree() for a in cand_mol.GetAtoms()])
        degree_diff = cand_degree - orig_degree
        
        return {
            "node_diff": nodes_diff,
            "edge_diff": edges_diff,
            "degree_diff": float(degree_diff)
        }
        
    def evaluate_fixed_bank(self, path, name):
        bank = self.load_jsonl_bank(path, "mutations")
        print(f"Evaluating {name} on {len(bank)} sources...")
        
        results = []
        for item in bank:
            orig = item["original"]
            cands = item["mutations"]
            
            p_gnn_orig = self.predict_gnn([orig])[0]
            p_tx_orig = self.predict_tx([orig])[0]
            
            for cand in cands:
                g_diff = self.analyze_graph_diff(orig, cand)
                if not g_diff: continue # skip structurally identical
                
                p_gnn_cand = self.predict_gnn([cand])[0]
                p_tx_cand = self.predict_tx([cand])[0]
                
                drift_gnn = abs(p_gnn_cand - p_gnn_orig)
                drift_tx = abs(p_tx_cand - p_tx_orig)
                
                res = {
                    "source": orig,
                    "candidate": cand,
                    "drift_gnn": float(drift_gnn),
                    "drift_tx": float(drift_tx)
                }
                res.update(g_diff)
                results.append(res)
                
        df = pd.DataFrame(results)
        df.to_csv(self.out_dir / f"fixed_{name}_stress.csv", index=False)
        
        summary = {
            "Bank": name,
            "GraphMPNN_Mean": df["drift_gnn"].mean(),
            "Transformer_Mean": df["drift_tx"].mean()
        }
        print(summary)
        return df

    def run(self):
        # 1. Negative control
        print("Executing Negative Control (Equivalent SMILES)...")
        equiv_bank = self.load_jsonl_bank(self.root / "results/candidate_banks/two_branch_validation_phase3/randomization_candidates.jsonl", "equivalents")
        
        neg_drifts = []
        for item in equiv_bank:
            orig = item["original"]
            p_orig = self.predict_gnn([orig])[0]
            for cand in item["equivalents"]:
                if Chem.MolToSmiles(Chem.MolFromSmiles(orig)) != Chem.MolToSmiles(Chem.MolFromSmiles(cand)):
                    continue
                p_cand = self.predict_gnn([cand])[0]
                neg_drifts.append(abs(p_cand - p_orig))
                
        print(f"Negative Control Mean GraphMPNN Drift: {np.mean(neg_drifts)}")
        
        # 2. Fixed Bank Stress
        df_sub = self.evaluate_fixed_bank(self.root / "results/candidate_banks/two_branch_validation_phase3/substitution_candidates.jsonl", "substitution")
        df_del = self.evaluate_fixed_bank(self.root / "results/candidate_banks/deletion_transfer_phase5/deletion_candidates.jsonl", "deletion")
        
        # 3. Correlation (Vulnerability Agreement)
        corr_sub, _ = pearsonr(df_sub["drift_gnn"], df_sub["drift_tx"])
        corr_del, _ = pearsonr(df_del["drift_gnn"], df_del["drift_tx"])
        pd.DataFrame([
            {"Dataset": "Substitution", "Pearson": corr_sub},
            {"Dataset": "Deletion", "Pearson": corr_del}
        ]).to_csv(self.out_dir / "architecture_drift_correlation.csv", index=False)
        
        print("Completed Static Fixed Bank Eval.")
        # TODO: Adaptive Search
        print(f"Done. Outputs written to {self.out_dir}")
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Phase12Evaluator(Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial"), device)
    evaluator.run()

    def mutate_smiles(self, smi):
        tokens = tokenize(smi)
        if not tokens: return smi
        # simple single token substitution (model-independent)
        vocab_chars = [c for c in self.vocab if c not in ['[PAD]', '[BOS]', '[EOS]']]
        idx = random.randint(0, len(tokens)-1)
        # Avoid breaking rings/branches by only mutating standard atoms
        if tokens[idx] in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si']:
            tokens[idx] = random.choice(['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'])
        return "".join(tokens)

    def generate_valid_neighbors(self, source_smi, count=10):
        neighbors = set()
        attempts = 0
        orig_canon = Chem.MolToSmiles(Chem.MolFromSmiles(source_smi))
        while len(neighbors) < count and attempts < count * 10:
            attempts += 1
            cand = self.mutate_smiles(source_smi)
            mol = Chem.MolFromSmiles(cand)
            if mol:
                canon = Chem.MolToSmiles(mol)
                if canon != orig_canon:
                    neighbors.add(canon)
        return list(neighbors)
        
    def adaptive_search(self, target_model_name, source_smi, budget, method="greedy"):
        predict_fn = self.predict_gnn if target_model_name == "GraphMPNN" else self.predict_tx
        
        orig_pred = predict_fn([source_smi])[0]
        best_cand = source_smi
        best_drift = 0
        queries = 0
        
        current_smi = source_smi
        
        # We will use greedy iterative search for simplicity
        while queries < budget:
            neighbors = self.generate_valid_neighbors(current_smi, count=min(5, budget - queries))
            if not neighbors: break
            
            preds = predict_fn(neighbors)
            queries += len(neighbors)
            
            drifts = [abs(p - orig_pred) for p in preds]
            max_idx = np.argmax(drifts)
            
            if drifts[max_idx] > best_drift:
                best_drift = drifts[max_idx]
                best_cand = neighbors[max_idx]
                current_smi = best_cand
                
        return best_cand, best_drift, queries

    def execute_adaptive_search(self):
        print("Executing Adaptive Search...")
        sources = self.val_df["original_representation"].tolist()[:50] # Subset for speed
        budgets = [10, 20, 50]
        
        results = []
        top_examples = []
        
        for q in budgets:
            print(f"Budget Q={q}")
            for smi in sources:
                # Attack GraphMPNN
                gnn_cand, gnn_drift, gnn_q = self.adaptive_search("GraphMPNN", smi, q)
                # Cross-eval on Transformer
                tx_cross_pred = self.predict_tx([gnn_cand])[0]
                tx_orig_pred = self.predict_tx([smi])[0]
                tx_cross_drift = abs(tx_cross_pred - tx_orig_pred)
                
                results.append({
                    "Source": smi, "Candidate": gnn_cand, 
                    "TargetModel": "GraphMPNN", "Budget": q, 
                    "TargetDrift": float(gnn_drift), "TransferDrift": float(tx_cross_drift)
                })
                
                if gnn_drift > 0.5:
                    top_examples.append({
                        "source": smi, "candidate": gnn_cand, "target": "GraphMPNN",
                        "drift": float(gnn_drift), "transfer_drift": float(tx_cross_drift),
                        "budget": q
                    })
                
                # Attack Transformer
                tx_cand, tx_drift, tx_q = self.adaptive_search("Transformer", smi, q)
                # Cross-eval on GraphMPNN
                gnn_cross_pred = self.predict_gnn([tx_cand])[0]
                gnn_orig_pred = self.predict_gnn([smi])[0]
                gnn_cross_drift = abs(gnn_cross_pred - gnn_orig_pred)
                
                results.append({
                    "Source": smi, "Candidate": tx_cand, 
                    "TargetModel": "Transformer", "Budget": q, 
                    "TargetDrift": float(tx_drift), "TransferDrift": float(gnn_cross_drift)
                })
                
        df = pd.DataFrame(results)
        df.to_csv(self.out_dir / "adaptive_search_summary.csv", index=False)
        
        with open(self.out_dir / "top_adversarial_examples.jsonl", "w") as f:
            for ex in top_examples:
                f.write(json.dumps(ex) + "\n")

# Patch run method to include adaptive search

Phase12Evaluator.mutate_smiles = mutate_smiles
Phase12Evaluator.generate_valid_neighbors = generate_valid_neighbors
Phase12Evaluator.adaptive_search = adaptive_search
Phase12Evaluator.execute_adaptive_search = execute_adaptive_search

old_run = Phase12Evaluator.run

def new_run(self):
    old_run(self)
    self.execute_adaptive_search()

Phase12Evaluator.run = new_run

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Phase12Evaluator(Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial"), device)
    evaluator.run()
