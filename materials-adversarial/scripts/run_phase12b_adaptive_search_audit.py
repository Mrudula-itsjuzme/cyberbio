import json
import sys
import time
import random
from pathlib import Path
import hashlib
from collections import defaultdict
from dataclasses import dataclass, asdict
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from materials_adv.data.scaler import TargetScaler
from materials_adv.data.tokenizer import tokenize
from materials_adv.data.graph_dataset import GraphDataset
from materials_adv.models.graph_predictor import GraphPredictor
from materials_adv.models.specialized_transformer import TwoBranchTransformerRegressorModel

@dataclass
class CandidateState:
    source_id: int
    source_smiles: str
    target_model: str
    search_method: str
    seed: int
    query_budget: int
    query_index: int
    current_candidate: str
    proposed_candidate: str
    edit_list: list
    edit_count: int
    validity: str # 'valid', 'invalid_rdkit', 'invalid_graph', 'invalid_duplicate', 'invalid_budget'
    duplicate: bool
    prediction: float
    drift_from_original: float
    accepted: bool
    best_so_far_drift: float

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

class Phase12bEvaluator:
    def __init__(self, root: Path, device: torch.device):
        self.root = root
        self.device = device
        self.run_id = str(int(time.time()))
        self.out_dir = root / "results/phase12b_adaptive_search" / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.root / "data/processed/vocab.json") as f: self.vocab = json.load(f)
        self.scaler = TargetScaler.load(self.root / "results/models/transformer_regressor/scaler.json")
        self.vocab_chars = [c for c in self.vocab if c not in ['[PAD]', '[BOS]', '[EOS]']]
        
        self.load_models()
        self.load_datasets()
        self.trajectories = []
        self.validity_stats = defaultdict(int)

    def load_models(self):
        print("Loading GraphMPNN Canonical...")
        self.path_gnn = self.root / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
        self.m_gnn = GraphPredictor(node_dim=7, hidden_dim=64, num_layers=3).to(self.device)
        self.m_gnn.load_state_dict(torch.load(self.path_gnn, map_location=self.device, weights_only=True))
        self.m_gnn.eval()
        
        print("Loading Transformer Reference...")
        self.path_tx = self.root / "results/phase4_controlled_robustness/mix_robust_0.1/model.pt"
        self.m_tx = TwoBranchTransformerRegressorModel(vocab_size=len(self.vocab)).to(self.device)
        self.m_tx.load_state_dict(torch.load(self.path_tx, map_location=self.device, weights_only=True))
        self.m_tx.eval()
        
    def load_datasets(self):
        splits = json.loads((self.root / "data/processed/splits.json").read_text())
        df = pd.read_csv(self.root / "data/processed/processed.csv")
        # 50 fixed sources for canonical comparison
        self.val_df = df.iloc[splits["val"]].copy().reset_index(drop=True).head(50)
        
        manifest = {
            "GraphMPNN": {"path": str(self.path_gnn), "hash": get_file_hash(self.path_gnn)},
            "Transformer_MixedRobust": {"path": str(self.path_tx), "hash": get_file_hash(self.path_tx)},
            "Scaler": {"hash": get_file_hash(self.root / "results/models/transformer_regressor/scaler.json")},
            "Sources": [{"id": i, "smiles": row["original_representation"]} for i, row in self.val_df.iterrows()]
        }
        with open(self.out_dir / "source_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

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

    def mutate_smiles(self, smi, rng):
        tokens = tokenize(smi)
        if not tokens: return smi
        idx = rng.randint(0, len(tokens)-1)
        if tokens[idx] in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si']:
            tokens[idx] = rng.choice(['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I'])
        return "".join(tokens)

    def compute_edits(self, original, proposed):
        t_orig = tokenize(original)
        t_prop = tokenize(proposed)
        
        # Pure Python Levenshtein
        size_x = len(t_orig) + 1
        size_y = len(t_prop) + 1
        matrix = np.zeros ((size_x, size_y))
        for x in range(size_x): matrix [x, 0] = x
        for y in range(size_y): matrix [0, y] = y
        for x in range(1, size_x):
            for y in range(1, size_y):
                if t_orig[x-1] == t_prop[y-1]:
                    matrix [x,y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1],
                        matrix[x, y-1] + 1
                    )
                else:
                    matrix [x,y] = min(
                        matrix[x-1,y] + 1,
                        matrix[x-1,y-1] + 1,
                        matrix[x,y-1] + 1
                    )
        dist = int(matrix[size_x - 1, size_y - 1])
        return dist, [] # Abstracted edit list for brevity, distance is absolute truth

    def check_validity(self, original, proposed):
        dist, _ = self.compute_edits(original, proposed)
        if dist > 3:
            return "invalid_budget"
            
        mol = Chem.MolFromSmiles(proposed)
        if not mol:
            return "invalid_rdkit"
            
        canon_prop = Chem.MolToSmiles(mol)
        canon_orig = Chem.MolToSmiles(Chem.MolFromSmiles(original))
        if canon_prop == canon_orig:
            return "invalid_duplicate" # Chemically identical graph
            
        try:
            # Check GraphMPNN conversion
            ds = GraphDataset(pd.DataFrame({"original_representation": [proposed], "property_value": [0.0]}))
            _ = ds[0]
        except Exception:
            return "invalid_graph"
            
        return "valid"

    def generate_valid_neighbor(self, original, current, rng, max_attempts=100):
        for _ in range(max_attempts):
            cand = self.mutate_smiles(current, rng)
            val = self.check_validity(original, cand)
            self.validity_stats[val] += 1
            if val == "valid":
                return cand
        return None

    def search_random(self, model_name, source_smi, source_id, budget, seed):
        rng = random.Random(seed)
        predict_fn = self.predict_gnn if model_name == "GraphMPNN" else self.predict_tx
        orig_pred = predict_fn([source_smi])[0]
        
        best_drift = 0
        best_cand = source_smi
        queries = 0
        
        while queries < budget:
            cand = self.generate_valid_neighbor(source_smi, source_smi, rng)
            if not cand: break # Stuck
            
            p = predict_fn([cand])[0]
            queries += 1
            drift = abs(p - orig_pred)
            
            dist, edits = self.compute_edits(source_smi, cand)
            
            if drift > best_drift:
                best_drift = drift
                best_cand = cand
                
            state = CandidateState(
                source_id=source_id, source_smiles=source_smi, target_model=model_name,
                search_method="random", seed=seed, query_budget=budget, query_index=queries,
                current_candidate=source_smi, proposed_candidate=cand, edit_list=edits,
                edit_count=dist, validity="valid", duplicate=False, prediction=float(p),
                drift_from_original=float(drift), accepted=True, best_so_far_drift=float(best_drift)
            )
            self.trajectories.append(asdict(state))
            
        return best_cand, best_drift, queries

    def search_greedy(self, model_name, source_smi, source_id, budget, seed):
        rng = random.Random(seed)
        predict_fn = self.predict_gnn if model_name == "GraphMPNN" else self.predict_tx
        orig_pred = predict_fn([source_smi])[0]
        
        best_drift = 0
        best_cand = source_smi
        current_smi = source_smi
        queries = 0
        
        while queries < budget:
            # Generate 5 neighbors
            neighbors = []
            for _ in range(5):
                cand = self.generate_valid_neighbor(source_smi, current_smi, rng)
                if cand: neighbors.append(cand)
            if not neighbors: break
            
            # evaluate
            batch = neighbors[:budget - queries]
            if not batch: break
            
            preds = predict_fn(batch)
            queries += len(batch)
            
            drifts = [abs(p - orig_pred) for p in preds]
            max_idx = np.argmax(drifts)
            
            for i, (cand, p, d) in enumerate(zip(batch, preds, drifts)):
                dist, edits = self.compute_edits(source_smi, cand)
                accepted = (i == max_idx) and (d > best_drift)
                
                if accepted:
                    best_drift = d
                    best_cand = cand
                    current_smi = cand
                    
                state = CandidateState(
                    source_id=source_id, source_smiles=source_smi, target_model=model_name,
                    search_method="greedy", seed=seed, query_budget=budget, query_index=queries - len(batch) + i + 1,
                    current_candidate=current_smi if not accepted else cand, proposed_candidate=cand, edit_list=edits,
                    edit_count=dist, validity="valid", duplicate=False, prediction=float(p),
                    drift_from_original=float(d), accepted=bool(accepted), best_so_far_drift=float(best_drift)
                )
                self.trajectories.append(asdict(state))
                
        return best_cand, best_drift, queries

    def search_metropolis(self, model_name, source_smi, source_id, budget, seed):
        rng = random.Random(seed)
        predict_fn = self.predict_gnn if model_name == "GraphMPNN" else self.predict_tx
        orig_pred = predict_fn([source_smi])[0]
        
        best_drift = 0
        best_cand = source_smi
        current_smi = source_smi
        current_drift = 0
        queries = 0
        T = 0.1 # Fixed temperature
        
        while queries < budget:
            cand = self.generate_valid_neighbor(source_smi, current_smi, rng)
            if not cand: break
            
            p = predict_fn([cand])[0]
            queries += 1
            drift = abs(p - orig_pred)
            dist, edits = self.compute_edits(source_smi, cand)
            
            if drift > best_drift:
                best_drift = drift
                best_cand = cand
                
            delta = drift - current_drift
            accepted = False
            if delta > 0:
                accepted = True
            else:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    accepted = True
                    
            if accepted:
                current_smi = cand
                current_drift = drift
                
            state = CandidateState(
                source_id=source_id, source_smiles=source_smi, target_model=model_name,
                search_method="metropolis", seed=seed, query_budget=budget, query_index=queries,
                current_candidate=current_smi, proposed_candidate=cand, edit_list=edits,
                edit_count=dist, validity="valid", duplicate=False, prediction=float(p),
                drift_from_original=float(drift), accepted=accepted, best_so_far_drift=float(best_drift)
            )
            self.trajectories.append(asdict(state))
            
        return best_cand, best_drift, queries

    def run_adaptive_audits(self):
        print("Executing Adaptive Search Audit...")
        sources = [row for i, row in self.val_df.iterrows()]
        budgets = [10, 20, 50]
        seeds = [42, 1337, 7]
        methods = {
            "random": self.search_random,
            "greedy": self.search_greedy,
            "metropolis": self.search_metropolis
        }
        
        results = []
        top_examples = []
        
        for q in budgets:
            print(f"Budget Q={q}")
            for seed in seeds:
                for idx, row in enumerate(sources):
                    smi = row["original_representation"]
                    
                    for m_name in ["GraphMPNN", "Transformer"]:
                        for meth_name, meth_func in methods.items():
                            cand, drift, q_used = meth_func(m_name, smi, idx, q, seed)
                            
                            # Cross Eval (Not counting against budget)
                            if m_name == "GraphMPNN":
                                tx_cross_pred = self.predict_tx([cand])[0]
                                tx_orig_pred = self.predict_tx([smi])[0]
                                x_drift = abs(tx_cross_pred - tx_orig_pred)
                            else:
                                gnn_cross_pred = self.predict_gnn([cand])[0]
                                gnn_orig_pred = self.predict_gnn([smi])[0]
                                x_drift = abs(gnn_cross_pred - gnn_orig_pred)
                                
                            results.append({
                                "SourceID": idx, "Source": smi, "Candidate": cand, 
                                "TargetModel": m_name, "Method": meth_name, "Seed": seed,
                                "Budget": q, "QueriesUsed": q_used,
                                "TargetDrift": float(drift), "TransferDrift": float(x_drift)
                            })
                            
                            dist, _ = self.compute_edits(smi, cand)
                            if m_name == "GraphMPNN":
                                top_examples.append({
                                    "source": smi, "candidate": cand, "target": "GraphMPNN",
                                    "drift": float(drift), "transfer_drift": float(x_drift),
                                    "budget": q, "method": meth_name, "seed": seed, "edit_count": dist
                                })
                            
        df = pd.DataFrame(results)
        df.to_csv(self.out_dir / "search_summary.csv", index=False)
        
        with open(self.out_dir / "attack_trajectories.jsonl", "w") as f:
            for t in self.trajectories:
                f.write(json.dumps(t) + "\n")
                
        # Top 20 Audit
        top_df = pd.DataFrame(top_examples)
        top20 = top_df.sort_values("drift", ascending=False).drop_duplicates(subset=["candidate"]).head(20)
        top20.to_csv(self.out_dir / "top_candidate_audit.csv", index=False)
        
        with open(self.out_dir / "proposal_statistics.json", "w") as f:
            json.dump(self.validity_stats, f, indent=2)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Phase12bEvaluator(Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial"), device)
    evaluator.run_adaptive_audits()
    print(f"Phase 12B Execution Complete. Run ID: {evaluator.run_id}")
