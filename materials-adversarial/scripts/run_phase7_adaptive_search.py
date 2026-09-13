#!/usr/bin/env python3
"""Run Phase 7 Adaptive Adversarial Search.

Evaluates the Ordinary Baseline and Mixed Robust model against
Random, Greedy, and Metropolis stochastic search using a strict query budget.
"""

import json
from pathlib import Path
import time

import numpy as np

from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab
from src.materials_adv.attacks.substitution import SubstitutionAttack
from src.materials_adv.attacks.search.proposals import CompositeProposalOperator
from src.materials_adv.attacks.search.strategies import RandomSearch, GreedySearch, MetropolisSearch

class ModelPredictor:
    def __init__(self, model):
        self.model = model
    def predict(self, texts):
        return self.model.predict(texts)

def run_phase7_search():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    models_dir = repo_root / "results" / "models"
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    out_dir = repo_root / "results" / "phase7_adaptive_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vocab = load_vocab()
    
    print("Loading models...")
    models = {
        "ordinary_baseline": load_ordinary_baseline(vocab),
        "mixed_robust": load_two_branch_model(vocab, p4_dir / "mix_robust_0.1"),
    }
    predictors = {k: ModelPredictor(v) for k, v in models.items()}
    
    # Extract 100 validation items
    splits_path = data_dir / "processed" / "splits.json"
    processed_path = data_dir / "processed" / "processed.csv"
    
    with splits_path.open("r", encoding="utf-8") as f:
        val_ids = json.load(f)["val"]
        
    df = __import__('pandas').read_csv(processed_path)
    val_df = df.iloc[val_ids]
    
    source_items = []
    for _, row in val_df.iterrows():
        source_items.append({
            "id": row["polymer_id"],
            "representation": row["original_representation"]
        })
        if len(source_items) >= 100:
            break
                
    print(f"Loaded {len(source_items)} source polymers for evaluation.")
    
    rng = np.random.default_rng(42)
    # Using single-token substitution
    # vocab is already a list of tokens from load_vocab
    allowed_tokens = vocab
    
    sub_attack = SubstitutionAttack(rng, allowed_tokens=allowed_tokens, attack_budget=1, role_preserving=True)
    proposal = CompositeProposalOperator([sub_attack], rng)
    
    budgets = [5, 10, 20, 50]
    
    results = []
    
    out_file = out_dir / "search_trajectories.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for budget in budgets:
            print(f"--- Query Budget: {budget} ---")
            
            strategies = [
                RandomSearch(predictors["ordinary_baseline"], proposal, rng, query_budget=budget, max_changes=1),
                GreedySearch(predictors["ordinary_baseline"], proposal, rng, query_budget=budget, max_changes=1),
                MetropolisSearch(predictors["ordinary_baseline"], proposal, rng, query_budget=budget, max_changes=1)
            ]
            
            for strategy in strategies:
                strat_name = strategy.name
                for model_name, predictor in predictors.items():
                    print(f"Running {strat_name} on {model_name}...")
                    # Update strategy predictor
                    strategy.predictor = predictor
                    
                    for item in source_items:
                        original_rep = item["representation"]
                        source_id = item["id"]
                        
                        search_result = strategy.search(original_rep)
                        
                        record = search_result.summary()
                        record["model"] = model_name
                        record["budget"] = budget
                        record["source_id"] = source_id
                        
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        results.append(record)

if __name__ == "__main__":
    run_phase7_search()
