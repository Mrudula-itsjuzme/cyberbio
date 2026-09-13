#!/usr/bin/env python3
"""Run Phase 8: Multi-Edit Adaptive Adversarial Search.

Executes search under attack_budget=3 for Random, Greedy, and Metropolis.
Extracts best drift achieved at E=1, E=2, and E=3.
"""

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab
from src.materials_adv.attacks.substitution import SubstitutionAttack
from src.materials_adv.attacks.search.proposals import CompositeProposalOperator
from src.materials_adv.attacks.search.strategies import RandomSearch, GreedySearch, MetropolisSearch
from src.materials_adv.validation.pipeline import validate

class ModelPredictor:
    def __init__(self, model):
        self.model = model
    def predict(self, texts):
        return self.model.predict(texts)

def extract_budget_curve(trace, original_prediction):
    """Extract the best drift achieved at exactly E=1, E=2, and E=3."""
    best_drifts = {1: 0.0, 2: 0.0, 3: 0.0}
    
    for entry in trace:
        if entry.get("representation_valid") and entry.get("plausible", True) and "perturbation_size" in entry and entry.get("queried", False):
            size = entry["perturbation_size"]
            drift = entry.get("absolute_drift", 0.0)
            if size in best_drifts:
                if drift > best_drifts[size]:
                    best_drifts[size] = drift
                    
    if best_drifts[2] < best_drifts[1]: best_drifts[2] = best_drifts[1]
    if best_drifts[3] < best_drifts[2]: best_drifts[3] = best_drifts[2]
    
    return best_drifts

def run_phase8():
    run_id = str(int(time.time()))
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    out_dir = repo_root / "results" / "phase8_multi_edit_search" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    vocab = load_vocab()
    
    print("Loading models...")
    models = {
        "ordinary_baseline": load_ordinary_baseline(vocab),
        "mixed_robust": load_two_branch_model(vocab, p4_dir / "mix_robust_0.1"),
    }
    predictors = {k: ModelPredictor(v) for k, v in models.items()}
    
    splits_path = data_dir / "processed" / "splits.json"
    processed_path = data_dir / "processed" / "processed.csv"
    
    with splits_path.open("r", encoding="utf-8") as f:
        val_ids = json.load(f)["val"]
        
    df = pd.read_csv(processed_path)
    val_df = df.iloc[val_ids]
    
    source_items = []
    for _, row in val_df.iterrows():
        source_items.append({
            "id": str(row["polymer_id"]),
            "representation": row["original_representation"],
            "prediction": {}
        })
        if len(source_items) >= 100:
            break
            
    with (out_dir / "source_manifest.json").open("w") as f:
        json.dump(source_items, f, indent=2)
        
    # Strategies configuration
    configs = [
        {"name": "random", "class": RandomSearch, "seeds": [42, 100, 2026, 9999, 12345]},
        {"name": "metropolis", "class": MetropolisSearch, "seeds": [42, 100, 2026, 9999, 12345]},
        {"name": "greedy", "class": GreedySearch, "seeds": [42]},  # deterministic
    ]
    budgets = [10, 20, 50]
    
    all_trajectories = []
    summary_stats = []
    
    for model_name, predictor in predictors.items():
        print(f"\\nEvaluating against {model_name}...")
        
        # Cache original predictions
        reps = [item["representation"] for item in source_items]
        preds = predictor.predict(reps)
        for i, item in enumerate(source_items):
            item["prediction"][model_name] = float(preds[i])
            
        for config in configs:
            strat_name = config["name"]
            StratClass = config["class"]
            
            for seed in config["seeds"]:
                print(f"  {strat_name} (Seed {seed})...")
                
                rng = np.random.default_rng(seed)
                sub_attack = SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1, role_preserving=True)
                proposal = CompositeProposalOperator([sub_attack], rng)
                
                kwargs = {"max_changes": 3, "query_budget": 50}
                if strat_name == "metropolis":
                    kwargs["temperature"] = 0.1
                    
                strategy = StratClass(predictor, proposal, rng, **kwargs)
                
                for item in source_items:
                    orig = item["representation"]
                    orig_pred = item["prediction"][model_name]
                    
                    res = strategy.search(orig)
                    
                    for budget in budgets:
                        # Slice the trace up to the given budget
                        sliced_trace = []
                        q_count = 0
                        for t in res.trace:
                            sliced_trace.append(t)
                            if t.get("queried", False):
                                q_count += 1
                            if q_count >= budget:
                                break
                                
                        # Re-calculate best drift and best candidate for the sliced trace
                        best_drift = 0.0
                        best_cand = orig
                        for t in sliced_trace:
                            if t.get("queried", False) and t.get("representation_valid", False) and t.get("plausible", True):
                                d = t.get("absolute_drift", 0.0)
                                if d > best_drift:
                                    best_drift = d
                                    best_cand = t.get("candidate", orig)
                                    
                        budget_curves = extract_budget_curve(sliced_trace, orig_pred)
                        
                        traj_doc = {
                            "source_id": item["id"],
                            "model": model_name,
                            "strategy": strat_name,
                            "seed": seed,
                            "budget": budget,
                            "best_drift": best_drift,
                            "best_candidate": best_cand,
                            "budget_curves": budget_curves,
                            "total_queries": sum(1 for t in sliced_trace if t.get("queried", False)),
                            "total_valid": sum(1 for t in sliced_trace if t.get("representation_valid", False)),
                        }
                        all_trajectories.append(traj_doc)
                        
    print("\\nWriting trajectories...")
    with (out_dir / "search_trajectories.jsonl").open("w") as f:
        for t in all_trajectories:
            f.write(json.dumps(t) + "\\n")
            
    print("Generating summaries...")
    df_traj = pd.DataFrame(all_trajectories)
    
    # Average across seeds and sources
    group_cols = ["model", "strategy", "budget"]
    
    summary = []
    for (mod, strat, b), group in df_traj.groupby(group_cols):
        e1 = np.mean([c["1"] for c in group["budget_curves"]])
        e2 = np.mean([c["2"] for c in group["budget_curves"]])
        e3 = np.mean([c["3"] for c in group["budget_curves"]])
        
        summary.append({
            "model": mod,
            "strategy": strat,
            "budget": b,
            "mean_drift": group["best_drift"].mean(),
            "std_drift": group["best_drift"].std(),
            "mean_queries": group["total_queries"].mean(),
            "mean_valid": group["total_valid"].mean(),
            "mean_drift_E1": e1,
            "mean_drift_E2": e2,
            "mean_drift_E3": e3,
        })
        
    pd.DataFrame(summary).to_csv(out_dir / "query_efficiency.csv", index=False)
    
    # Attack Transfer
    print("Evaluating Attack Transfer...")
    transfer_records = []
    
    baseline_q50 = df_traj[(df_traj["model"] == "ordinary_baseline") & (df_traj["budget"] == 50)]
    best_cands = baseline_q50.loc[baseline_q50.groupby(["source_id", "strategy"])["best_drift"].idxmax()]
    
    for _, row in best_cands.iterrows():
        sid = row["source_id"]
        strat = row["strategy"]
        cand = row["best_candidate"]
        baseline_drift = row["best_drift"]
        
        item = next(i for i in source_items if i["id"] == sid)
        orig_baseline = item["prediction"]["ordinary_baseline"]
        orig_robust = item["prediction"]["mixed_robust"]
        
        pred_robust = float(predictors["mixed_robust"].predict([cand])[0])
        robust_drift = abs(pred_robust - orig_robust)
        
        transfer_records.append({
            "source_id": sid,
            "strategy": strat,
            "candidate": cand,
            "baseline_drift": baseline_drift,
            "robust_drift": robust_drift,
            "transfer_ratio": robust_drift / baseline_drift if baseline_drift > 1e-6 else 0.0
        })
        
    pd.DataFrame(transfer_records).to_csv(out_dir / "attack_transfer.csv", index=False)
    
    print("Done!")

if __name__ == "__main__":
    run_phase8()
