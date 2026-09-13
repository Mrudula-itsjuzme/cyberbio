import json
from pathlib import Path
import pandas as pd
import numpy as np

def run_summarize(run_id="1789290270"):
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "results" / "phase8_multi_edit_search" / run_id
    
    print("Loading trajectories...")
    all_trajectories = []
    with (out_dir / "search_trajectories.jsonl").open("r") as f:
        content = f.read()
        # The file contains literal \n instead of actual newlines
        lines = content.split("\\n")
        for line in lines:
            if line.strip():
                all_trajectories.append(json.loads(line.strip()))
            
    with (out_dir / "source_manifest.json").open("r") as f:
        source_items = json.load(f)
            
    print("Generating summaries...")
    df_traj = pd.DataFrame(all_trajectories)
    
    # Average across seeds and sources
    group_cols = ["model", "strategy", "budget"]
    
    summary = []
    for (mod, strat, b), group in df_traj.groupby(group_cols):
        # JSON stringifies dictionary keys, so in the loaded dict they are strings "1", "2", "3"
        e1 = np.mean([c.get("1", 0.0) for c in group["budget_curves"]])
        e2 = np.mean([c.get("2", 0.0) for c in group["budget_curves"]])
        e3 = np.mean([c.get("3", 0.0) for c in group["budget_curves"]])
        
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
    from scripts.evaluate_phase4 import load_ordinary_baseline, load_two_branch_model, load_vocab
    class ModelPredictor:
        def __init__(self, model):
            self.model = model
        def predict(self, texts):
            return self.model.predict(texts)
            
    vocab = load_vocab()
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    baseline_model = load_ordinary_baseline(vocab)
    mixed_robust_model = load_two_branch_model(vocab, p4_dir / "mix_robust_0.1")
    
    pred_base = ModelPredictor(baseline_model)
    pred_robust = ModelPredictor(mixed_robust_model)
    
    # Precompute original predictions
    all_reps = [item["representation"] for item in source_items]
    base_preds = pred_base.predict(all_reps)
    robust_preds = pred_robust.predict(all_reps)
    for i, item in enumerate(source_items):
        item["prediction"] = {
            "ordinary_baseline": float(base_preds[i]),
            "mixed_robust": float(robust_preds[i])
        }
    
    transfer_records = []
    
    baseline_q50 = df_traj[(df_traj["model"] == "ordinary_baseline") & (df_traj["budget"] == 50)]
    # Pick the best seed per source + strategy by best_drift
    best_cands = baseline_q50.loc[baseline_q50.groupby(["source_id", "strategy"])["best_drift"].idxmax()]
    
    for _, row in best_cands.iterrows():
        sid = row["source_id"]
        strat = row["strategy"]
        cand = row["best_candidate"]
        baseline_drift = row["best_drift"]
        
        item = next(i for i in source_items if i["id"] == sid)
        orig_baseline = item["prediction"]["ordinary_baseline"]
        orig_robust = item["prediction"]["mixed_robust"]
        
        pred_robust_cand = float(pred_robust.predict([cand])[0])
        robust_drift = abs(pred_robust_cand - orig_robust)
        
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
    run_summarize()
