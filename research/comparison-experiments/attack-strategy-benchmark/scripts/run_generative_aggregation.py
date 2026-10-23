import json, os
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def aggregate_generative(domain):
    results_dir = ROOT / f"results/generative/{domain}"
    
    if not (results_dir / "per_example_results_trained.csv").exists():
        return
        
    df1 = pd.read_csv(results_dir / "per_example_results_trained.csv")
    df2 = pd.read_csv(results_dir / "per_example_results_untrained.csv")
    
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(results_dir / "per_example_results.csv", index=False)
    
    agg_funcs = {
        "absolute_drift": "mean",
        "edit_distance": "mean",
        "proposal_attempts": "mean",
        "valid_candidates": "mean",
        "attack_model_queries": "mean",
        "total_model_queries": "mean",
        "runtime_seconds": "mean",
        "generator_inference_count": "mean"
    }
    
    if domain == "bio_cyber":
        agg_funcs["success"] = "mean"
    elif domain == "materials":
        agg_funcs["tanimoto_similarity"] = "mean"
        
    agg = df.groupby(["attack_condition", "search_budget"]).agg(agg_funcs).reset_index()
    
    # termination reasons
    for reason in ["QUERY_BUDGET_EXHAUSTED", "PROPOSAL_CAP_REACHED", "NO_VALID_CANDIDATE"]:
        agg[f"frac_{reason}"] = df.groupby(["attack_condition", "search_budget"])["termination_reason"].apply(lambda x: (x == reason).mean()).values
        
    agg.to_csv(results_dir / "aggregated_results.csv", index=False)

if __name__ == "__main__":
    aggregate_generative("bio_cyber")
    aggregate_generative("materials")
