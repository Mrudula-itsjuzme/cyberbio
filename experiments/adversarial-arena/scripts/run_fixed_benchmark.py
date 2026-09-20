from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "materials-adversarial" / "src")]

from arena.attackers import mcmc_adapter, random_smiles_adapter
from arena.adapters import cem_adapter, smc_adapter, validity_smc_adapter
from arena.contracts import SourceSample
from arena.transpolymer import TransPolymerDefender

def main():
    seed = 42
    query_budget = 25
    df = pd.read_csv(ROOT / "materials-adversarial" / "data" / "processed" / "processed.csv")
    def sample(rows): return [SourceSample(str(r.polymer_id),r.original_representation,float(r.property_value)) for r in rows.itertuples()]
    validation = sample(df[df.split=="val"].head(10))
    
    vocab = json.loads((ROOT / "materials-adversarial" / "data" / "processed" / "vocab.json").read_text())
    checkpoint_dir = ROOT / "materials-adversarial" / "results" / "models" / "transformer_regressor"
    
    defender = TransPolymerDefender(
        str(ROOT/"materials-adversarial"),
        seed=seed,
        checkpoint_dir=checkpoint_dir,
        lambda_rep=0.1,
        learning_rate=0.001,
        epochs=1,
        batch_size=8
    )
    
    attackers = {
        "RandomSMILES": random_smiles_adapter(seed, query_budget),
        "MCMC": mcmc_adapter(seed, vocab, steps=5, proposal_budget=query_budget),
        "CEM": cem_adapter(seed, query_budget),
        "SMC": smc_adapter(seed, query_budget),
        "Validity-SMC": validity_smc_adapter(seed, query_budget),
    }
    
    results = {}
    for name, adapter in attackers.items():
        print(f"Running {name}...")
        started = time.perf_counter()
        records = adapter.propose(validation, defender)
        elapsed = time.perf_counter() - started
        
        valid_records = [r for r in records if r.validity_status == "valid"]
        drifts = [r.drift for r in valid_records]
        
        mean_drift = np.mean(drifts) if drifts else 0.0
        max_drift = np.max(drifts) if drifts else 0.0
        
        # Calculate validity rate: total valid / total proposed
        # Or from target_query_count
        target_queries = sum(r.target_query_count for r in records)
        if hasattr(adapter, 'get_state'):
            state = adapter.get_state()
            target_queries = state.get("target_queries", target_queries)
        
        results[name] = {
            "mean_drift": float(mean_drift),
            "max_drift": float(max_drift),
            "valid_candidates": len(valid_records),
            "total_queries": target_queries,
            "runtime_seconds": elapsed,
        }
    
    print("\nBENCHMARK RESULTS:")
    print(json.dumps(results, indent=2))
    
    with open("results/fixed_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
