import json
import numpy as np
from collections import defaultdict

def calculate_metrics(records):
    valid_records = [r for r in records if r["validity_status"] == "valid"]
    metrics = defaultdict(dict)
    
    attacks = defaultdict(list)
    for r in valid_records:
        attacks[r["attack_type"]].append(r["prediction_drift"])
        
    for atype, drifts in attacks.items():
        if not drifts: continue
        abs_drifts = [abs(d) for d in drifts if d is not None]
        metrics[atype] = {
            "mean": np.mean(abs_drifts),
            "median": np.median(abs_drifts),
            "p95": np.percentile(abs_drifts, 95),
            "max": np.max(abs_drifts),
            "above_52": sum(1 for d in abs_drifts if d > 52.02) / len(abs_drifts)
        }
    return metrics

def run_audit():
    with open("data/attacks/phase1e_results.jsonl") as f:
        base_records = [json.loads(line) for line in f]
        
    with open("data/attacks/phase2a_results.jsonl") as f:
        def_records = [json.loads(line) for line in f]
        
    base_metrics = calculate_metrics(base_records)
    def_metrics = calculate_metrics(def_records)
    
    print("="*60)
    print(f"{'PHASE 2A ADVERSARIAL DEFENSE AUDIT':^60}")
    print("="*60)
    
    for atype in ["substitution", "insertion", "deletion", "rearrangement"]:
        b = base_metrics.get(atype)
        d = def_metrics.get(atype)
        
        if not b or not d: continue
        
        print(f"--- {atype.upper()} ---")
        print(f"  Mean Drift:   {b['mean']:.2f} K  ->  {d['mean']:.2f} K  (Red: {b['mean'] - d['mean']:.2f} K)")
        print(f"  Median Drift: {b['median']:.2f} K  ->  {d['median']:.2f} K  (Red: {b['median'] - d['median']:.2f} K)")
        print(f"  P95 Drift:    {b['p95']:.2f} K  ->  {d['p95']:.2f} K  (Red: {b['p95'] - d['p95']:.2f} K)")
        print(f"  Max Drift:    {b['max']:.2f} K  ->  {d['max']:.2f} K  (Red: {b['max'] - d['max']:.2f} K)")
        print(f"  > 52.02 K %:  {b['above_52']:.1%}  ->  {d['above_52']:.1%}")
        print()

if __name__ == "__main__":
    run_audit()
