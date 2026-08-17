import json
import numpy as np

for mode in ["baseline", "residualized", "defended"]:
    try:
        with open(f"data/attacks/phase3_{mode}_attacks.jsonl") as f:
            records = [json.loads(line) for line in f]
    except Exception as e:
        continue
        
    print(f"\n--- {mode.upper()} ---")
    valid_records = [r for r in records if r["validity_status"] == "valid"]
    print(f"Total attacks: {len(records)}, Valid: {len(valid_records)}")
    
    types = set(r["attack_type"] for r in valid_records)
    for t in sorted(types):
        type_recs = [r for r in valid_records if r["attack_type"] == t]
        drifts = [abs(r["prediction_drift"]) for r in type_recs if r["prediction_drift"] is not None]
        mcmc_accepted = [r for r in type_recs if r.get("attack_params", {}).get("mcmc_accepted", False)]
        if drifts:
            print(f"  {t}: {len(type_recs)} valid, Mean drift: {np.mean(drifts):.2f} K, Max drift: {np.max(drifts):.2f} K")
        if t == "probabilistic_mcmc":
            print(f"    MCMC Accepts: {len(mcmc_accepted)}")
