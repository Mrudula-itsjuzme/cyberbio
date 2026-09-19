import json
import csv
import os

outputs = []
p1 = "materials-adversarial/results/canonical_benchmark_no_leakage.json"
if os.path.exists(p1):
    with open(p1, "r") as f:
        data = json.load(f)
        outputs.append({
            "artifact": p1,
            "script": "scripts/run_comprehensive_benchmark_suite.py",
            "config": "canonical",
            "dataset_split": "scaffold",
            "seed": "42",
            "metric": "clean RMSE",
            "value": data.get("clean_rmse", "N/A")
        })
        outputs.append({
            "artifact": p1,
            "script": "scripts/run_comprehensive_benchmark_suite.py",
            "config": "canonical",
            "dataset_split": "scaffold",
            "seed": "42",
            "metric": "mean representation drift",
            "value": data.get("mean_drift", "N/A")
        })

os.makedirs("materials-adversarial/results", exist_ok=True)
with open("materials-adversarial/results/verified_canonical_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["artifact", "script", "config", "dataset_split", "seed", "metric", "value"])
    writer.writeheader()
    writer.writerows(outputs)

with open("materials-adversarial/results/verified_canonical_results.json", "w") as f:
    json.dump(outputs, f, indent=4)
    
print("Verified canonical results exported.")
