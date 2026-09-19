import os
import json
import pandas as pd
import glob

results = {
    "datasets": [],
    "models": [],
    "attack_implementations": []
}

data_files = [
    "data/raw/dataset.csv",
    "data/raw/adv_dataset_against_defended.csv",
    "data/raw/adv_dataset.csv"
]
for p in data_files:
    if os.path.exists(p):
        df = pd.read_csv(p)
        results["datasets"].append({
            "path": p,
            "rows": len(df),
            "columns": list(df.columns)
        })

for p in glob.glob("models/*.pt") + glob.glob("models/*.pth"):
    results["models"].append(p)

for p in glob.glob("src/attacks/*.py"):
    results["attack_implementations"].append(p)

os.makedirs("results", exist_ok=True)
with open("results/real_state.json", "w") as f:
    json.dump(results, f, indent=4)

print("Phase G Complete.")
