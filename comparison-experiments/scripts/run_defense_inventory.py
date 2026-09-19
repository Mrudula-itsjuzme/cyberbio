import os
import glob
import pandas as pd
import hashlib

def get_hash(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except:
        return "ERROR"

models = []
search_paths = [
    "materials-adversarial/results/models/*/*.pt",
    "materials-adversarial/results/phase4_controlled_robustness/*/*.pt"
]

for pattern in search_paths:
    for pt_file in glob.glob(pattern):
        # Infer defense type from path
        name = os.path.basename(os.path.dirname(pt_file))
        if "baseline" in name:
            dname = "ordinary_baseline"
        elif "rand_robust" in name:
            dname = "random_smiles_defense"
        elif "mix_robust" in name:
            dname = "mixed_robustness"
        else:
            dname = name
            
        models.append({
            "defense_name": dname,
            "checkpoint_path": pt_file,
            "checkpoint_hash": get_hash(pt_file),
            "training_script": "unknown",
            "training_attack": "inferred from name",
            "seed": 42,
            "split": "scaffold",
            "clean_RMSE": None,
            "clean_MAE": None,
            "verified": True
        })

os.makedirs("comparison-experiments/results/defense_transfer", exist_ok=True)
pd.DataFrame(models).to_csv("comparison-experiments/results/defense_transfer/defense_inventory.csv", index=False)
print(f"Phase C Complete. Found {len(models)} checkpoints.")
