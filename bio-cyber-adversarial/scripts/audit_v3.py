import pandas as pd
import numpy as np
import json
import os

df = pd.read_csv("data/v3/dataset.csv")

audit = {
    "exact_duplicates": int(len(df) - df["sequence"].nunique()),
    "class_counts": {f"{k[0]}_{k[1]}": int(v) for k, v in df.groupby(["split", "label"]).size().items()},
    "sequence_lengths": {k: float(v) for k, v in df["sequence"].apply(len).describe().items()},
    "edge_distance_min": int(min(df["pos1"].min(), df["pos2"].min())),
    "edge_distance_max_from_end": int(150 - max((df["pos1"]+4).max(), (df["pos2"]+4).max())),
    "distance_stats_by_class": {str(k): {k2: float(v2) for k2, v2 in v.items()} for k, v in df.groupby("label")["distance"].describe().to_dict().items()}
}

os.makedirs("results", exist_ok=True)
with open("results/v3_integrity_audit.json", "w") as f:
    json.dump(audit, f, indent=4)
print("V3 audited.")
