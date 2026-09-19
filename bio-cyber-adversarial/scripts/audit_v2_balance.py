import pandas as pd
import json
import numpy as np
from collections import Counter

df = pd.read_csv("data/v2/dataset.csv")

def get_distance(pos_str):
    if "_" in str(pos_str):
        p1, p2 = map(int, str(pos_str).split("_"))
        return abs(p1 - p2)
    return 0

def get_pos(pos_str):
    if "_" in str(pos_str):
        p1, p2 = map(int, str(pos_str).split("_"))
        return p1
    return int(pos_str)

df['length'] = df['sequence'].apply(len)
df['distance'] = df['motif_pos'].apply(get_distance)
df['start_pos'] = df['motif_pos'].apply(get_pos)

audit = {
    "total_rows": len(df),
    "train_rows": len(df[df['split'] == 'train']),
    "val_rows": len(df[df['split'] == 'val']),
    "test_rows": len(df[df['split'] == 'test']),
    "class_counts_by_split": df.groupby(['split', 'label']).size().to_dict(),
    "sequence_length_distribution": df['length'].describe().to_dict(),
    "duplicates": len(df) - len(df['sequence'].unique()),
    "motif_position_distribution": df['start_pos'].describe().to_dict(),
    "distance_distribution": df['distance'].value_counts().to_dict()
}

# Cross split overlap
train_seqs = set(df[df['split'] == 'train']['sequence'])
val_seqs = set(df[df['split'] == 'val']['sequence'])
test_seqs = set(df[df['split'] == 'test']['sequence'])

audit["cross_split_overlap"] = {
    "train_val": len(train_seqs.intersection(val_seqs)),
    "train_test": len(train_seqs.intersection(test_seqs)),
    "val_test": len(val_seqs.intersection(test_seqs))
}

# Convert tuple keys in class_counts to strings for JSON
new_counts = {}
for k, v in audit["class_counts_by_split"].items():
    new_counts[f"{k[0]}_class_{k[1]}"] = int(v)
audit["class_counts_by_split"] = new_counts

with open("results/v2_balance_audit.json", "w") as f:
    json.dump(audit, f, indent=4)
print("V2 Balance Audit complete.")
