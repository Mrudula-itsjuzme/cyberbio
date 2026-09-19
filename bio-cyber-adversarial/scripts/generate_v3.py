import numpy as np
import pandas as pd
import json
import os
import hashlib
from collections import Counter

np.random.seed(42)
N_TRAIN_PER_CLASS = 8000
N_VAL_PER_CLASS = 1000
N_TEST_PER_CLASS = 1000
SEQ_LEN = 150

MOTIF_A = "ATGC"
MOTIF_B = "GCAT"

def generate_random_background(length):
    return "".join(np.random.choice(["A", "C", "G", "T"], size=length))

def insert_motifs(distance, length, motif1, motif2):
    # distance is between the START of motif1 and START of motif2
    # But wait, motif length is 4. distance must be >= 4 to avoid overlap.
    # User ranges: 30-40, 60-70. No overlap is guaranteed.
    max_pos1 = length - distance - len(motif2)
    pos1 = np.random.randint(0, max_pos1 + 1)
    pos2 = pos1 + distance
    
    bg = list(generate_random_background(length))
    bg[pos1:pos1+4] = list(motif1)
    bg[pos2:pos2+4] = list(motif2)
    return "".join(bg), pos1, pos2

def generate_class(label, n, dist_min, dist_max):
    seqs = []
    seen = set()
    
    while len(seqs) < n:
        dist = np.random.randint(dist_min, dist_max + 1)
        if np.random.rand() > 0.5:
            m1, m2 = MOTIF_A, MOTIF_B
            order = "A_then_B"
        else:
            m1, m2 = MOTIF_B, MOTIF_A
            order = "B_then_A"
            
        seq, p1, p2 = insert_motifs(dist, SEQ_LEN, m1, m2)
        if seq not in seen:
            seen.add(seq)
            seqs.append({
                "sequence": seq,
                "label": label,
                "distance": dist,
                "pos1": p1,
                "pos2": p2,
                "order": order
            })
    return pd.DataFrame(seqs)

os.makedirs("data/v3", exist_ok=True)

df_c1_train = generate_class(1, N_TRAIN_PER_CLASS, 30, 40)
df_c1_val = generate_class(1, N_VAL_PER_CLASS, 30, 40)
df_c1_test = generate_class(1, N_TEST_PER_CLASS, 30, 40)

df_c0_train = generate_class(0, N_TRAIN_PER_CLASS, 60, 70)
df_c0_val = generate_class(0, N_VAL_PER_CLASS, 60, 70)
df_c0_test = generate_class(0, N_TEST_PER_CLASS, 60, 70)

df_c1_train["split"] = "train"
df_c0_train["split"] = "train"
df_c1_val["split"] = "val"
df_c0_val["split"] = "val"
df_c1_test["split"] = "test"
df_c0_test["split"] = "test"

df = pd.concat([df_c1_train, df_c0_train, df_c1_val, df_c0_val, df_c1_test, df_c0_test])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["id"] = range(len(df))

# Ensure absolutely no duplicates anywhere
assert df["sequence"].nunique() == len(df), "Duplicates found!"

df.to_csv("data/v3/dataset.csv", index=False)

manifest = {
    "n_train_per_class": N_TRAIN_PER_CLASS,
    "n_val_per_class": N_VAL_PER_CLASS,
    "n_test_per_class": N_TEST_PER_CLASS,
    "seq_len": SEQ_LEN,
    "class_1_dist": [30, 40],
    "class_0_dist": [60, 70],
    "total_sequences": len(df),
    "dataset_hash": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
}

with open("data/v3/generation_manifest.json", "w") as f:
    json.dump(manifest, f, indent=4)
print("V3 dataset generated.")
