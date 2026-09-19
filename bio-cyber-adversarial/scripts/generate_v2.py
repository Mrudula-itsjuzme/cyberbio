import os
import random
import pandas as pd

random.seed(42)

def generate_random_dna(length):
    return "".join(random.choices("ACGT", k=length))

def inject_motif(seq, pos1, pos2, m1="ATGC", m2="GCAT"):
    s = list(seq)
    s[pos1:pos1+4] = list(m1)
    s[pos2:pos2+4] = list(m2)
    return "".join(s)

dataset = []
splits = [("train", 16000), ("val", 2000), ("test", 2000)]

seq_len = 100

for split_name, count in splits:
    # Generate pool
    split_data = []
    
    # We want exactly count/2 of label 0 and count/2 of label 1
    # We'll overgenerate and filter to unique, then take exact amounts
    class_0_pool = set()
    class_1_pool = set()
    
    # Class 1 (distance exactly 15)
    while len(class_1_pool) < count // 2:
        base_seq = generate_random_dna(seq_len)
        pos1 = random.randint(0, seq_len - 15 - 4)
        pos2 = pos1 + 15
        final_seq = inject_motif(base_seq, pos1, pos2)
        class_1_pool.add((final_seq, 1, f"{pos1}_{pos2}", split_name))
        
    # Class 0 (distance 5 or 25)
    while len(class_0_pool) < count // 2:
        base_seq = generate_random_dna(seq_len)
        dist = random.choice([5, 25])
        pos1 = random.randint(0, seq_len - dist - 4)
        pos2 = pos1 + dist
        final_seq = inject_motif(base_seq, pos1, pos2)
        # Ensure it doesn't overlap with class 1 just in case
        if (final_seq, 1, f"{pos1}_{pos2}", split_name) not in class_1_pool:
            class_0_pool.add((final_seq, 0, f"{pos1}_{pos2}", split_name))
            
    dataset.extend(list(class_0_pool))
    dataset.extend(list(class_1_pool))

# Ensure no cross-split duplicates
df = pd.DataFrame(dataset, columns=["sequence", "label", "motif_pos", "split"])

# Check cross-split duplicates
seen_seqs = set()
valid_rows = []
for idx, row in df.iterrows():
    if row["sequence"] not in seen_seqs:
        seen_seqs.add(row["sequence"])
        valid_rows.append(row)

df_clean = pd.DataFrame(valid_rows)

print("V2 Dataset generation complete. Rows:", len(df_clean))
print(df_clean.groupby(['split', 'label']).size())

os.makedirs("data/v2", exist_ok=True)
df_clean.to_csv("data/v2/dataset.csv", index=False)
