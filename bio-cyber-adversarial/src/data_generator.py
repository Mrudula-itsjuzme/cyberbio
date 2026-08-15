import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split

def generate_background_sequence(length, alphabet, forbid_motifs=None):
    """Generates a random sequence, ensuring it DOES NOT contain any of the forbidden motifs."""
    while True:
        seq = "".join(random.choices(alphabet, k=length))
        if forbid_motifs:
            if any(m in seq for m in forbid_motifs):
                continue # Retry if it accidentally generated a motif
        return seq

def implant_motif(sequence, motif):
    """Implants a motif at a random position in the sequence."""
    if len(motif) > len(sequence):
        raise ValueError("Motif is longer than sequence.")
    
    pos = random.randint(0, len(sequence) - len(motif))
    new_sequence = sequence[:pos] + motif + sequence[pos + len(motif):]
    return new_sequence, pos

def generate_dataset(
    num_samples=20000,
    seq_length=50,
    motif_0="ACGTACGT",
    motif_1="TGCATGCA",
    seed=42,
    output_path="../data/raw/dataset.csv"
):
    print(f"Generating {num_samples} symmetric synthetic sequences (length {seq_length})...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    alphabet = ['A', 'C', 'G', 'T']
    
    sequences = []
    labels = []
    motif_positions = []
    
    num_class_0 = num_samples // 2
    num_class_1 = num_samples - num_class_0
    
    forbid = [motif_0, motif_1]
    
    # Class 0
    for _ in range(num_class_0):
        bg_seq = generate_background_sequence(seq_length, alphabet, forbid_motifs=forbid)
        seq, pos = implant_motif(bg_seq, motif_0)
        sequences.append(seq)
        labels.append(0)
        motif_positions.append(pos)
        
    # Class 1
    for _ in range(num_class_1):
        bg_seq = generate_background_sequence(seq_length, alphabet, forbid_motifs=forbid)
        seq, pos = implant_motif(bg_seq, motif_1)
        sequences.append(seq)
        labels.append(1)
        motif_positions.append(pos)
        
    df = pd.DataFrame({
        "sequence": sequences,
        "label": labels,
        "motif_pos": motif_positions
    })
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['label'], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['label'], random_state=seed
    )
    
    df['split'] = 'none'
    df.loc[train_df.index, 'split'] = 'train'
    df.loc[val_df.index, 'split'] = 'val'
    df.loc[test_df.index, 'split'] = 'test'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # --- RIGOROUS VALIDATION CHECKS ---
    print("\n--- Validation Statistics ---")
    print(f"Class Balance: {df['label'].value_counts().to_dict()}")
    
    df['has_m0'] = df['sequence'].str.contains(motif_0)
    df['has_m1'] = df['sequence'].str.contains(motif_1)
    
    m0_freq = pd.crosstab(df['label'], df['has_m0']).to_dict()
    m1_freq = pd.crosstab(df['label'], df['has_m1']).to_dict()
    print(f"Motif 0 by Class (should be 100% in 0, 0% in 1): \n{pd.crosstab(df['label'], df['has_m0'])}")
    print(f"Motif 1 by Class (should be 0% in 0, 100% in 1): \n{pd.crosstab(df['label'], df['has_m1'])}")
    
    pos_0 = df[df['label'] == 0]['motif_pos']
    pos_1 = df[df['label'] == 1]['motif_pos']
    print(f"Motif 0 position mean: {pos_0.mean():.2f} std: {pos_0.std():.2f}")
    print(f"Motif 1 position mean: {pos_1.mean():.2f} std: {pos_1.std():.2f}")
    
    invalid_chars = sum([1 for seq in df['sequence'] if set(seq) - set(alphabet)])
    print(f"Samples with invalid characters: {invalid_chars}")
    
    wrong_length = sum([1 for seq in df['sequence'] if len(seq) != seq_length])
    print(f"Samples with incorrect length: {wrong_length}")
    
    print("\nSplit counts:")
    print(df['split'].value_counts().to_dict())

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "data", "raw", "dataset.csv")
    generate_dataset(output_path=output_path)
