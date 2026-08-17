import json
import pandas as pd
from pathlib import Path

def is_label_preserving(row):
    # Only Substitution and Rearrangement with budget=1 are eligible
    if row['attack_type'] in ['rearrangement', 'substitution'] and row.get('attack_budget', 1) == 1:
        return True
    return False

def main():
    # Load original processed data
    proc_dir = Path("data/processed")
    orig_df = pd.read_csv(proc_dir / "processed.csv")
    with open(proc_dir / "splits.json") as f:
        splits = json.load(f)
        
    train_idx = splits["train"]
    train_df = orig_df.iloc[train_idx].copy()
    print(f"Original training records: {len(train_df)}")
    
    sample_to_val = {}
    sample_to_units = {}
    sample_to_name = {}
    for idx in train_idx:
        row = orig_df.iloc[idx]
        s_id = f"sample_{idx}"
        sample_to_val[s_id] = row["property_value"]
        sample_to_units[s_id] = row["units"]
        sample_to_name[s_id] = row["property_name"]
        
    augmented_records = []
    
    # Track unique sequences to avoid duplicates
    existing_seqs = set(train_df["original_representation"])
    
    with open("data/attacks/phase2_train_attacks.jsonl", "r") as f:
        for line in f:
            r = json.loads(line)
            # Must be valid, plausible, label-preserving, and not already in the train set
            if r["validity_status"] == "valid" and r["plausibility_status"] == "plausible":
                if is_label_preserving(r):
                    adv_seq = r["adversarial_representation"]
                    if adv_seq not in existing_seqs:
                        s_id = r["sample_id"]
                        augmented_records.append({
                            "polymer_id": f"aug_{r['attack_id']}",
                            "original_representation": adv_seq,
                            "property_name": sample_to_name[s_id],
                            "property_value": sample_to_val[s_id],
                            "units": sample_to_units[s_id],
                            "source_dataset": "adversarial_augmentation",
                            "split": "train"
                        })
                        existing_seqs.add(adv_seq)
                
    aug_df = pd.DataFrame(augmented_records)
    print(f"Adding {len(aug_df)} strictly label-preserving adversarial examples.")
    
    combined_df = pd.concat([train_df, aug_df], ignore_index=True)
    out_path = proc_dir / "augmented_train.csv"
    combined_df.to_csv(out_path, index=False)
    print(f"Saved augmented dataset to {out_path} (Total rows: {len(combined_df)})")

if __name__ == "__main__":
    main()
