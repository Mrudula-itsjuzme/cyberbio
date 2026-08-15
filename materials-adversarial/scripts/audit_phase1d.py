import json
import numpy as np
import pandas as pd
from collections import Counter
from rdkit import Chem

from materials_adv.data.tokenizer import tokenize

def analyze():
    records = []
    with open("data/attacks/substitution.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
            
    print(f"Total Raw Candidates: {len(records)}")
    
    unique_adv_strings = set(r["adversarial_psmiles"] for r in records)
    print(f"Unique Adversarial Strings: {len(unique_adv_strings)}")
    
    unique_canonicals = set()
    for s in unique_adv_strings:
        mol = Chem.MolFromSmiles(s.replace("[*]", "[At]"))
        if mol:
            unique_canonicals.add(Chem.MolToSmiles(mol))
    print(f"Unique Canonical Structures (of valid subset): {len(unique_canonicals)}")
    
    valid_count = sum(1 for r in records if r["validity_status"] == "valid")
    invalid_count = len(records) - valid_count
    print(f"\nValid Candidates: {valid_count} ({valid_count/len(records):.1%})")
    print(f"Invalid Candidates: {invalid_count} ({invalid_count/len(records):.1%})")
    
    reasons = Counter()
    for r in records:
        if r["validity_status"] != "valid":
            reasons.update(r["rejection_reasons"])
    print("\nInvalidity Reasons:")
    for reason, count in reasons.most_common():
        print(f"  - {reason}: {count}")
        
    drifts = [abs(r["prediction_drift"]) for r in records if r["validity_status"] == "valid" and r["prediction_drift"] is not None]
    
    if drifts:
        print("\nDrift Distribution (Kelvin, Valid only):")
        print(f"  Mean: {np.mean(drifts):.2f}")
        print(f"  Median: {np.median(drifts):.2f}")
        print(f"  Std Dev: {np.std(drifts):.2f}")
        print(f"  75th %ile: {np.percentile(drifts, 75):.2f}")
        print(f"  90th %ile: {np.percentile(drifts, 90):.2f}")
        print(f"  95th %ile: {np.percentile(drifts, 95):.2f}")
        print(f"  Max: {np.max(drifts):.2f}")
        
        exceed_52 = sum(1 for d in drifts if d > 52.02)
        print(f"\nCandidates exceeding 52.02 K (baseline MAE): {exceed_52} ({exceed_52/len(drifts):.1%} of valid)")

if __name__ == "__main__":
    analyze()
