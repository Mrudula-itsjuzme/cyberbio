import json
import numpy as np
from collections import Counter, defaultdict
from rdkit import Chem

def analyze():
    records = []
    with open("data/attacks/phase1e_results.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
            
    # Group by attack type
    attacks = defaultdict(list)
    for r in records:
        attacks[r["attack_type"]].append(r)
        
    print("="*60)
    print(f"{'PHASE 1E EXPERIMENT RESULTS':^60}")
    print("="*60)
    
    for attack_type, type_records in attacks.items():
        print(f"\n--- ATTACK: {attack_type.upper()} ---")
        total = len(type_records)
        unique_strings = set(r["adversarial_psmiles"] for r in type_records)
        
        valid_subset = [r for r in type_records if r["validity_status"] == "valid"]
        valid_count = len(valid_subset)
        invalid_count = total - valid_count
        
        unique_canonicals = set()
        for s in set(r["adversarial_psmiles"] for r in valid_subset):
            mol = Chem.MolFromSmiles(s.replace("[*]", "[At]"))
            if mol:
                unique_canonicals.add(Chem.MolToSmiles(mol))
                
        print(f"Raw candidates: {total}")
        print(f"Unique strings: {len(unique_strings)}")
        print(f"Unique canonical structures (valid only): {len(unique_canonicals)}")
        print(f"RDKit-valid candidates: {valid_count} ({valid_count/total:.1%} of generated)")
        print(f"Invalid candidates: {invalid_count} ({invalid_count/total:.1%} of generated)")
        
        if invalid_count > 0:
            reasons = Counter()
            for r in type_records:
                if r["validity_status"] != "valid":
                    for reason in r.get("rejection_reasons", []):
                        reasons[reason] += 1
            # print("  Top 3 invalidity reasons:")
            # for reason, count in reasons.most_common(3):
            #     print(f"    - {reason}: {count}")
                
        if valid_count > 0:
            drifts = [abs(r["prediction_drift"]) for r in valid_subset if r["prediction_drift"] is not None]
            
            print(f"\nDrift Distribution (Kelvin, computed over {valid_count} valid candidates):")
            print(f"  Mean:       {np.mean(drifts):.2f}")
            print(f"  Median:     {np.median(drifts):.2f}")
            print(f"  Std Dev:    {np.std(drifts):.2f}")
            print(f"  75th %ile:  {np.percentile(drifts, 75):.2f}")
            print(f"  90th %ile:  {np.percentile(drifts, 90):.2f}")
            print(f"  95th %ile:  {np.percentile(drifts, 95):.2f}")
            print(f"  Max drift:  {np.max(drifts):.2f}")
            
            exceed_52 = sum(1 for d in drifts if d > 52.02)
            print(f"  > 52.02 K:  {exceed_52} ({exceed_52/valid_count:.1%} of valid, {exceed_52/total:.1%} of raw)")
            
            # Since n_edits/window changes vary, calculate drift per edit roughly
            # (substitution, insertion, deletion = 1 edit. rearrangement = window size edits)
            changes = [r.get("number_of_changes", 1) for r in valid_subset if r["prediction_drift"] is not None]
            drifts_per_edit = [d/max(c, 1) for d, c in zip(drifts, changes)]
            print(f"  Mean drift per edit: {np.mean(drifts_per_edit):.2f}")

if __name__ == "__main__":
    analyze()
