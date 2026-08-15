import json
import numpy as np
from collections import defaultdict
import sys
import os

# Add src to path to import tokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from materials_adv.data.tokenizer import tokenize
from materials_adv.attacks.token_space import count_changes

def analyze_attribution():
    records = []
    with open("data/attacks/phase1e_results.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
            
    # Filter only valid records
    valid_records = [r for r in records if r["validity_status"] == "valid"]
    
    print("="*60)
    print(f"{'PHASE 1F PERTURBATION ATTRIBUTION ANALYSIS':^60}")
    print("="*60)
    print(f"Analyzing {len(valid_records)} representation-valid candidates.\n")
    
    attacks = defaultdict(list)
    delta_length_groups = defaultdict(list)
    changes_groups = defaultdict(list)
    
    for r in valid_records:
        orig_tokens = tokenize(r["original_psmiles"])
        adv_tokens = tokenize(r["adversarial_psmiles"])
        
        orig_len = len(orig_tokens)
        adv_len = len(adv_tokens)
        delta_len = adv_len - orig_len
        
        n_changes = count_changes(orig_tokens, adv_tokens)
        
        orig_pred = r["original_prediction"]
        adv_pred = r["adversarial_prediction"]
        signed_shift = adv_pred - orig_pred
        abs_drift = abs(signed_shift)
        
        # Attach to the record
        r["delta_length"] = delta_len
        r["number_of_changes_computed"] = n_changes
        r["signed_shift"] = signed_shift
        r["abs_drift"] = abs_drift
        
        attacks[r["attack_type"]].append(r)
        delta_length_groups[delta_len].append(r)
        changes_groups[n_changes].append(r)
        
    print("--- 1. DRIFT BY ATTACK FAMILY ---")
    for atype in ["substitution", "insertion", "deletion", "rearrangement"]:
        group = attacks[atype]
        if not group: continue
        abs_drifts = [r["abs_drift"] for r in group]
        signed_shifts = [r["signed_shift"] for r in group]
        changes = [r["number_of_changes_computed"] for r in group]
        
        print(f"Attack: {atype.upper()} (n={len(group)})")
        print(f"  Mean Delta Length: {np.mean([r['delta_length'] for r in group]):.2f}")
        print(f"  Mean Number of Changes: {np.mean(changes):.2f}")
        print(f"  Mean Absolute Drift: {np.mean(abs_drifts):.2f} K")
        print(f"  Mean Signed Shift:   {np.mean(signed_shifts):.2f} K")
        
        positive_shifts = sum(1 for s in signed_shifts if s > 0)
        print(f"  Directional bias: {positive_shifts/len(group):.1%} positive shifts")
        print()
        
    print("--- 2. DRIFT BY DELTA LENGTH ---")
    for dlen in sorted(delta_length_groups.keys()):
        group = delta_length_groups[dlen]
        abs_drifts = [r["abs_drift"] for r in group]
        print(f"Delta Length = {dlen:>2} (n={len(group):>3}): Mean Absolute Drift = {np.mean(abs_drifts):>6.2f} K")
        
    length_preserving = [r["abs_drift"] for r in valid_records if r["delta_length"] == 0]
    length_changing = [r["abs_drift"] for r in valid_records if r["delta_length"] != 0]
    print(f"\nLength-Preserving (Delta=0, n={len(length_preserving)}): Mean Drift = {np.mean(length_preserving):.2f} K")
    print(f"Length-Changing   (Delta!=0, n={len(length_changing)}): Mean Drift = {np.mean(length_changing):.2f} K")
    print()
    
    print("--- 3. DRIFT BY NUMBER OF CHANGES (MAGNITUDE) ---")
    for n_changes in sorted(changes_groups.keys()):
        group = changes_groups[n_changes]
        abs_drifts = [r["abs_drift"] for r in group]
        print(f"Changes = {n_changes:>2} (n={len(group):>3}): Mean Absolute Drift = {np.mean(abs_drifts):>6.2f} K")
        
if __name__ == "__main__":
    analyze_attribution()
