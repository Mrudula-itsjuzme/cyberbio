#!/usr/bin/env python3
"""Oracle candidate selection utility.

Extracts a diverse set of chemistry-changing adversarial candidates from the
frozen paired benchmark results for future evaluation by a physical oracle.

This script DOES NOT claim the candidates are physically adversarial. It only
selects them so they *can* be evaluated by an oracle. It outputs a CSV manifest
along with provenance metadata.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# RDKit must fail closed
try:
    from rdkit import Chem
except ImportError:
    print("FATAL: RDKit is required to run oracle candidate selection safely. "
          "It must be used to deduplicate by canonical smiles.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("select_oracle_candidates")

def parse_args():
    parser = argparse.ArgumentParser(description="Select candidates for physical oracle evaluation")
    parser.add_argument("--benchmark-file", default="results/phase2_paired_benchmark/clean_attack_records.jsonl")
    parser.add_argument("--out-dir", default="results/oracle_candidates")
    parser.add_argument("--max-candidates", type=int, default=500, help="Maximum candidates to select")
    parser.add_argument("--min-drift", type=float, default=0.2, help="Minimum absolute drift (eV) to consider")
    return parser.parse_args()

def main():
    args = parse_args()
    bench_file = Path(args.benchmark_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not bench_file.exists():
        log.error(f"Benchmark file not found: {bench_file}")
        sys.exit(1)

    log.info(f"Reading records from {bench_file}")
    
    candidates = []
    seen_canonical = set()
    
    with open(bench_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            
            if rec.get("validity_status") != "valid":
                continue
            
            # Skip non-chemistry changing attacks
            attack_type = rec.get("attack_type", "")
            if attack_type in ["randomized_smiles", "randomization", "None", ""]:
                continue
                
            drift = rec.get("signed_prediction_drift")
            if drift is None:
                continue
                
            abs_drift = rec.get("absolute_prediction_drift", abs(drift))
            if abs_drift < args.min_drift:
                continue
                
            adv_smiles = rec.get("adversarial_representation")
            orig_smiles = rec.get("original_representation")
            if not adv_smiles or not orig_smiles:
                continue
                
            # Fail closed: must be parsed by RDKit
            mol = Chem.MolFromSmiles(adv_smiles)
            if not mol:
                continue
            mol_orig = Chem.MolFromSmiles(orig_smiles)
            if not mol_orig:
                continue
                
            can_adv = Chem.MolToSmiles(mol, canonical=True)
            if can_adv in seen_canonical:
                continue
                
            can_orig = Chem.MolToSmiles(mol_orig, canonical=True)
            if can_orig == can_adv:
                # Chemistry did not actually change!
                continue
                
            seen_canonical.add(can_adv)
            
            candidates.append({
                "source_id": rec.get("sample_id"),
                "source_smiles": orig_smiles,
                "candidate_smiles": adv_smiles,
                "canonical_candidate_smiles": can_adv,
                "attack_family": attack_type,
                "model_drift_eV": abs_drift,
                "original_prediction": rec.get("original_prediction"),
                "candidate_prediction": rec.get("adversarial_prediction"),
                "attack_budget": rec.get("attack_budget", rec.get("number_of_changes", 1)),
                "source_artifact": str(bench_file),
            })

    log.info(f"Found {len(candidates)} unique RDKit-valid, chemistry-changed candidates above {args.min_drift} eV drift.")
    if not candidates:
        log.warning("No candidates met the criteria.")
        sys.exit(0)
        
    df = pd.DataFrame(candidates)
    
    # Stratified sampling
    # 1. Stratify by attack family
    # 2. Within attack family, bin by drift quantiles
    # 3. Limit per source molecule to prevent dominance
    
    # Compute drift quantiles
    df["drift_q"] = pd.qcut(df["model_drift_eV"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    
    selected_indices = []
    source_counts = defaultdict(int)
    max_per_source = max(1, args.max_candidates // (df["source_id"].nunique() or 1) * 2) # Allow some clustering but not total dominance
    if max_per_source < 3: max_per_source = 3
    
    # Try to pick evenly across (attack_family, drift_q)
    groups = df.groupby(["attack_family", "drift_q"], observed=True)
    
    # Round-robin selection across groups
    group_iters = {name: group.sample(frac=1, random_state=42).index.tolist() for name, group in groups}
    
    while len(selected_indices) < args.max_candidates and any(group_iters.values()):
        for name in list(group_iters.keys()):
            if not group_iters[name]:
                del group_iters[name]
                continue
            
            # Pop next candidate in this group
            idx = group_iters[name].pop(0)
            src_id = df.loc[idx, "source_id"]
            
            if source_counts[src_id] < max_per_source:
                selected_indices.append(idx)
                source_counts[src_id] += 1
                
            if len(selected_indices) >= args.max_candidates:
                break
                
    final_df = df.loc[selected_indices].drop(columns=["drift_q"])
    final_df.sort_values(by=["attack_family", "model_drift_eV"], ascending=[True, False], inplace=True)
    
    csv_path = out_dir / "candidates.csv"
    final_df.to_csv(csv_path, index=False)
    log.info(f"Wrote {len(final_df)} stratified candidates to {csv_path}")
    
    # Save provenance metadata
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(bench_file),
        "total_candidates_selected": len(final_df),
        "selection_criteria": {
            "min_drift_eV": args.min_drift,
            "must_be_rdkit_valid": True,
            "must_be_chemistry_changing": True,
            "deduplicated_by_canonical_smiles": True,
            "stratified_by": ["attack_family", "drift_quantiles"]
        }
    }
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from materials_adv.utils.provenance import capture, save
        prov = capture(seed=None, extra=meta)
        save(prov, out_dir / "provenance.json")
    except Exception as e:
        log.warning(f"Could not capture full provenance: {e}")
        with open(out_dir / "provenance.json", "w") as f:
            json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
