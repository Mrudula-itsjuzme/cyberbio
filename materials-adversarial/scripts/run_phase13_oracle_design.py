import json
import csv
import os
import random
import hashlib
from datetime import datetime

# Determinism
random.seed(42)

def main():
    run_id = "run_1"
    output_dir = f"results/phase13_oracle_design/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Provenance
    target_provenance = {
        "dataset": "polyVERSE (Ramprasad Group)",
        "property": "Bandgap",
        "units": "eV",
        "type": "Computational",
        "method": "DFT",
        "functional": "UNKNOWN (Likely PBE or HSE06 based on literature)",
        "basis_set": "UNKNOWN (Likely PAW based on literature)",
        "periodic_vs_molecular": "Periodic (1D chain)",
        "geometry_optimization": "UNKNOWN",
        "oligomer_representation": "UNKNOWN",
        "definition": "Electronic bandgap (eV) of a 1D periodic polymer chain computed via DFT."
    }
    with open(os.path.join(output_dir, "target_provenance.json"), "w") as f:
        json.dump(target_provenance, f, indent=2)

    # 2. Oracle Options
    oracle_options = [
        {"class": "MATCHED ORACLE", "description": "1D Periodic DFT calculation matching original dataset protocol (e.g. VASP with HSE06).", "fidelity": "High", "cost": "Very High", "automation": "Difficult"},
        {"class": "CALIBRATABLE ORACLE", "description": "Semi-empirical or ML-DFT (e.g. xTB, MACE, ALIGNN) calibrated to dataset.", "fidelity": "Medium", "cost": "Low", "automation": "High"},
        {"class": "MISMATCHED ORACLE", "description": "Molecular HOMO-LUMO gap of capped oligomers.", "fidelity": "Low", "cost": "Medium", "automation": "High"}
    ]
    with open(os.path.join(output_dir, "oracle_options.csv"), "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["class", "description", "fidelity", "cost", "automation"])
        writer.writeheader()
        writer.writerows(oracle_options)

    # 3. Polymer Quantum Representation
    rep_md = """# Polymer Quantum Representation Strategy
To evaluate a polymer sequence `[*]...[*]` via DFT:
1. **Periodic Strategy (Recommended for matched oracle):** Map `[*]` attachment points to periodic boundary conditions along the 1D chain axis.
2. **Oligomer Strategy (Fallback):** Construct $N$-mer oligomers (e.g., $N=3, 5$) and cap attachment points with Hydrogen (`[H]`) or Methyl (`[CH3]`) groups. Requires extrapolating HOMO-LUMO gap to $N \\to \\infty$.
"""
    with open(os.path.join(output_dir, "polymer_quantum_representation.md"), "w") as f:
        f.write(rep_md)

    # 4. Calibration Manifest
    # Select 20 deterministic examples from raw data
    raw_data_path = "data/raw/bandgap_chain.csv"
    calibration_manifest = []
    if os.path.exists(raw_data_path):
        with open(raw_data_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Sample across range
            rows = sorted(rows, key=lambda x: float(x["bandgap_chain"]))
            # Select roughly uniformly
            indices = [int(i * (len(rows)-1) / 19) for i in range(20)]
            for i in indices:
                calibration_manifest.append(rows[i])
    
    with open(os.path.join(output_dir, "calibration_manifest.csv"), "w", newline='') as f:
        if calibration_manifest:
            writer = csv.DictWriter(f, fieldnames=calibration_manifest[0].keys())
            writer.writeheader()
            writer.writerows(calibration_manifest)

    # 5. Candidate Shortlist from Phase 12B
    phase12b_traj_file = "results/phase12b_adaptive_search/1789331655/attack_trajectories.jsonl"
    shortlist = []
    source_candidate_pairs = []
    
    if os.path.exists(phase12b_traj_file):
        with open(phase12b_traj_file, "r") as f:
            candidates = [json.loads(line) for line in f]
            
            # Filter distinct ones, non-zero edits, valid edits <= 3
            candidates = [c for c in candidates if c.get("edit_count", 0) > 0 and c.get("edit_count", 0) <= 3 and "drift_from_original" in c]
            
            # High GraphMPNN drift
            c_g = [c for c in candidates if c["target_model"] == "GraphMPNN"]
            high_g = sorted(c_g, key=lambda x: abs(x["drift_from_original"]), reverse=True)[:5]
            
            # High Transformer drift
            c_t = [c for c in candidates if c["target_model"] == "Transformer"]
            high_s = sorted(c_t, key=lambda x: abs(x["drift_from_original"]), reverse=True)[:5]
            
            # Moderate drift (mix)
            mod = sorted(candidates, key=lambda x: abs(x["drift_from_original"]))
            moderate = mod[len(mod)//2 : len(mod)//2 + 5]
            
            # Different edit counts (1, 2, 3)
            e1 = [c for c in candidates if c["edit_count"] == 1][:2]
            e2 = [c for c in candidates if c["edit_count"] == 2][:2]
            e3 = [c for c in candidates if c["edit_count"] == 3][:1]
            
            # Combine unique
            seen = set()
            for c in high_g + high_s + moderate + e1 + e2 + e3:
                cand_smiles = c["proposed_candidate"]
                if cand_smiles not in seen and len(shortlist) < 20:
                    shortlist.append(c)
                    seen.add(cand_smiles)
                    
                    graph_drift = c["drift_from_original"] if c["target_model"] == "GraphMPNN" else "UNKNOWN"
                    trans_drift = c["drift_from_original"] if c["target_model"] == "Transformer" else "UNKNOWN"
                    
                    source_candidate_pairs.append({
                        "source_smiles": c["source_smiles"],
                        "candidate_smiles": cand_smiles,
                        "edit_count": c["edit_count"],
                        "graph_drift": graph_drift,
                        "transformer_drift": trans_drift
                    })
    
    # We only care about a subset of keys for the shortlist
    shortlist_filtered = []
    for c in shortlist:
        shortlist_filtered.append({
            "source_smiles": c["source_smiles"],
            "candidate_smiles": c["proposed_candidate"],
            "edit_count": c["edit_count"],
            "target_model": c["target_model"],
            "drift_from_original": c["drift_from_original"]
        })

    with open(os.path.join(output_dir, "candidate_shortlist.csv"), "w", newline='') as f:
        if shortlist_filtered:
            writer = csv.DictWriter(f, fieldnames=shortlist_filtered[0].keys())
            writer.writeheader()
            writer.writerows(shortlist_filtered)
            
    with open(os.path.join(output_dir, "source_candidate_pairs.csv"), "w", newline='') as f:
        if source_candidate_pairs:
            writer = csv.DictWriter(f, fieldnames=["source_smiles", "candidate_smiles", "edit_count", "graph_drift", "transformer_drift"])
            writer.writeheader()
            writer.writerows(source_candidate_pairs)

    # 6. Oracle Protocol
    protocol = {
        "step_1": "Calibration run on calibration_manifest.csv",
        "step_2": "Verify MAE < 0.2 eV against polyVERSE dataset labels",
        "step_3": "Execute DFT on source_smiles and candidate_smiles for each pair",
        "step_4": "Compute Delta_T = T(candidate) - T(source)",
        "step_5": "Report E_G = |Delta_G - Delta_T| and E_S = |Delta_S - Delta_T|"
    }
    with open(os.path.join(output_dir, "oracle_protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2)

    # 7. Summary & Reproducibility
    summary = {
        "status": "SUCCESS",
        "phase": 13,
        "oracle_calculations_performed": False,
        "calibration_manifest_size": len(calibration_manifest),
        "shortlist_size": len(shortlist)
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    reproducibility = {
        "timestamp": datetime.now().isoformat(),
        "phase12b_traj_hash": hashlib.sha256(open(phase12b_traj_file, 'rb').read()).hexdigest() if os.path.exists(phase12b_traj_file) else None
    }
    with open(os.path.join(output_dir, "reproducibility.json"), "w") as f:
        json.dump(reproducibility, f, indent=2)
        
    print(f"Phase 13 completed. Results in {output_dir}")

if __name__ == "__main__":
    main()
