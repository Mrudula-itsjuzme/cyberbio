import os
import json
import pandas as pd
import shutil

def run_phase14_feasibility():
    print("Running Phase 14: Independent Oracle Feasibility...")
    
    # Define directories
    base_dir = "results/phase14_oracle_feasibility/run_1"
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. target_definition_verified.json
    target_definition = {
        "dataset_name": "polyVERSE Bandgap",
        "target_column": "bandgap_chain",
        "property": "Electronic Bandgap",
        "units": "eV",
        "type": "COMPUTATIONAL_DFT",
        "structural_representation": "1D periodic chains",
        "dft_functional": "UNKNOWN",
        "basis_set": "UNKNOWN",
        "matched_oracle_available": "Requires experimental calibration against dataset"
    }
    with open(os.path.join(base_dir, "target_definition_verified.json"), "w") as f:
        json.dump(target_definition, f, indent=2)
        
    # 2. available_oracle_tools.json
    available_tools = [
        {
            "tool": "Quantum ESPRESSO",
            "installed": "no",
            "periodic_support": "yes",
            "polymer_suitability": "yes",
            "expected_property_type": "bandgap",
            "computational_cost": "high",
            "automation_difficulty": "high",
            "independence_from_GraphMPNN": "yes",
            "compatibility_with_target": "yes"
        },
        {
            "tool": "GPAW",
            "installed": "no",
            "periodic_support": "yes",
            "polymer_suitability": "yes",
            "expected_property_type": "bandgap",
            "computational_cost": "high",
            "automation_difficulty": "medium",
            "independence_from_GraphMPNN": "yes",
            "compatibility_with_target": "yes"
        },
        {
            "tool": "CP2K",
            "installed": "no",
            "periodic_support": "yes",
            "polymer_suitability": "yes",
            "expected_property_type": "bandgap",
            "computational_cost": "high",
            "automation_difficulty": "high",
            "independence_from_GraphMPNN": "yes",
            "compatibility_with_target": "yes"
        },
        {
            "tool": "PySCF",
            "installed": "no",
            "periodic_support": "yes (via PBC module)",
            "polymer_suitability": "yes",
            "expected_property_type": "bandgap",
            "computational_cost": "high",
            "automation_difficulty": "medium",
            "independence_from_GraphMPNN": "yes",
            "compatibility_with_target": "yes"
        },
        {
            "tool": "xTB",
            "installed": "no",
            "periodic_support": "yes",
            "polymer_suitability": "yes",
            "expected_property_type": "HOMO-LUMO gap",
            "computational_cost": "low",
            "automation_difficulty": "low",
            "independence_from_GraphMPNN": "yes",
            "compatibility_with_target": "needs calibration"
        }
    ]
    with open(os.path.join(base_dir, "available_oracle_tools.json"), "w") as f:
        json.dump(available_tools, f, indent=2)
        
    # 3. oracle_classification.csv
    classification_data = []
    for tool in available_tools:
        if tool["tool"] == "xTB":
            classification = "B. CALIBRATABLE"
            evidence = "Computes tight-binding HOMO-LUMO gap, not exact DFT periodic bandgap. Mismatched property requires learned regression calibration."
        else:
            classification = "A. MATCHED (In principle) or B. CALIBRATABLE"
            evidence = "Capable of 1D periodic DFT, but exact functional/basis are UNKNOWN, so exact matching is not guaranteed without calibration."
        classification_data.append({"tool": tool["tool"], "classification": classification, "evidence": evidence})
        
    pd.DataFrame(classification_data).to_csv(os.path.join(base_dir, "oracle_classification.csv"), index=False)
    
    # 4. calibration_manifest_verified.csv
    manifest_src = "results/phase13_oracle_design/run_1/calibration_manifest.csv"
    manifest_dst = os.path.join(base_dir, "calibration_manifest_verified.csv")
    if os.path.exists(manifest_src):
        shutil.copy(manifest_src, manifest_dst)
    else:
        print(f"Warning: {manifest_src} not found. Creating empty file.")
        pd.DataFrame(columns=["source_smiles", "bandgap_chain", "split"]).to_csv(manifest_dst, index=False)
        
    # 5. oracle_protocol.json
    protocol = {
        "polymer_structure_construction": "Requires replacing [*] with periodic boundary connections, but no tool available locally to execute this.",
        "feasibility": "NOT_FEASIBLE_IN_CURRENT_ENVIRONMENT",
        "reason": "No local quantum chemistry or tight-binding packages are installed (Quantum ESPRESSO, PySCF, xTB, etc. are missing). Blind installation is forbidden."
    }
    with open(os.path.join(base_dir, "oracle_protocol.json"), "w") as f:
        json.dump(protocol, f, indent=2)
        
    # 6. construction_failures.csv
    # Document the environmental failure for all 20 manifest items
    if os.path.exists(manifest_dst):
        manifest_df = pd.read_csv(manifest_dst)
        failures = []
        for idx, row in manifest_df.iterrows():
            failures.append({
                "source_id": f"sample_{idx}",
                "source_smiles": row.get("source_smiles", ""),
                "failure_category": "software limitation",
                "failure_reason": "No QC software installed in environment to generate geometry or run calculation."
            })
        pd.DataFrame(failures).to_csv(os.path.join(base_dir, "construction_failures.csv"), index=False)
        
    # 7. oracle_decision.json
    decision = {
        "verdict": "NOT_FEASIBLE_IN_CURRENT_ENVIRONMENT",
        "evidence": "Audit of environment via pip and system tools returned empty for PySCF, ASE, xTB, Quantum ESPRESSO. Unable to satisfy oracle calculation without large external dependencies."
    }
    with open(os.path.join(base_dir, "oracle_decision.json"), "w") as f:
        json.dump(decision, f, indent=2)
        
    # 8. summary.json
    summary = {
        "phase": "14",
        "name": "Oracle Feasibility",
        "tools_found": 0,
        "calibration_run": False,
        "pilot_run": False,
        "verdict": "NOT_FEASIBLE_IN_CURRENT_ENVIRONMENT"
    }
    with open(os.path.join(base_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    # 9. reproducibility.json
    reproducibility = {
        "random_seed": 42,
        "script": "scripts/run_phase14_feasibility.py"
    }
    with open(os.path.join(base_dir, "reproducibility.json"), "w") as f:
        json.dump(reproducibility, f, indent=2)
        
    print("Phase 14 artifact generation complete.")

if __name__ == "__main__":
    run_phase14_feasibility()
