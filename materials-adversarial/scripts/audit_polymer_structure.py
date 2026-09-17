import os
import json
import pandas as pd
import sys
from pathlib import Path
from rdkit import Chem

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.materials_adv.domain.chemistry.polymer_repeat import PolymerRepeatGraph

def audit_structures():
    out_dir = Path("results/polymer_structure_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pilot Readiness
    pilot_path = "hpc_oracle/manifests/calibration_pilot.csv"
    if os.path.exists(pilot_path):
        pilot_df = pd.read_csv(pilot_path)
        
        readiness_records = []
        for row in pilot_df.itertuples():
            smiles = row.polymer_representation
            graph = PolymerRepeatGraph(smiles)
            analysis = graph.analyze()
            
            readiness_records.append({
                "pilot_id": row.pilot_id,
                "source_id": row.source_id,
                "smiles": smiles,
                "is_unambiguous_2d": analysis["is_unambiguous"],
                "oligomer_2_valid": analysis["oligomer_2_valid"],
                "oligomer_3_valid": analysis["oligomer_3_valid"],
                "unique_3d_recoverable": analysis["unique_3d_periodic_recoverable"],
                "warnings": "; ".join(analysis["warnings"])
            })
            
        pd.DataFrame(readiness_records).to_csv(out_dir / "pilot_structure_readiness.csv", index=False)
        print("Generated pilot structure readiness report.")

    # 2. General Dataset Audit
    raw_path = "data/raw/bandgap_chain.csv"
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
        sample = raw_df.head(50) # Representative sample
        
        audit_records = []
        for row in sample.itertuples():
            smiles = row.polymer_string if hasattr(row, 'polymer_string') else row.SMILES if hasattr(row, 'SMILES') else None
            
            if smiles is None:
                # check if original_representation exists
                smiles = getattr(row, 'original_representation', getattr(row, 'polymer', None))
                
            if smiles is None and hasattr(row, 'Index'): # sometimes just columns 1 is string
                smiles = row[1] if isinstance(row[1], str) else row[2]
            
            if not isinstance(smiles, str):
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                wildcards = [a for a in mol.GetAtoms() if a.GetSymbol() == '*']
                num_wildcards = len(wildcards)
                
                neighbors = []
                bond_types = []
                for w in wildcards:
                    if w.GetDegree() > 0:
                        nb = w.GetNeighbors()[0]
                        neighbors.append(nb.GetSymbol())
                        bond_types.append(mol.GetBondBetweenAtoms(w.GetIdx(), nb.GetIdx()).GetBondType().name)
                        
                aromaticity = any(a.GetIsAromatic() for a in mol.GetAtoms())
                formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
                stereochem = len(Chem.FindMolChiralCenters(mol, force=True)) > 0
                
                audit_records.append({
                    "raw_string": smiles,
                    "num_wildcards": num_wildcards,
                    "wildcard_neighbors": ", ".join(neighbors),
                    "bond_types": ", ".join(bond_types),
                    "direction_encoded": "No", # SMILES doesn't encode HT/HH without explicit tags
                    "aromaticity": aromaticity,
                    "formal_charge": formal_charge,
                    "stereochemistry": stereochem,
                    "repeat_ambiguity": "Unambiguous 2D" if num_wildcards == 2 else "Ambiguous"
                })
                
        if audit_records:
            pd.DataFrame(audit_records).to_csv(out_dir / "representation_audit.csv", index=False)
            print("Generated representation audit report.")

    # 3. Provenance Findings
    provenance = {
        "finding_1": {
            "type": "DIRECTLY_VERIFIED",
            "finding": "No prebuilt periodic geometry files (.cif, .xyz, POSCAR) exist in the local dataset repository."
        },
        "finding_2": {
            "type": "DIRECTLY_VERIFIED",
            "finding": "No automated 3D builder or geometry generation scripts exist in the repository."
        },
        "finding_3": {
            "type": "UNKNOWN",
            "finding": "Exact DFT functional, basis set, cutoff, and k-point mesh used by the original authors are unknown."
        },
        "finding_4": {
            "type": "INFERRED",
            "finding": "Since exact input structures are missing and calculation parameters are missing, only a SURROGATE_STRUCTURE_PROTOCOL can be constructed."
        }
    }
    
    with open(out_dir / "provenance_findings.json", "w") as f:
        json.dump(provenance, f, indent=4)
        
    print("Generated provenance findings report.")

if __name__ == "__main__":
    audit_structures()
