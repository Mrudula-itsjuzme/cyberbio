import pandas as pd
import numpy as np
import os

def select_pilot():
    np.random.seed(42)
    processed_path = "data/processed/processed.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Could not find {processed_path}")
        
    df = pd.read_csv(processed_path)
    
    # Stratify by bandgap
    df['bg_bin'] = pd.qcut(df['property_value'], 3, labels=['low', 'medium', 'high'])
    
    selected_indices = []
    
    # Try to select simple structures (fewer heavy atoms as a proxy, or shorter SMILES length)
    df['smiles_len'] = df['original_representation'].apply(len)
    
    for category in ['low', 'medium', 'high']:
        subset = df[df['bg_bin'] == category].sort_values('smiles_len')
        # Pick 2-3 per category
        selected = subset.head(10).sample(n=3, random_state=42) 
        selected_indices.extend(selected.index)
        
    pilot_df = df.loc[selected_indices].copy()
    
    # Create the pilot structure
    records = []
    for i, row in enumerate(pilot_df.itertuples()):
        repr_str = row.original_representation
        num_attachments = repr_str.count('[*]')
        records.append({
            "pilot_id": f"pilot_{i}",
            "source_id": row.polymer_id,
            "polymer_representation": repr_str,
            "dataset_bandgap_eV": row.property_value,
            "structural_category": getattr(row, 'bg_bin', 'unknown'),
            "selection_reason": f"{getattr(row, 'bg_bin')} bandgap, simple SMILES",
            "representation": "1D wildcard string",
            "attachment_point_count": num_attachments,
            "attachment_atoms": "[*]",
            "valence_check": "assume complete",
            "charge": 0,
            "multiplicity": 1,
            "periodic_structure_available": "No",
            "geometry_available": "No",
            "ambiguity_notes": "Requires periodic 1D building",
            "construction_blocker": "No 3D builder implemented, no prebuilt geometries",
            "structure_status": "BLOCKED",
            "oracle_job_status": "PENDING_STRUCTURE"
        })
        
    pilot_out = pd.DataFrame(records)
    out_dir = "hpc_oracle/manifests"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "calibration_pilot.csv")
    pilot_out.to_csv(out_path, index=False)
    print(f"Selected {len(pilot_out)} pilot samples and saved to {out_path}")

if __name__ == "__main__":
    select_pilot()
