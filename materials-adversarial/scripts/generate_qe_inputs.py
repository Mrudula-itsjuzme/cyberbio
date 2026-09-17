import os
import pandas as pd
from rdkit import Chem
import shutil
import json

def generate_qe_inputs():
    manifest_path = "hpc_oracle/manifests/surrogate_qc_pilot_9jobs.csv"
    if not os.path.exists(manifest_path):
        print(f"Missing {manifest_path}")
        return
        
    jobs_df = pd.read_csv(manifest_path)
    
    out_dir = "oracle_surrogate_qc_bundle/qe_inputs"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("oracle_surrogate_qc_bundle/structures", exist_ok=True)
    
    # Pre-execution guard for cutoffs (mock implementation, since we don't have UPFs)
    # Mark CUTOFF_COMPATIBILITY_UNVERIFIED
    with open("oracle_surrogate_qc_bundle/CUTOFF_COMPATIBILITY_UNVERIFIED", "w") as f:
        f.write("True\n")
        
    for idx, row in jobs_df.iterrows():
        job_id = row['job_id']
        sdf_path = row['structure_path']
        
        if not os.path.exists(sdf_path):
            print(f"Structure not found: {sdf_path}")
            continue
            
        # Copy structure
        dest_sdf = f"oracle_surrogate_qc_bundle/structures/{os.path.basename(sdf_path)}"
        shutil.copy(sdf_path, dest_sdf)
        
        # Read geometry
        mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
        if mol is None:
            print(f"Failed to load {sdf_path}")
            continue
            
        conf = mol.GetConformer()
        atoms = mol.GetAtoms()
        
        # Determine cell size (bounding box + 15 A padding)
        min_x = min([conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())])
        max_x = max([conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())])
        min_y = min([conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())])
        max_y = max([conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())])
        min_z = min([conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())])
        max_z = max([conf.GetAtomPosition(i).z for i in range(mol.GetNumAtoms())])
        
        cell_x = (max_x - min_x) + 15.0
        cell_y = (max_y - min_y) + 15.0
        cell_z = (max_z - min_z) + 15.0
        
        # Count elements
        element_types = {}
        for atom in atoms:
            sym = atom.GetSymbol()
            if sym not in element_types:
                element_types[sym] = atom.GetMass()
                
        # Generate QE input
        out_in = f"{out_dir}/{job_id}.in"
        with open(out_in, "w") as f:
            f.write(f"&CONTROL\n")
            f.write(f"  calculation = 'scf',\n")
            f.write(f"  prefix = '{job_id}',\n")
            f.write(f"  pseudo_dir = './pseudo',\n")
            f.write(f"  outdir = './out',\n")
            f.write(f"/\n")
            f.write(f"&SYSTEM\n")
            f.write(f"  ibrav = 8,\n") # Orthorhombic
            f.write(f"  A = {cell_x:.4f},\n")
            f.write(f"  B = {cell_y:.4f},\n")
            f.write(f"  C = {cell_z:.4f},\n")
            f.write(f"  nat = {mol.GetNumAtoms()},\n")
            f.write(f"  ntyp = {len(element_types)},\n")
            f.write(f"  ecutwfc = 40.0,\n")
            f.write(f"  ecutrho = 320.0,\n") # typically 8x ecutwfc for ultrasoft/PAW
            f.write(f"  occupations = 'smearing',\n")
            f.write(f"  smearing = 'gaussian',\n")
            f.write(f"  degauss = 0.01,\n")
            # Assumed non-magnetic, charge 0
            f.write(f"/\n")
            f.write(f"&ELECTRONS\n")
            f.write(f"  conv_thr = 1.0e-6,\n")
            f.write(f"/\n")
            f.write(f"ATOMIC_SPECIES\n")
            for sym, mass in element_types.items():
                f.write(f"  {sym} {mass:.3f} {sym}_sssp_eff_1.2.upf\n")
            
            f.write(f"ATOMIC_POSITIONS (angstrom)\n")
            for i, atom in enumerate(atoms):
                pos = conf.GetAtomPosition(i)
                f.write(f"  {atom.GetSymbol()}  {pos.x:.5f}  {pos.y:.5f}  {pos.z:.5f}\n")
                
            f.write(f"K_POINTS (gamma)\n")
            
    print("Generated QE inputs")
    
if __name__ == "__main__":
    generate_qe_inputs()
