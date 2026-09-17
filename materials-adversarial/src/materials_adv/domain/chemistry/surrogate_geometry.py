import numpy as np
import yaml
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from materials_adv.domain.chemistry.polymer_repeat import PolymerRepeatGraph

class SurrogateGeometryBuilder:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.mode = self.config.get("mode", "CAPPED_OLIGOMER")
        if self.mode not in ["CAPPED_OLIGOMER", "PERIODIC_CHAIN_SURROGATE"]:
            raise ValueError(f"Unknown mode {self.mode}")
            
    def _apply_capping(self, mol: Chem.Mol, rule: str) -> Optional[Chem.Mol]:
        """
        Replaces remaining '*' atoms with the capping rule (e.g. H).
        """
        rw_mol = Chem.RWMol(mol)
        
        cap_mol = Chem.MolFromSmiles(rule)
        if cap_mol is None:
            return None
            
        cap_atom_symbol = cap_mol.GetAtomWithIdx(0).GetSymbol()
        
        atoms_to_replace = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetSymbol() == '*']
        
        for idx in atoms_to_replace:
            rw_mol.GetAtomWithIdx(idx).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(cap_atom_symbol))
            
        try:
            Chem.SanitizeMol(rw_mol)
            return rw_mol.GetMol()
        except:
            return None
            
    def _optimize_conformers(self, mol: Chem.Mol, num_confs: int, seed: int, ff: str, max_steps: int) -> Tuple[Optional[Chem.Mol], Dict[str, Any]]:
        """
        Embeds the molecule, generates conformers, optimizes them, and selects the lowest energy.
        """
        mol_H = Chem.AddHs(mol)
        
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        
        cids = AllChem.EmbedMultipleConfs(mol_H, numConfs=num_confs, params=params)
        
        if len(cids) == 0:
            return None, {"error": "Embedding failed"}
            
        results = []
        for cid in cids:
            try:
                if ff == "MMFF94":
                    if AllChem.MMFFHasAllMoleculeParams(mol_H):
                        ff_opt = AllChem.MMFFGetMoleculeForceField(mol_H, AllChem.MMFFGetMoleculeProperties(mol_H), confId=cid)
                    else:
                        # Fallback
                        ff_opt = AllChem.UFFGetMoleculeForceField(mol_H, confId=cid)
                else:
                    ff_opt = AllChem.UFFGetMoleculeForceField(mol_H, confId=cid)
                    
                ff_opt.Initialize()
                ff_opt.Minimize(maxIts=max_steps)
                energy = ff_opt.CalcEnergy()
                results.append((cid, energy))
            except Exception as e:
                pass
                
        if not results:
             return None, {"error": "Optimization failed for all conformers"}
             
        # Sort by energy
        results.sort(key=lambda x: x[1])
        best_cid, best_energy = results[0]
        
        energies = [r[1] for r in results]
        
        uncertainty = {
            "num_conformers_generated": len(cids),
            "num_conformers_optimized": len(results),
            "energy_min": float(np.min(energies)),
            "energy_max": float(np.max(energies)),
            "energy_std": float(np.std(energies)),
            "selected_conformer_energy": float(best_energy),
            "force_field_used": ff
        }
        
        # Isolate best conformer
        best_mol = Chem.Mol(mol_H)
        best_mol.RemoveAllConformers()
        best_mol.AddConformer(mol_H.GetConformer(best_cid), assignId=True)
        
        return best_mol, uncertainty

    def build_surrogate(self, smiles: str, oligomer_length: int) -> Tuple[Optional[Chem.Mol], Dict[str, Any]]:
        if self.mode == "PERIODIC_CHAIN_SURROGATE":
            if self.config.get("cell_length") is None or self.config.get("vacuum_dimensions") is None:
                return None, {"error": "Periodic mode blocks without cell parameters"}
                
        graph = PolymerRepeatGraph(smiles)
        if not graph.is_unambiguous:
             return None, {"error": "Ambiguous 2D topology"}
             
        oligomer = graph.build_oligomer(oligomer_length)
        if oligomer is None:
             return None, {"error": "Failed to build topological oligomer"}
             
        capped = self._apply_capping(oligomer, self.config.get("capping_rule", "[H]"))
        if capped is None:
             return None, {"error": "Capping failed"}
             
        # Verify no wildcards left
        if any(a.GetSymbol() == '*' for a in capped.GetAtoms()):
             return None, {"error": "Wildcards remain after capping"}
             
        # 3D embedding
        best_mol, uncertainty = self._optimize_conformers(
            capped, 
            num_confs=self.config.get("num_conformers", 5),
            seed=self.config.get("embedding_seed", 42),
            ff=self.config.get("force_field", "MMFF94"),
            max_steps=self.config.get("max_optimization_steps", 1000)
        )
        
        if best_mol is None:
            return None, uncertainty
            
        uncertainty["oligomer_length"] = oligomer_length
        uncertainty["capping_rule"] = self.config.get("capping_rule", "[H]")
        uncertainty["topology_hash"] = graph.get_topology_hash()
        
        return best_mol, uncertainty
