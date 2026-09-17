import hashlib
from typing import Dict, Any, Tuple, Optional
from rdkit import Chem

class PolymerRepeatGraph:
    """
    Constructs a 2D topology/graph representing a polymer repeat unit,
    validating exactly two attachment points and allowing n-mer oligomer 
    sanity checks. No 3D coordinates are generated.
    """
    
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.is_valid = self.mol is not None
        self.attachment_indices = []
        self.warnings = []
        
        if self.is_valid:
            self._find_attachments()
            
    def _find_attachments(self):
        for atom in self.mol.GetAtoms():
            if atom.GetSymbol() == '*':
                self.attachment_indices.append(atom.GetIdx())
                
        if len(self.attachment_indices) != 2:
            self.warnings.append(f"Expected exactly 2 attachment points, found {len(self.attachment_indices)}.")
            
    @property
    def is_unambiguous(self) -> bool:
        return self.is_valid and len(self.attachment_indices) == 2

    def build_oligomer(self, n: int) -> Optional[Chem.Mol]:
        """
        Builds a finite oligomer graph of length n purely for connectivity validation.
        Connects the two attachment points.
        """
        if not self.is_unambiguous:
            return None
            
        # For an n-mer, we create n copies and link them head-to-tail.
        # This is a conceptual topological mapping.
        
        # We assume attachment_indices[0] is 'head' and attachment_indices[1] is 'tail'
        # To connect head of n to tail of n+1:
        
        combo = Chem.RWMol(self.mol)
        
        for i in range(n - 1):
            next_mol = self.mol
            
            # Combine
            offset = combo.GetNumAtoms()
            combo = Chem.CombineMols(combo, next_mol)
            combo = Chem.RWMol(combo)
            
            # The tail of the current is at attachment_indices[1] + (i * orig_size)
            # But wait, when we remove atoms, indices shift. 
            # A safer way: tag atoms with isotopes or properties.
            
        # A simpler robust way using RDKit reaction or custom tagging
        return self._build_oligomer_robust(n)

    def _build_oligomer_robust(self, n: int) -> Optional[Chem.Mol]:
        if not self.is_unambiguous:
            return None
            
        # Tag attachments
        mol_copy = Chem.RWMol(self.mol)
        head_idx, tail_idx = self.attachment_indices
        
        # Get neighbors
        head_neighbor = mol_copy.GetAtomWithIdx(head_idx).GetNeighbors()[0].GetIdx()
        tail_neighbor = mol_copy.GetAtomWithIdx(tail_idx).GetNeighbors()[0].GetIdx()
        
        # What is the bond order?
        head_bond = mol_copy.GetBondBetweenAtoms(head_idx, head_neighbor).GetBondType()
        tail_bond = mol_copy.GetBondBetweenAtoms(tail_idx, tail_neighbor).GetBondType()
        
        if head_bond != tail_bond:
            self.warnings.append("Mismatched bond types at attachment points.")
            
        combo = Chem.RWMol(self.mol)
        orig_size = combo.GetNumAtoms()
        
        current_tail_neighbor = tail_neighbor
        
        for i in range(1, n):
            combo = Chem.CombineMols(combo, self.mol)
            combo = Chem.RWMol(combo)
            
            next_head_neighbor = head_neighbor + (i * orig_size)
            next_tail_neighbor = tail_neighbor + (i * orig_size)
            
            # Add bond
            combo.AddBond(current_tail_neighbor, next_head_neighbor, head_bond)
            current_tail_neighbor = next_tail_neighbor
            
        # Now remove all '*' atoms
        atoms_to_remove = [atom.GetIdx() for atom in combo.GetAtoms() if atom.GetSymbol() == '*']
        for idx in sorted(atoms_to_remove, reverse=True):
            combo.RemoveAtom(idx)
            
        try:
            Chem.SanitizeMol(combo)
            return combo.GetMol()
        except Exception as e:
            self.warnings.append(f"Sanitization failed: {e}")
            return None

    def get_topology_hash(self) -> str:
        """Returns a hash of the topology (canonical SMILES with attachments stripped)"""
        if not self.is_unambiguous:
            return "INVALID"
        mol_copy = Chem.RWMol(self.mol)
        atoms_to_remove = [atom.GetIdx() for atom in mol_copy.GetAtoms() if atom.GetSymbol() == '*']
        for idx in sorted(atoms_to_remove, reverse=True):
            mol_copy.RemoveAtom(idx)
        try:
            Chem.SanitizeMol(mol_copy)
            smi = Chem.MolToSmiles(mol_copy)
            return hashlib.md5(smi.encode()).hexdigest()
        except:
            return "ERROR"
            
    def analyze(self) -> Dict[str, Any]:
        """Provides a structured summary of the representation sufficiency."""
        
        # Test oligomers
        oligomer_2 = self.build_oligomer(2)
        oligomer_3 = self.build_oligomer(3)
        
        return {
            "is_valid": self.is_valid,
            "attachment_point_count": len(self.attachment_indices),
            "is_unambiguous": self.is_unambiguous,
            "unique_2d_graph_recoverable": self.is_unambiguous and (oligomer_3 is not None),
            "unique_3d_periodic_recoverable": False, # Always False per constraints (needs conformational search/params)
            "unique_periodic_cell_recoverable": False,
            "topology_hash": self.get_topology_hash(),
            "warnings": self.warnings,
            "oligomer_2_valid": oligomer_2 is not None,
            "oligomer_3_valid": oligomer_3 is not None
        }
