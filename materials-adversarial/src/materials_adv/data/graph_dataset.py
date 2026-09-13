import torch
from torch.utils.data import Dataset
from rdkit import Chem
import numpy as np

def get_node_features(atom):
    """
    Compact fixed node feature set:
    - atomic number (float)
    - degree (float)
    - formal charge (float)
    - aromaticity (0 or 1)
    - hybridization (float)
    - hydrogen count (float)
    - wildcard/attachment-point flag (0 or 1)
    """
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(int(atom.GetIsAromatic())),
        float(atom.GetHybridization()),
        float(atom.GetTotalNumHs()),
        float(int(atom.GetAtomicNum() == 0))
    ]

def get_edge_features(bond):
    """
    Edge features:
    - bond type (1.0, 1.5, 2.0, 3.0)
    - aromatic flag
    - conjugation flag
    """
    bt = bond.GetBondTypeAsDouble()
    return [
        float(bt),
        float(int(bond.GetIsAromatic())),
        float(int(bond.GetIsConjugated()))
    ]

class GraphDataset(Dataset):
    def __init__(self, df, max_nodes=256):
        self.smiles = df["original_representation"].tolist()
        self.targets = df["property_value"].astype(float).tolist()
        self.max_nodes = max_nodes
        
    def __len__(self):
        return len(self.smiles)
        
    def __getitem__(self, idx):
        smi = self.smiles[idx]
        y = self.targets[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Fallback for invalid SMILES - should not happen in clean data
            return self.__getitem__((idx + 1) % len(self))
            
        n_atoms = mol.GetNumAtoms()
        x = np.zeros((self.max_nodes, 7), dtype=np.float32)
        adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        edge_attr = np.zeros((self.max_nodes, self.max_nodes, 3), dtype=np.float32)
        
        actual_nodes = min(n_atoms, self.max_nodes)
        
        for i, atom in enumerate(mol.GetAtoms()):
            if i >= actual_nodes: break
            x[i] = get_node_features(atom)
            
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i < actual_nodes and j < actual_nodes:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
                e_feat = get_edge_features(bond)
                edge_attr[i, j] = e_feat
                edge_attr[j, i] = e_feat
                
        # Self-loops
        for i in range(actual_nodes):
            adj[i, i] = 1.0
            
        mask = np.zeros((self.max_nodes,), dtype=bool)
        mask[:actual_nodes] = True
        
        return {
            "x": torch.tensor(x),
            "adj": torch.tensor(adj),
            "edge_attr": torch.tensor(edge_attr),
            "mask": torch.tensor(mask),
            "target": torch.tensor([y], dtype=torch.float32),
            "smiles": smi
        }
