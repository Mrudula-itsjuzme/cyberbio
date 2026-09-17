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
    """Graph featurisation for the GraphMPNN.

    ``strict`` controls what happens to a SMILES string that does not parse.

    * ``strict=False`` (default, historical behaviour): the dataset silently walks
      forward and returns a DIFFERENT molecule's graph. That is acceptable for clean
      training data, where it never fires, but it is dangerous for adversarial
      evaluation: a malformed attack candidate would be scored as if it were valid.
    * ``strict=True``: raise. Adversarial / evaluation code must use this, so an
      invalid candidate is rejected instead of silently replaced.

    See docs/FRAMEWORK_V2_BENCHMARK_AUDIT.md section 8.
    """

    def __init__(self, df, max_nodes=256, strict: bool = False):
        self.smiles = df["original_representation"].tolist()
        self.targets = df["property_value"].astype(float).tolist()
        self.max_nodes = max_nodes
        self.strict = strict
        self.skipped_unparseable = 0

    def __len__(self):
        return len(self.smiles)

    def _first_parseable_from(self, start: int):
        """Legacy fallback: locate the next parseable row, or fail loudly."""
        for offset in range(1, len(self) + 1):
            nxt = (start + offset) % len(self)
            if Chem.MolFromSmiles(self.smiles[nxt]) is not None:
                self.skipped_unparseable += offset
                return nxt
        raise ValueError(
            "GraphDataset: no parseable SMILES in this dataset; refusing to loop forever"
        )

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        y = self.targets[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            if getattr(self, "strict", False):
                raise ValueError(
                    f"GraphDataset(strict=True): row {idx} does not parse as SMILES: {smi!r}. "
                    "Adversarial evaluation must reject an invalid candidate, never "
                    "silently score a different molecule."
                )
            return self.__getitem__(self._first_parseable_from(idx))
            
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
