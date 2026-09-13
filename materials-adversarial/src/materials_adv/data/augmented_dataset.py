import random
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from .tokenizer import tokenize

class AugmentedSMILESDataset(Dataset):
    def __init__(self, df, vocab, max_seq_len=256, augment_prob=0.5):
        self.smiles = df["original_representation"].tolist()
        self.targets = df["property_value"].astype(float).tolist()
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.augment_prob = augment_prob
        
    def __len__(self):
        return len(self.smiles)
        
    def _encode(self, s: str) -> torch.Tensor:
        tokens = tokenize(s)
        # 1-based indexing for vocab to reserve 0 for padding
        vocab_map = {t: i for i, t in enumerate(self.vocab)}
        ids = [vocab_map.get(t, vocab_map.get("<unk>", 1)) + 1 for t in tokens]
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
        padded = ids + [0] * (self.max_seq_len - len(ids))
        return torch.tensor(padded, dtype=torch.long)
        
    def __getitem__(self, idx):
        smi = self.smiles[idx]
        y = self.targets[idx]
        
        if random.random() < self.augment_prob:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Randomize representations: do_random=True
                new_smi = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
                if new_smi:
                    smi = new_smi
                    
        return {
            "x": self._encode(smi),
            "target": torch.tensor([y], dtype=torch.float32),
            "smiles": smi
        }
