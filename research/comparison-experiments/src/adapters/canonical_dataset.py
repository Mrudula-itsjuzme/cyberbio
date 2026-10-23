import sys
import json
from pathlib import Path
import pandas as pd

comp_exp_root = Path(__file__).resolve().parent.parent.parent
if str(comp_exp_root) not in sys.path:
    sys.path.insert(0, str(comp_exp_root))

repo_root = comp_exp_root.parent / "materials-adversarial"
if str(repo_root) not in sys.path and repo_root.exists():
    sys.path.insert(0, str(repo_root / "src"))

class CanonicalDatasetAdapter:
    def __init__(self, data_csv_path: str = None, frozen_bank_path: str = None):
        if data_csv_path is None:
            data_csv_path = repo_root / "data/processed/processed.csv"
        if frozen_bank_path is None:
            frozen_bank_path = comp_exp_root / "data/frozen_source_ids.json"
            
        self.df = pd.read_csv(data_csv_path)
        
        with open(frozen_bank_path, "r") as f:
            data = json.load(f)
            self.source_ids = data["source_ids"]
            
        # Ensure we have strings and assuming index matching or an 'id' column
        # In materials-adversarial, splits are usually just row indices in processed.csv
        # Let's extract the sequences. The column is usually 'original_representation' or 'smiles' or 'psmiles'
        if 'original_representation' in self.df.columns:
            self.seq_col = 'original_representation'
        elif 'smiles' in self.df.columns:
            self.seq_col = 'smiles'
        else:
            self.seq_col = self.df.columns[0] # fallback

    def get_frozen_sources(self) -> list[tuple[str, str]]:
        """Returns list of (id, sequence) for the frozen test set."""
        sources = []
        for idx in self.source_ids:
            seq = self.df.iloc[idx][self.seq_col]
            sources.append((str(idx), seq))
        return sources
