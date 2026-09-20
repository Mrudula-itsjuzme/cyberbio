import numpy as np

class ContextFeatureExtractor:
    """Extracts zero-shot structural features from an unperturbed source SMILES sequence."""
    
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        # Choose a small set of interpretable tokens to count
        self.target_tokens = ['C', 'O', 'N', 'c', '1', '2', '(', ')', '=']
        self.feature_names = [
            "sequence_length",
            "ring_count",
            "branch_count",
            "aromatic_fraction",
            "heteroatom_fraction"
        ] + [f"freq_{t}" for t in self.target_tokens]
        
        self.dim = len(self.feature_names)
        
    def extract(self, representation: str) -> np.ndarray:
        """
        Extract features from a SMILES string.
        Must be strictly deterministic and depend ONLY on the original string.
        """
        # Very simple parser. For strict leakage rules, we only use the string itself, 
        # avoiding even RDKit parsing which could fail.
        
        # 1. sequence_length
        seq_len = len(representation)
        if seq_len == 0:
            return np.zeros(self.dim)
            
        # 2. ring_count (heuristic by numbers)
        ring_chars = [c for c in representation if c.isdigit()]
        ring_count = len(ring_chars) / 2.0  # approximate
        
        # 3. branch_count (parentheses)
        branch_count = representation.count("(")
        
        # 4. aromatic_fraction (lowercase letters)
        aromatic_count = sum(1 for c in representation if c.islower() and c.isalpha())
        aromatic_fraction = aromatic_count / seq_len
        
        # 5. heteroatom_fraction (N, O, S, P, F, Cl, Br, I)
        heteroatoms = {'N', 'O', 'S', 'P', 'F', 'C', 'l', 'B', 'r', 'I'} # 'C' is not heteroatom, but keeping logic simple. 
        # Actually proper heteroatom:
        true_heteros = {'N', 'O', 'S', 'P', 'F'} # Simplified
        hetero_count = sum(1 for c in representation if c in true_heteros)
        hetero_fraction = hetero_count / seq_len
        
        features = [
            seq_len,
            ring_count,
            branch_count,
            aromatic_fraction,
            hetero_fraction
        ]
        
        for t in self.target_tokens:
            features.append(representation.count(t) / seq_len)
            
        return np.array(features, dtype=float)

class ContextScaler:
    """Standardizes context features using strictly TRAIN statistics."""
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
        
    def fit(self, features: np.ndarray):
        """Fit on a matrix of train features [N, dim]."""
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        # Prevent division by zero
        self.std[self.std < 1e-8] = 1.0
        self.is_fitted = True
        
    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ContextScaler must be fitted on TRAIN data first.")
        if features.ndim == 1:
            return (features - self.mean) / self.std
        return (features - self.mean) / self.std

