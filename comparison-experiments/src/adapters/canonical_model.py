import sys
import os
import torch
from pathlib import Path
from typing import List

repo_root = Path(__file__).resolve().parent.parent.parent.parent / "materials-adversarial"
if str(repo_root) not in sys.path and repo_root.exists():
    sys.path.insert(0, str(repo_root / "src"))

from materials_adv.framework.predictors import GraphMPNNPredictor

class CanonicalModelAdapter:
    def __init__(self, checkpoint_path: str = None, scaler_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = repo_root / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
        if scaler_path is None:
            scaler_path = repo_root / "results/models/transformer_regressor/scaler.json"
            
        self.checkpoint_path = Path(checkpoint_path)
        self.scaler_path = Path(scaler_path)
        
        # This predictor already performs the inverse scaling mapping
        self.predictor = GraphMPNNPredictor(
            checkpoint=self.checkpoint_path,
            scaler_path=self.scaler_path,
            strict=True
        )
    
    def predict(self, sequence: str) -> float:
        """
        Predict property for a single sequence (already inverse transformed by GraphMPNNPredictor).
        """
        return self.predictor.predict(sequence)

    def predict_batch(self, sequences: List[str]) -> List[float]:
        return self.predictor.predict_batch(sequences)

