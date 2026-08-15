"""Predictor wrapper adapting a trained model to PredictorProtocol. PENDING(torch).

The target model is consumed by attacks ONLY through
`attacks.generator.PredictorProtocol` (`predict(list[str]) -> np.ndarray` plus
`target_units`). Keeping that seam narrow is what allows a gradient-based or
probabilistic attack to be added later without touching the evaluator.

`attacks.generator.ConstantPredictor` implements the same protocol and lets the
full pipeline be exercised without torch.
"""

from __future__ import annotations

import json
import torch
import numpy as np
from pathlib import Path
from ..data.tokenizer import tokenize
from ..data.scaler import TargetScaler

class TransformerRegressor:
    """Wraps a trained checkpoint as a Predictor."""

    def __init__(self, model: torch.nn.Module, vocab: list[str], scaler: TargetScaler, target_units: str = "K") -> None:
        self.model = model
        self.vocab = vocab
        self.scaler = scaler
        self.char2idx = {c: i + 1 for i, c in enumerate(vocab)} # 0 is padding
        self.target_units = target_units
        self.model.eval()

    def predict(self, psmiles_list: list[str]) -> np.ndarray:
        tokenized = [tokenize(p) for p in psmiles_list]
        max_len = max((len(t) for t in tokenized), default=1)
        
        input_ids = []
        padding_masks = []
        
        for t in tokenized:
            # truncate to model's max_len if needed, but typically we enforce a big enough max_len
            # In our case config says max_seq_len is 256
            ids = [self.char2idx.get(c, 0) for c in t]
            mask = [False] * len(ids)
            
            while len(ids) < max_len:
                ids.append(0)
                mask.append(True)
                
            input_ids.append(ids)
            padding_masks.append(mask)
            
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        mask_tensor = torch.tensor(padding_masks, dtype=torch.bool)
        
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        
        with torch.no_grad():
            preds = self.model(input_tensor, padding_mask=mask_tensor)
            
        preds_np = preds.cpu().numpy()
        return self.scaler.inverse_transform(preds_np)
