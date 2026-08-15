"""Small Transformer encoder for Tg regression. PENDING(torch + dataset).

Architecture is fixed and agreed:

    PSMILES -> chemical tokenizer -> token ids -> embedding
      -> Transformer encoder -> sequence pooling -> regression head -> Tg

Two independent blockers:

1. torch is not installed (and this machine has no GPU, so the CPU wheel is the
   right choice: pip install torch --index-url https://download.pytorch.org/whl/cpu).

2. Hyperparameters must not be guessed. n_layers, d_model, n_heads, dropout and
   learning rate depend on vocabulary size and usable sample count, neither of
   which is known. Published OpenPoly figures (~3985 pairs over 26 properties,
   ~745 polymers) suggest the Tg slice may be small, and the paper reports
   XGBoost beating deep learning on Tg for that reason.

No sample-size threshold is being asserted as a rule. Capacity gets chosen from
the measured count after the audit, and if the slice is small the honest framing
is a deliberately small Transformer serving as a CONTROLLED ATTACK TARGET rather
than a competitive predictor. That is a legitimate design for this research
question, but it must be stated explicitly in the writeup.
"""

from __future__ import annotations

from ..utils.pending import PendingImplementation

import torch
import torch.nn as nn

class TransformerRegressorModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, dim_feedforward, dropout, max_seq_len, pooling="mean"):
        super().__init__()
        # +1 for padding index 0
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooling = pooling
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, src, padding_mask=None):
        # src: [batch_size, seq_len]
        seq_len = src.size(1)
        positions = torch.arange(seq_len, device=src.device).unsqueeze(0).expand(src.size(0), seq_len)
        
        emb = self.embedding(src) + self.pos_encoder(positions)
        out = self.transformer_encoder(emb, src_key_padding_mask=padding_mask)
        
        # out: [batch_size, seq_len, d_model]
        if self.pooling == "mean":
            if padding_mask is not None:
                mask = (~padding_mask).unsqueeze(-1).float()
                out = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                out = out.mean(dim=1)
        elif self.pooling == "max":
            if padding_mask is not None:
                out = out.masked_fill(padding_mask.unsqueeze(-1), -float('inf'))
            out = out.max(dim=1)[0]
        elif self.pooling == "cls":
            out = out[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling {self.pooling}")
            
        return self.regressor(out).squeeze(-1)
