"""
Denoising Sequence Autoencoder — the generative proposal model.

Architecture:
  Embedding → GRU encoder → GRU decoder → linear token head

The encoder reads the *corrupted* source sequence.
The decoder reconstructs the *clean* source token-by-token.
At inference, we feed the (slightly corrupted) source through
the encoder and sample/greedily decode to get a proposal edit.

Parameters: ~300k–800k for typical vocab / seq sizes. Well under 2 M.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingSeqAE(nn.Module):
    """Source-conditioned denoising autoencoder used as the generative proposal model."""

    def __init__(
        self,
        vocab_size: int,        # number of token types (excl. padding)
        pad_idx: int,
        d_embed: int = 64,
        d_hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.d_embed = d_embed
        self.d_hidden = d_hidden

        self.embedding = nn.Embedding(vocab_size + 1, d_embed, padding_idx=pad_idx)

        self.encoder = nn.GRU(
            d_embed, d_hidden, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )
        # project bidirectional hidden → decoder hidden (one direction per layer)
        self.enc2dec = nn.Linear(d_hidden * 2, d_hidden)

        self.decoder = nn.GRU(
            d_embed + d_hidden * 2, d_hidden, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(d_hidden, vocab_size + 1)

    # ------------------------------------------------------------------
    def _encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode corrupted source. Returns (enc_outputs, h_dec_init)."""
        emb = self.embedding(src)                             # (B, L, E)
        enc_out, h_n = self.encoder(emb)                      # h_n: (2*nl, B, H)
        # Take last-layer bidirectional hidden → decoder init
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)        # (B, 2H)
        h_dec = torch.tanh(self.enc2dec(h_last))              # (B, H)
        # Expand to n_layers (share same init for all layers)
        n_layers = self.decoder.num_layers
        h_dec = h_dec.unsqueeze(0).expand(n_layers, -1, -1).contiguous()
        return enc_out, h_dec

    def forward(
        self,
        src: torch.Tensor,      # (B, L) — corrupted source
        tgt_in: torch.Tensor,   # (B, T) — teacher-forced target prefix (clean[:-1])
    ) -> torch.Tensor:
        """Returns logits (B, T, vocab+1)."""
        enc_out, h_dec = self._encode(src)
        # Global context = mean of encoder outputs
        ctx = enc_out.mean(dim=1, keepdim=True).expand(-1, tgt_in.size(1), -1)  # (B,T,2H)

        dec_emb = self.embedding(tgt_in)                      # (B, T, E)
        dec_in = torch.cat([dec_emb, ctx], dim=-1)            # (B, T, E+2H)
        dec_out, _ = self.decoder(dec_in, h_dec)              # (B, T, H)
        return self.out_proj(dec_out)                         # (B, T, V+1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,    # (1, L)
        max_len: int,
        temperature: float = 1.0,
        greedy: bool = False,
        rng: torch.Generator | None = None,
    ) -> list[int]:
        """Autoregressive decoding from encoder context."""
        self.eval()
        enc_out, h_dec = self._encode(src)
        ctx = enc_out.mean(dim=1, keepdim=True)               # (1, 1, 2H)

        # Start token = same as BOS; we use a real token drawn uniformly.
        # We handle BOS externally — caller provides first real token.
        # Here we start with padding and let the model produce the sequence.
        cur = torch.tensor([[self.pad_idx]], dtype=torch.long, device=src.device)
        generated: list[int] = []

        for _ in range(max_len):
            dec_emb = self.embedding(cur)                     # (1, 1, E)
            dec_in = torch.cat([dec_emb, ctx], dim=-1)        # (1, 1, E+2H)
            dec_out, h_dec = self.decoder(dec_in, h_dec)
            logits = self.out_proj(dec_out[:, 0, :])          # (1, V+1)
            # mask pad
            logits[:, self.pad_idx] = -1e9
            if greedy:
                tok = logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                tok = torch.multinomial(probs, 1, generator=rng).item()
            generated.append(int(tok))
            cur = torch.tensor([[tok]], dtype=torch.long, device=src.device)

        return generated

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
