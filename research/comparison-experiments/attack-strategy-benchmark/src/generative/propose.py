"""
Proposer wrapper that uses the trained generator.
"""
from __future__ import annotations
import json
import torch
import numpy as np
from pathlib import Path

from generative.model import DenoisingSeqAE
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent.parent.parent

class GenerativeProposer:
    def __init__(self, domain: str, ckpt_path: str, config_path: str, pad_idx: int = 0, vocab_size: int = None, max_len: int = 256):
        self.domain = domain
        self.device = get_device()
        
        config = json.load(open(config_path))
        if vocab_size is None:
            vocab_size = 4 if domain == "bio_cyber" else 36
            
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.noise_rate = config.get("noise_rate", 0.15)
        
        self.model = DenoisingSeqAE(vocab_size=vocab_size, pad_idx=pad_idx, **config["model_kwargs"])
        if Path(ckpt_path).exists():
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def _corrupt(self, seq: list[int], rng: np.random.Generator) -> list[int]:
        res = list(seq)
        valid_tokens = [t for t in range(1, self.vocab_size + 1) if t != self.pad_idx]
        for i in range(len(res)):
            if rng.random() < self.noise_rate:
                res[i] = int(rng.choice(valid_tokens))
        return res

    def propose(self, source_seq: list[int], rng: np.random.Generator, temperature: float = 1.0, max_len: int = None) -> tuple[list[int], int]:
        """
        Returns (proposed_sequence, inference_count)
        """
        if max_len is None:
            max_len = self.max_len
            
        corrupted = self._corrupt(source_seq, rng)
        # padding
        corrupted_pad = corrupted + [self.pad_idx] * max(0, max_len - len(corrupted))
        corrupted_pad = corrupted_pad[:max_len]
        
        src_t = torch.tensor([corrupted_pad], dtype=torch.long, device=self.device)
        
        # We need a torch Generator for sampling
        trng = torch.Generator(device=self.device)
        trng.manual_seed(int(rng.integers(0, 2**32 - 1)))
        
        generated = self.model.generate(
            src=src_t, 
            max_len=max_len, 
            temperature=temperature, 
            greedy=False, 
            rng=trng
        )
        
        # Trim at first pad
        if self.pad_idx in generated:
            generated = generated[:generated.index(self.pad_idx)]
            
        return generated, 1
