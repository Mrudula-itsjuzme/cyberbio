"""
model.py — Baseline deep-learning property-prediction model for Phase 1.

Architecture: Small Transformer encoder (2 layers, 4 heads, d_model=64)
              + mean-pooling over non-padded positions
              + linear regression head → scalar property prediction.

Kept deliberately small so it trains in < 2 minutes on CPU.

Part of: Learning to Attack and Defend — Phase 1
"""

import os
import math
import time
import numpy as np
from typing import List, Tuple, Dict, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    raise ImportError(
        "PyTorch is required for model.py. "
        "Install with: pip install torch"
    )


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class MaterialDataset(Dataset):
    """
    PyTorch Dataset for (sequence IDs, length, property) triples.

    Args:
        ids_array:  np.ndarray of shape (N, max_len), integer token IDs.
        lengths:    np.ndarray of shape (N,), actual sequence lengths.
        properties: np.ndarray of shape (N,), float regression targets.
    """

    def __init__(self,
                 ids_array:  np.ndarray,
                 lengths:    np.ndarray,
                 properties: np.ndarray):
        self.ids  = torch.from_numpy(ids_array).long()
        self.lens = torch.from_numpy(lengths).long()
        self.prop = torch.from_numpy(properties).float()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ids[idx], self.lens[idx], self.prop[idx]


# ── Model components ──────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding injected into the embedding.
    Supports sequences up to max_len positions.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                          # (L, d)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                                         # (1, L, d)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPropertyPredictor(nn.Module):
    """
    Small Transformer Encoder → mean-pooled regression head.

    Architecture:
        Embedding(vocab_size, d_model)
        + PositionalEncoding
        → TransformerEncoderLayer × n_layers  (multi-head self-attention)
        → mean pooling over non-padded positions
        → Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, 1)

    Output: scalar property prediction per sequence (shape: (batch,)).
    """

    def __init__(self,
                 vocab_size:  int,
                 d_model:     int   = 64,
                 n_heads:     int   = 4,
                 n_layers:    int   = 2,
                 d_ff:        int   = 128,
                 dropout:     float = 0.1,
                 max_len:     int   = 256,
                 pad_id:      int   = 0):
        super().__init__()
        self.pad_id  = pad_id
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc   = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,         # (batch, seq, d_model)
            norm_first=True,          # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def _make_key_padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """True where token = PAD (those positions are ignored by attention)."""
        return ids == self.pad_id                                    # (batch, seq)

    def forward(self, ids: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            ids:     (batch, seq_len) integer token IDs.
            lengths: (batch,) actual lengths (optional; used for mean-pooling mask).

        Returns:
            preds: (batch,) float property predictions.
        """
        key_pad_mask = self._make_key_padding_mask(ids)             # (batch, seq)

        # Embedding + positional encoding
        x = self.embedding(ids) * math.sqrt(self.d_model)           # (batch, seq, d)
        x = self.pos_enc(x)

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=key_pad_mask)      # (batch, seq, d)

        # Mean pooling: average only over non-padded positions
        pad_mask_float = (~key_pad_mask).float().unsqueeze(-1)       # (batch, seq, 1)
        x_pooled = (x * pad_mask_float).sum(dim=1) / \
                   pad_mask_float.sum(dim=1).clamp(min=1.0)          # (batch, d)

        # Regression head
        preds = self.regressor(x_pooled).squeeze(-1)                 # (batch,)
        return preds

    def predict(self, ids: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> np.ndarray:
        """Convenience method: run forward pass and return numpy array."""
        self.eval()
        with torch.no_grad():
            preds = self.forward(ids, lengths)
        return preds.cpu().numpy()

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training ──────────────────────────────────────────────────────────────────

def build_model(vocab_size:  int,
                d_model:     int   = 64,
                n_heads:     int   = 4,
                n_layers:    int   = 2,
                d_ff:        int   = 128,
                dropout:     float = 0.1,
                max_len:     int   = 256,
                pad_id:      int   = 0,
                device:      str   = "cpu") -> TransformerPropertyPredictor:
    """
    Instantiate and return the TransformerPropertyPredictor.
    Prints model summary (n_params, config).
    """
    model = TransformerPropertyPredictor(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        pad_id=pad_id,
    ).to(device)

    print(f"[Model] TransformerPropertyPredictor")
    print(f"        vocab_size={vocab_size}, d_model={d_model}, "
          f"n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}")
    print(f"        Trainable parameters: {model.n_params:,}")
    print(f"        Device: {device}")
    return model


def train_one_epoch(model:       TransformerPropertyPredictor,
                    loader:      DataLoader,
                    optimizer:   torch.optim.Optimizer,
                    device:      str = "cpu") -> float:
    """Run one training epoch. Returns mean MSE loss."""
    model.train()
    total_loss = 0.0
    for ids, lengths, targets in loader:
        ids, targets = ids.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(ids)
        loss  = F.mse_loss(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(ids)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model:  TransformerPropertyPredictor,
             loader: DataLoader,
             device: str = "cpu") -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a DataLoader.

    Returns:
        (mean_mse_loss, y_true_array, y_pred_array)
    """
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    for ids, lengths, targets in loader:
        ids, targets = ids.to(device), targets.to(device)
        preds = model(ids)
        loss  = F.mse_loss(preds, targets)
        total_loss += loss.item() * len(ids)
        all_true.append(targets.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    mean_loss = total_loss / max(len(loader.dataset), 1)
    return mean_loss, y_true, y_pred


def train(model:           TransformerPropertyPredictor,
          train_loader:    DataLoader,
          val_loader:      DataLoader,
          n_epochs:        int   = 30,
          lr:              float = 3e-4,
          weight_decay:    float = 1e-4,
          patience:        int   = 8,
          checkpoint_path: str   = "checkpoints/baseline.pt",
          device:          str   = "cpu",
          verbose:         bool  = True) -> Dict:
    """
    Full training loop with:
      - AdamW optimizer
      - CosineAnnealingLR scheduler
      - Early stopping (patience on val loss)
      - Best-model checkpointing

    Returns:
        history dict with keys 'train_losses', 'val_losses', 'best_epoch'.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.05
    )

    train_losses, val_losses = [], []
    best_val_loss   = float("inf")
    best_epoch      = 0
    patience_count  = 0
    t0              = time.time()

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, y_true, y_pred = evaluate(model, val_loader, device)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            mae  = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{n_epochs}  "
                  f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
                  f"MAE={mae:.4f}  RMSE={rmse:.4f}  "
                  f"lr={lr_now:.2e}  [{elapsed:.1f}s]")

        # Save best checkpoint
        if va_loss < best_val_loss:
            best_val_loss  = va_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         va_loss,
                "config": {
                    "vocab_size": model.embedding.num_embeddings,
                    "d_model":    model.d_model,
                    "n_heads":    model.encoder.layers[0].self_attn.num_heads,
                    "n_layers":   len(model.encoder.layers),
                    "pad_id":     model.pad_id,
                    # Save max_len so load_model() rebuilds pos_enc with the
                    # correct buffer size and avoids a shape-mismatch error.
                    "max_len":    model.pos_enc.pe.shape[1],
                },
            }, checkpoint_path)
        else:
            patience_count += 1

        if patience_count >= patience:
            if verbose:
                print(f"  [Early stop] No improvement for {patience} epochs. "
                      f"Best val loss = {best_val_loss:.4f} at epoch {best_epoch}.")
            break

    total_time = time.time() - t0
    print(f"\n[Train] Finished. Best epoch={best_epoch}  "
          f"best_val_loss={best_val_loss:.4f}  total_time={total_time:.1f}s")
    print(f"[Train] Checkpoint saved -> {checkpoint_path}")

    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "best_epoch":   best_epoch,
        "best_val_loss": best_val_loss,
    }


def load_model(checkpoint_path: str,
               device:          str = "cpu") -> TransformerPropertyPredictor:
    """
    Load a TransformerPropertyPredictor from a saved checkpoint.

    Robust to max_len mismatches: always reads the positional-encoding
    buffer shape directly from the saved state-dict, so the model is
    rebuilt with the exact max_len that was used during training.
    This means no shape mismatch can ever occur, regardless of kernel
    cache state or import order.

    Returns:
        Model in eval mode on device.
    """
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    state = ckpt["model_state_dict"]

    # Read max_len directly from the saved pe buffer — always reliable.
    if "pos_enc.pe" in state:
        max_len = state["pos_enc.pe"].shape[1]   # tensor shape: (1, max_len, d_model)
    elif "max_len" in cfg:
        max_len = cfg["max_len"]
    else:
        max_len = 256  # safe fallback

    model = TransformerPropertyPredictor(
        vocab_size = cfg["vocab_size"],
        d_model    = cfg.get("d_model", 64),
        n_heads    = cfg.get("n_heads", 4),
        n_layers   = cfg.get("n_layers", 2),
        pad_id     = cfg.get("pad_id", 0),
        max_len    = max_len,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"[Model] Loaded from {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}, "
          f"max_len={max_len})")
    return model


# ── __main__ quick-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from phase1.representation import build_default_tokenizer
    from phase1.data import generate_synthetic_dataset, clean_dataset, split_dataset

    print("── Quick-test: model.py ────────────────────────────────────")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # 1. Generate tiny dataset for speed
    smi_list, prop_list = generate_synthetic_dataset(n_samples=300, seed=0)
    smi_list, prop_list = clean_dataset(smi_list, prop_list)
    splits = split_dataset(smi_list, prop_list, seed=0)

    # 2. Tokenize
    tokenizer = build_default_tokenizer()
    vocab_size = tokenizer.vocab.size
    print(f"Vocab size: {vocab_size}")

    def make_loader(smi, props, batch_size=32, shuffle=True):
        ids_arr, lengths = tokenizer.encode_batch(smi)
        props_arr = np.array(props, dtype=np.float32)
        ds = MaterialDataset(ids_arr, lengths, props_arr)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    tr_loader  = make_loader(*splits["train"])
    val_loader = make_loader(*splits["val"],   shuffle=False)
    tst_loader = make_loader(*splits["test"],  shuffle=False)

    # 3. Build and train
    model = build_model(vocab_size=vocab_size, device=DEVICE)
    os.makedirs("checkpoints", exist_ok=True)
    history = train(
        model, tr_loader, val_loader,
        n_epochs=10, checkpoint_path="checkpoints/baseline_test.pt", device=DEVICE
    )

    # 4. Evaluate on test
    _, y_true, y_pred = evaluate(model, tst_loader, device=DEVICE)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    print(f"Test → MAE={mae:.4f}  RMSE={rmse:.4f}")

    print("\n[model.py] Quick-test complete.")
