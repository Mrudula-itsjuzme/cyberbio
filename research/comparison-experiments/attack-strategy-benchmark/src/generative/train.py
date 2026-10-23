import json, os, sys, math, time, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

from generative.model import DenoisingSeqAE
from generative.dataset import DenoisingDataset

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(logits, tgt_out, pad_idx):
    # logits: (B, T, V+1), tgt_out: (B, T)
    preds = logits.argmax(dim=-1)
    mask = tgt_out != pad_idx
    correct = (preds == tgt_out) & mask
    total_tokens = mask.sum().item()
    correct_tokens = correct.sum().item()
    
    # Exact sequence match
    seq_match = (preds == tgt_out) | (~mask)
    exact_matches = seq_match.all(dim=-1).sum().item()
    return correct_tokens, total_tokens, exact_matches, tgt_out.size(0)

def train_generator(domain: str):
    config = json.load(open(ROOT / f"configs/generative_{domain}.json"))
    out_dir = ROOT / f"results/generative/checkpoints/{domain}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(config["seed"])
    
    # Load test source IDs to exclude
    test_manifest = json.load(open(ROOT / f"results/manifests/{domain}_sources_n30.json"))
    test_ids = {r["source_id"] for r in test_manifest}
    
    # Load training data
    sequences = []
    if domain == "bio_cyber":
        df = pd.read_csv(ROOT.parent.parent / "bio-cyber-adversarial/data/v3/dataset.csv")
        df = df[df["split"] == "train"]
        df = df[~df["id"].isin(test_ids)]
        # Map ACGT to 1,2,3,4. pad=0
        vocab_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
        # subsample if requested
        if config.get("max_train_samples"):
            df = df.sample(n=min(len(df), config["max_train_samples"]), random_state=config["seed"])
        for seq in df["sequence"].tolist():
            sequences.append([vocab_map[c] for c in seq if c in vocab_map])
        vocab_size = 4
        pad_idx = 0
        max_len = 150
        
    elif domain == "materials":
        df = pd.read_csv(ROOT.parent.parent / "materials-adversarial/data/processed/train_aug_phase2b.csv")
        vocab = json.load(open(ROOT.parent.parent / "materials-adversarial/data/processed/vocab.json"))
        char2idx = {c: i+1 for i, c in enumerate(vocab)}
        # test_ids are like 'mat_test_123', here we just exclude SMILES that are in test set just to be safe
        test_smiles = {r["SMILES"] for r in test_manifest}
        df = df[~df["canonical_smiles"].isin(test_smiles)]
        if config.get("max_train_samples"):
            df = df.sample(n=min(len(df), config["max_train_samples"]), random_state=config["seed"])
        for smi in df["canonical_smiles"].dropna().tolist():
            sequences.append([char2idx[c] for c in smi if c in char2idx])
        vocab_size = 36
        pad_idx = 0
        max_len = 256
    else:
        raise ValueError("Unknown domain")
        
    print(f"[{domain}] Training set: {len(sequences)} sequences")
    # Split train/val
    np.random.shuffle(sequences)
    val_size = int(len(sequences) * 0.1)
    train_seq = sequences[:-val_size]
    val_seq = sequences[-val_size:]
    
    train_ds = DenoisingDataset(train_seq, pad_idx, vocab_size, max_len, config.get("noise_rate", 0.15), config["seed"])
    val_ds = DenoisingDataset(val_seq, pad_idx, vocab_size, max_len, config.get("noise_rate", 0.15), config["seed"] + 1)
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    
    model = DenoisingSeqAE(vocab_size=vocab_size, pad_idx=pad_idx, **config["model_kwargs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"[{domain}] Model params: {model.param_count()}")
    
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    history = []
    best_val_loss = float("inf")
    
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        tc, tt, ec, total_seq = 0, 0, 0, 0
        for src, tgt_in, tgt_out in train_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            opt.zero_grad()
            logits = model(src, tgt_in)
            loss = criterion(logits.view(-1, vocab_size + 1), tgt_out.view(-1))
            loss.backward()
            opt.step()
            train_loss += loss.item() * src.size(0)
            
            c, t, e, s = compute_metrics(logits, tgt_out, pad_idx)
            tc+=c; tt+=t; ec+=e; total_seq+=s
            
        train_loss /= total_seq
        train_acc = tc / max(tt, 1)
        train_exact = ec / max(total_seq, 1)
        
        model.eval()
        val_loss = 0.0
        tc, tt, ec, total_seq = 0, 0, 0, 0
        with torch.no_grad():
            for src, tgt_in, tgt_out in val_loader:
                src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
                logits = model(src, tgt_in)
                loss = criterion(logits.view(-1, vocab_size + 1), tgt_out.view(-1))
                val_loss += loss.item() * src.size(0)
                
                c, t, e, s = compute_metrics(logits, tgt_out, pad_idx)
                tc+=c; tt+=t; ec+=e; total_seq+=s
                
        val_loss /= total_seq
        val_acc = tc / max(tt, 1)
        val_exact = ec / max(total_seq, 1)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Exact: {val_exact:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_exact": train_exact,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_exact": val_exact
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "checkpoint.pt")
            
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    
    # Save manifest
    ckpt_bytes = open(out_dir / "checkpoint.pt", "rb").read()
    manifest = {
        "domain": domain,
        "dataset_split": "train",
        "num_training_examples": len(train_seq),
        "seed": config["seed"],
        "architecture": "DenoisingSeqAE",
        "parameter_count": model.param_count(),
        "epochs": config["epochs"],
        "optimizer": "Adam",
        "learning_rate": config["lr"],
        "checkpoint_SHA256": hashlib.sha256(ckpt_bytes).hexdigest(),
        "best_val_loss": best_val_loss,
        "val_exact_match_rate": val_exact
    }
    with open(out_dir / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_generator(sys.argv[1])
    else:
        train_generator("bio_cyber")
        train_generator("materials")
