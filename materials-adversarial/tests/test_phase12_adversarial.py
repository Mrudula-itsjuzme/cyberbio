import pytest
import torch
import json
from pathlib import Path
from rdkit import Chem

from materials_adv.data.tokenizer import tokenize

def get_file_hash(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def mutate_smiles(vocab, smi):
    tokens = tokenize(smi)
    if not tokens: return smi
    vocab_chars = [c for c in vocab if c not in ['[PAD]', '[BOS]', '[EOS]']]
    idx = 0
    if tokens[idx] in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si']:
        tokens[idx] = 'O'
    return "".join(tokens)

def test_phase12_canonical_checkpoints_exist():
    root = Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial")
    path_gnn = root / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
    path_tx = root / "results/phase4_controlled_robustness/mix_robust_0.1/model.pt"
    
    assert path_gnn.exists(), "Phase 11C canonical GNN checkpoint missing"
    assert path_tx.exists(), "Phase 4 canonical TX checkpoint missing"

def test_adaptive_search_constraints():
    # Model independent validity checks
    root = Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial")
    with open(root / "data/processed/vocab.json") as f: vocab = json.load(f)
    
    source = "CCO"
    cand = mutate_smiles(vocab, source)
    
    assert Chem.MolFromSmiles(source) is not None
    assert Chem.MolFromSmiles(cand) is not None
    
    orig_canon = Chem.MolToSmiles(Chem.MolFromSmiles(source))
    cand_canon = Chem.MolToSmiles(Chem.MolFromSmiles(cand))
    
    assert orig_canon != cand_canon, "Search must yield distinct graph"
