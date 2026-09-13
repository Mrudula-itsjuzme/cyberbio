import pytest
import torch
import json
import ast
from pathlib import Path
from rdkit import Chem
import numpy as np

from materials_adv.data.tokenizer import tokenize

def get_file_hash(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def test_phase12b_canonical_checkpoints_exist():
    root = Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial")
    path_gnn = root / "results/phase11c_canonical_verification/1789326819/models/graph_mpnn_small/model.pt"
    path_tx = root / "results/phase4_controlled_robustness/mix_robust_0.1/model.pt"
    
    assert path_gnn.exists(), "Phase 11C canonical GNN checkpoint missing"
    assert path_tx.exists(), "Phase 4 canonical TX checkpoint missing"

def test_phase12b_script_structure():
    """Assert exactly one main block and no monkey patching."""
    root = Path("/home/mrudula/Downloads/DL_cyberbio/materials-adversarial")
    script = root / "scripts/run_phase12b_adaptive_search_audit.py"
    if not script.exists():
        pytest.skip("Script not written yet")
    
    content = script.read_text()
    tree = ast.parse(content)
    
    main_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == '__name__':
                if len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Constant):
                    if node.test.comparators[0].value == '__main__':
                        main_count += 1
    
    assert main_count == 1, f"Expected exactly 1 main block, found {main_count}"
    
    # Assert no monkey patching of Phase12Evaluator
    assert "Phase12Evaluator." not in content or "Phase12bEvaluator." not in content, "No monkey patching allowed"

def test_edit_distance_bounds():
    """Verify standard Levenshtein distance on active tokens correctly limits budget."""
    source = "CCO"
    cands = [
        ("CFO", 1), 
        ("C", 2),
        ("C(F)(F)O", 6)
    ]
    
    for cand, expected in cands:
        t_orig = tokenize(source)
        t_prop = tokenize(cand)
        size_x = len(t_orig) + 1
        size_y = len(t_prop) + 1
        matrix = np.zeros ((size_x, size_y))
        for x in range(size_x): matrix [x, 0] = x
        for y in range(size_y): matrix [0, y] = y
        for x in range(1, size_x):
            for y in range(1, size_y):
                if t_orig[x-1] == t_prop[y-1]:
                    matrix [x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1], matrix[x, y-1] + 1)
                else:
                    matrix [x,y] = min(matrix[x-1,y] + 1, matrix[x-1,y-1] + 1, matrix[x,y-1] + 1)
        dist = int(matrix[size_x - 1, size_y - 1])
        assert dist == expected, f"Expected {expected} for {cand}, got {dist}"
