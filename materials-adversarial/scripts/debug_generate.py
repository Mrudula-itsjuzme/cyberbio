import numpy as np
from pathlib import Path
from materials_adv.attacks.substitution import SubstitutionAttack
from scripts.evaluate_phase4 import load_vocab
from materials_adv.data.tokenizer import tokenize
import inspect

def main():
    repo_root = Path(__file__).resolve().parent.parent
    vocab = load_vocab()
    
    rng = np.random.default_rng(42)
    sub = SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1, role_preserving=True)
    
    orig = "[*]CNC(=O)CCCCCCCCC(=O)N[*]"
    tokens = list(tokenize(orig))
    print("generate source:\n", inspect.getsource(sub.generate))
    print("RETURN VALUE:", sub.generate(tokens, n_variants=1))

if __name__ == "__main__":
    main()
