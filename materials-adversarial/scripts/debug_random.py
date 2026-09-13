import numpy as np
from pathlib import Path
from src.materials_adv.attacks.search.strategies import RandomSearch
from src.materials_adv.attacks.search.proposals import CompositeProposalOperator
from src.materials_adv.attacks.substitution import SubstitutionAttack
from scripts.evaluate_phase4 import load_vocab, load_ordinary_baseline

def main():
    repo_root = Path(__file__).resolve().parent.parent
    vocab = load_vocab()
    p4_dir = repo_root / "results" / "phase4_controlled_robustness"
    model = load_ordinary_baseline(vocab)
    
    class Pred:
        def predict(self, t): return model.predict(t)
        
    rng = np.random.default_rng(42)
    sub = SubstitutionAttack(rng, allowed_tokens=vocab, attack_budget=1, role_preserving=True)
    prop = CompositeProposalOperator([sub], rng)
    
    strat = RandomSearch(Pred(), prop, rng, max_changes=3, query_budget=10)
    orig = "[*]CNC(=O)CCCCCCCCC(=O)N[*]"
    res = strat.search(orig)
    
    for i, t in enumerate(res.trace[:10]):
        print(f"Step {i}: {t.get('rejection_reason', 'Accepted')} | {t.get('candidate')}")

if __name__ == "__main__":
    main()
