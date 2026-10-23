import sys, os
sys.path.insert(0, r'c:\S5\DL\Project')

import numpy as np

# ── Test 1: utils ──────────────────────────────────────────────────────────
from phase1.utils import set_seed, compute_mae, compute_rmse
set_seed(42)
y  = np.array([1.0, 2.0, 3.0])
yp = np.array([1.1, 1.9, 3.2])
print(f"[utils]  MAE={compute_mae(y,yp):.4f}  RMSE={compute_rmse(y,yp):.4f}  OK")

# ── Test 2: representation ────────────────────────────────────────────────
from phase1.representation import build_default_tokenizer
tok = build_default_tokenizer()
print(repr(tok.vocab))
ids   = tok.encode("CC(=O)N")
recon = tok.decode(ids)
assert recon == "CC(=O)N", f"Decode mismatch: {recon}"
print(f"[repr]   encode/decode roundtrip OK  ->  {recon}")

# ── Test 3: data ──────────────────────────────────────────────────────────
from phase1.data import generate_synthetic_dataset, clean_dataset, split_dataset
smi, prop = generate_synthetic_dataset(n_samples=200, seed=42)
smi, prop = clean_dataset(smi, prop)
splits    = split_dataset(smi, prop, seed=42)
tr, vl, ts = splits["train"], splits["val"], splits["test"]
print(f"[data]   train={len(tr[0])}  val={len(vl[0])}  test={len(ts[0])}  OK")

# ── Test 4: attack_engine — TransitionModel ───────────────────────────────
from phase1.attack_engine import TransitionModel, PlausibilityChecker
from phase1.representation import CHEMISTRY_TOKENS

encoded = [tok.encode(s) for s in smi[:80]]
tm = TransitionModel(vocab_size=tok.vocab.size, token_list=tok.vocab._tokens, alpha=0.5)
tm.fit(encoded)
top5 = tm.top_k_replacements(tok.vocab.encode_token("C"), k=5)
print(f"[TM]     top-5 for 'C': {top5}")

# ── Test 5: PlausibilityChecker ───────────────────────────────────────────
pc = PlausibilityChecker(allowed_tokens=set(CHEMISTRY_TOKENS), min_length=4)

cases = [
    (["C","C","O","=","C"],     True),
    (["(","C","C","O"],         False),  # unmatched (
    (["C","1","C","C","O"],     False),  # odd ring digit
    (["=","C","O","N"],         False),  # starts with bond
    (["C","C","C","C","O"],     False),  # 4 consecutive C exceeds max_repeat=3
]
print("[PC]     Plausibility checks:")
all_ok = True
for tokens, expected in cases:
    vr = pc.check(tokens)
    ok = vr.is_valid == expected
    all_ok = all_ok and ok
    mark = "OK" if ok else "FAIL"
    print(f"         [{mark}]  {tokens}  valid={vr.is_valid}  ({vr.reason})")

# ── Test 6: MCMCSampler (with dummy predictor) ────────────────────────────
from phase1.attack_engine import MCMCSampler, summarise_attack_results

def dummy_predict(smiles):
    return float(sum(1 for c in smiles if c.isupper())) * 0.15

sampler = MCMCSampler(
    transition_model=tm,
    plausibility_checker=pc,
    tokenizer=tok,
    predict_fn=dummy_predict,
    beta=2.0,
    n_steps=20,
    max_mutations=3,
    seed=42,
)
results = sampler.attack_batch(smi[:5], verbose=True)
for r in results:
    print(f"         orig={r.original_smiles:<18} adv={r.adversarial_smiles:<18} drift={r.drift:.3f}")

summary = summarise_attack_results(results, drift_threshold=0.05)

print()
if all_ok:
    print("ALL SMOKE TESTS PASSED.")
else:
    print("SOME PLAUSIBILITY TESTS FAILED — check above.")
    sys.exit(1)
