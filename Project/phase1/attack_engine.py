"""
attack_engine.py — Probabilistic MCMC Attack Engine for Phase 1.

Three sub-components (per spec §4):
  4.1  TransitionModel  — learn P(token_j | token_i) from corpus bigrams.
  4.2  PlausibilityChecker — rule-based validity checks (no RDKit).
  4.3  MCMCSampler — Metropolis-Hastings adversarial sequence search.

Part of: Learning to Attack and Defend — Phase 1
"""

import os
import math
import random
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable

# ── SMILES grammar constants (mirrored from representation.py to keep engine self-contained)
RING_DIGITS = set("12345678")
OPEN_BR     = {"("}
CLOSE_BR    = {")"}
OPEN_SQ     = {"["}
CLOSE_SQ    = {"]"}

# Maximum consecutive occurrences of the same atom token (valency heuristic)
_MAX_REPEAT = 3

# ─────────────────────────────────────────────────────────────────────────────
# 4.1  TRANSITION / MUTATION MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TransitionModel:
    """
    Learns a bigram transition matrix P(token_j | token_i) from a tokenised
    corpus.  Used to propose chemically plausible token substitutions during
    MCMC.

    The matrix is smoothed with Laplace (add-α) smoothing so that every
    token → token pair has non-zero probability.

    Attributes:
        vocab_size:  number of tokens in the vocabulary.
        matrix:      np.ndarray of shape (vocab_size, vocab_size),
                     row-stochastic (rows sum to 1).
        token_list:  ordered list of token strings (index = token id).
    """

    def __init__(self, vocab_size: int, token_list: List[str], alpha: float = 0.5):
        """
        Args:
            vocab_size:  size of the token vocabulary.
            token_list:  list of token strings where index i = token id i.
            alpha:       Laplace smoothing parameter.
        """
        self.vocab_size = vocab_size
        self.token_list = token_list
        self.alpha      = alpha
        self.matrix     = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        self._fitted    = False

    def fit(self, sequences: List[List[int]]) -> "TransitionModel":
        """
        Estimate the transition matrix from a list of tokenised sequences
        (each sequence is a list of integer token IDs).

        Pairs involving PAD (id=0) are excluded.

        Args:
            sequences: list of token-ID lists (from tokenizer.encode()).

        Returns:
            self  (for chaining).
        """
        counts = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        PAD_ID = 0   # hard-coded; consistent with Vocabulary

        for seq in sequences:
            for a, b in zip(seq[:-1], seq[1:]):
                if a != PAD_ID and b != PAD_ID:
                    counts[a, b] += 1.0

        # Laplace smoothing
        counts += self.alpha

        # Row-normalise to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        self.matrix = counts / np.maximum(row_sums, 1e-12)
        self._fitted = True

        total_bigrams = int(counts.sum() - self.alpha * self.vocab_size ** 2)
        print(f"[TransitionModel] Fitted on {len(sequences)} sequences, "
              f"~{total_bigrams:,} bigrams. "
              f"Matrix: {self.matrix.shape}, alpha={self.alpha}")
        return self

    def sample_replacement(self, token_id: int,
                            rng: Optional[np.random.Generator] = None,
                            exclude_ids: Optional[List[int]] = None) -> int:
        """
        Sample a replacement token for token_id using the learned transition
        distribution.

        Args:
            token_id:    integer ID of the current token.
            rng:         np.random.Generator (uses module-level rng if None).
            exclude_ids: token IDs to exclude from sampling (e.g. the token itself).

        Returns:
            Sampled replacement token ID.
        """
        if rng is None:
            rng = np.random.default_rng()

        probs = self.matrix[token_id].copy()

        if exclude_ids:
            for ex in exclude_ids:
                if 0 <= ex < len(probs):
                    probs[ex] = 0.0

        total = probs.sum()
        if total < 1e-12:
            # Fallback: uniform over non-excluded tokens
            probs = np.ones(self.vocab_size, dtype=np.float64)
            if exclude_ids:
                for ex in exclude_ids:
                    if 0 <= ex < len(probs):
                        probs[ex] = 0.0
            total = probs.sum()

        probs /= total
        return int(rng.choice(self.vocab_size, p=probs))

    def top_k_replacements(self, token_id: int, k: int = 5) -> List[Tuple[str, float]]:
        """
        Return the top-k most probable replacement tokens for token_id.

        Returns:
            List of (token_string, probability) pairs, highest probability first.
        """
        row   = self.matrix[token_id]
        top_k = np.argsort(row)[::-1][:k]
        return [(self.token_list[i], float(row[i])) for i in top_k]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.matrix)
        print(f"[TransitionModel] Matrix saved → {path}")

    def load(self, path: str) -> "TransitionModel":
        self.matrix  = np.load(path)
        self._fitted = True
        print(f"[TransitionModel] Matrix loaded ← {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  SCIENTIFIC VALIDITY / PLAUSIBILITY CHECKER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidityResult:
    is_valid: bool
    reason:   str = "OK"


class PlausibilityChecker:
    """
    Rule-based SMILES-like sequence validity checker.

    No RDKit dependency.  Checks applied (in order):
      1. Minimum length  (≥ min_length tokens, excluding specials).
      2. No unknown tokens  (every token must be in the allowed set).
      3. Parenthesis balance  — matched ( / ) count.
      4. Square bracket balance  — matched [ / ] count.
      5. Ring-digit parity  — each digit 1-8 must appear an even number of
         times (each ring closure opens and closes).
      6. Max-consecutive-repeat heuristic  — no more than max_repeat of
         the same atom token in an unbroken run.
      7. No bare bond at start or end (sequence can't open/close with = # -).
    """

    def __init__(self,
                 allowed_tokens:    Optional[set] = None,
                 min_length:        int  = 4,
                 max_repeat:        int  = _MAX_REPEAT):
        """
        Args:
            allowed_tokens: set of valid token strings; if None, accepts all.
            min_length:     minimum number of chemistry tokens (ignoring specials).
            max_repeat:     max same-atom token allowed consecutively.
        """
        self.allowed_tokens = allowed_tokens
        self.min_length     = min_length
        self.max_repeat     = max_repeat
        self._special       = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        self._bond_only     = {"=", "#", "-", ":"}

    def check(self, tokens: List[str]) -> ValidityResult:
        """
        Run all validity checks on a token list.

        Args:
            tokens: list of token strings (may include specials).

        Returns:
            ValidityResult(is_valid, reason).
        """
        # Strip specials for analysis
        chem_tokens = [t for t in tokens if t not in self._special]

        # 1. Minimum length
        if len(chem_tokens) < self.min_length:
            return ValidityResult(False,
                f"Too short: {len(chem_tokens)} tokens (min={self.min_length})")

        # 2. Unknown tokens
        if self.allowed_tokens is not None:
            unknowns = [t for t in chem_tokens if t not in self.allowed_tokens]
            if unknowns:
                return ValidityResult(False,
                    f"Unknown tokens: {unknowns[:5]}")

        # 3. Parenthesis balance
        depth = 0
        for t in chem_tokens:
            if t in OPEN_BR:
                depth += 1
            elif t in CLOSE_BR:
                depth -= 1
            if depth < 0:
                return ValidityResult(False,
                    "Unmatched ')': parenthesis depth went negative")
        if depth != 0:
            return ValidityResult(False,
                f"Unbalanced parentheses: {depth} unclosed '('")

        # 4. Square bracket balance
        sq_depth = 0
        for t in chem_tokens:
            if t in OPEN_SQ:
                sq_depth += 1
            elif t in CLOSE_SQ:
                sq_depth -= 1
            if sq_depth < 0:
                return ValidityResult(False,
                    "Unmatched ']': square bracket depth went negative")
        if sq_depth != 0:
            return ValidityResult(False,
                f"Unbalanced square brackets: {sq_depth} unclosed '['")

        # 5. Ring-digit parity
        for digit in RING_DIGITS:
            count = chem_tokens.count(digit)
            if count % 2 != 0:
                return ValidityResult(False,
                    f"Ring digit '{digit}' appears {count} time(s) (must be even)")

        # 6. Max-consecutive-repeat heuristic
        run_tok, run_len = None, 0
        for t in chem_tokens:
            if t == run_tok:
                run_len += 1
            else:
                run_tok, run_len = t, 1
            if run_len > self.max_repeat and t not in self._bond_only \
                    and t not in OPEN_BR and t not in CLOSE_BR \
                    and t not in RING_DIGITS:
                return ValidityResult(False,
                    f"Token '{t}' repeated {run_len} times consecutively "
                    f"(max={self.max_repeat})")

        # 7. Bond-only start / end
        if chem_tokens and chem_tokens[0] in self._bond_only:
            return ValidityResult(False,
                f"Sequence starts with bond token '{chem_tokens[0]}'")
        if chem_tokens and chem_tokens[-1] in self._bond_only:
            return ValidityResult(False,
                f"Sequence ends with bond token '{chem_tokens[-1]}'")

        return ValidityResult(True, "OK")

    def is_valid(self, tokens: List[str]) -> bool:
        return self.check(tokens).is_valid


# ─────────────────────────────────────────────────────────────────────────────
# Attack result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttackResult:
    """Container for the outcome of one MCMC attack run."""
    original_smiles:    str
    adversarial_smiles: str
    original_pred:      float
    adversarial_pred:   float
    drift:              float              # |ŷ_adv − ŷ_orig|
    n_mutations:        int                # tokens changed vs original
    is_valid:           bool
    validity_reason:    str
    n_steps:            int                # total MCMC steps run
    n_accepted:         int                # accepted proposals in chain
    chain_drifts:       List[float] = field(default_factory=list)  # drift at each accepted step

    @property
    def perturbation_rate(self) -> float:
        """Fraction of tokens that were changed."""
        orig_len = max(len(self.original_smiles), 1)
        return self.n_mutations / orig_len


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  MCMC SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

class MCMCSampler:
    """
    Metropolis-Hastings adversarial sequence sampler.

    The "energy" being maximised is the prediction drift:
        score(sequence) = |ŷ(sequence) − ŷ(original)|

    Acceptance criterion (drift-maximising Metropolis):
        α = min(1, exp(β * (score_new − score_current)))

    High β → greedy exploitation (always accept if drift improves).
    Low  β → more exploration (accept some worsening steps).
    """

    def __init__(self,
                 transition_model:   TransitionModel,
                 plausibility_checker: PlausibilityChecker,
                 tokenizer,                       # SMILESTokenizer from representation.py
                 predict_fn:         Callable,    # (List[str]) → float
                 beta:               float  = 2.0,
                 n_steps:            int    = 50,
                 max_mutations:      int    = 5,
                 seed:               int    = 42):
        """
        Args:
            transition_model:      fitted TransitionModel.
            plausibility_checker:  PlausibilityChecker.
            tokenizer:             SMILESTokenizer with vocab.
            predict_fn:            callable: takes a SMILES string → float prediction.
                                   Should be fast (called many times per sequence).
            beta:                  MCMC temperature.  Higher = more greedy.
            n_steps:               number of MCMC proposals per attack run.
            max_mutations:         hard cap on number of token substitutions
                                   allowed in one adversarial sequence.
            seed:                  random seed.
        """
        self.tm      = transition_model
        self.pc      = plausibility_checker
        self.tok     = tokenizer
        self.predict = predict_fn
        self.beta    = beta
        self.n_steps = n_steps
        self.max_mutations = max_mutations
        self.rng_py  = random.Random(seed)
        self.rng_np  = np.random.default_rng(seed)

        self._special_ids = {
            tokenizer.vocab.pad_id,
            tokenizer.vocab.bos_id,
            tokenizer.vocab.eos_id,
            tokenizer.vocab.unk_id,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tokens_to_smiles(self, tokens: List[str]) -> str:
        return self.tok.decode_tokens(tokens, strip_special=True)

    def _count_mutations(self, orig: List[str], current: List[str]) -> int:
        """Count positions where tokens differ (compared element-wise)."""
        return sum(a != b for a, b in zip(orig, current))

    def _propose(self, current_tokens: List[str],
                 orig_tokens:    List[str]) -> Optional[List[str]]:
        """
        Propose a new sequence by substituting one token.

        Strategy:
          1. Pick a random chemistry-token position.
          2. Sample a replacement from the transition distribution.
          3. Return the modified token list.

        Returns None if no valid position can be found.
        """
        # Find mutable positions: not special tokens, not already at max mutations
        mutable_positions = [
            i for i, t in enumerate(current_tokens)
            if t not in {"<PAD>", "<BOS>", "<EOS>", "<UNK>"}
        ]
        if not mutable_positions:
            return None

        pos    = self.rng_py.choice(mutable_positions)
        tok_id = self.tok.vocab.encode_token(current_tokens[pos])

        # Sample replacement, excluding the identity token to guarantee a change
        new_id  = self.tm.sample_replacement(
            tok_id,
            rng=self.rng_np,
            exclude_ids=[tok_id] + list(self._special_ids),
        )
        new_tok = self.tok.vocab.decode_id(new_id)

        # Build candidate token list
        candidate = list(current_tokens)
        candidate[pos] = new_tok
        return candidate

    def _accept(self, score_current: float, score_proposed: float) -> bool:
        """
        Metropolis acceptance rule.
        We are *maximising* score (drift), so:
            α = min(1, exp(β * (score_new − score_current)))
        """
        delta = score_proposed - score_current
        if delta >= 0:
            return True
        log_alpha = self.beta * delta
        return math.log(self.rng_py.random() + 1e-300) < log_alpha

    # ── Main attack method ────────────────────────────────────────────────────

    def attack(self, smiles: str) -> AttackResult:
        """
        Run MCMC adversarial attack on a single SMILES string.

        Args:
            smiles: the original SMILES sequence to attack.

        Returns:
            AttackResult containing the best adversarial sequence found.
        """
        orig_tokens  = self.tok.tokenize(smiles)
        orig_pred    = self.predict(smiles)

        current_tokens = list(orig_tokens)
        current_score  = 0.0   # initial drift = 0 (same as original)

        best_tokens = list(orig_tokens)
        best_score  = 0.0
        best_pred   = orig_pred

        chain_drifts = [0.0]
        n_accepted   = 0

        for _ in range(self.n_steps):
            # 1. Propose a mutation
            candidate = self._propose(current_tokens, orig_tokens)
            if candidate is None:
                continue

            # 2. Check mutation cap
            n_mut = self._count_mutations(orig_tokens, candidate)
            if n_mut > self.max_mutations:
                continue

            # 3. Plausibility gate (fast, no model call)
            vr = self.pc.check(candidate)
            if not vr.is_valid:
                continue

            # 4. Score via model
            cand_smiles = self._tokens_to_smiles(candidate)
            try:
                cand_pred  = self.predict(cand_smiles)
            except Exception:
                continue
            cand_score = abs(cand_pred - orig_pred)

            # 5. Metropolis acceptance
            if self._accept(current_score, cand_score):
                current_tokens = candidate
                current_score  = cand_score
                n_accepted    += 1
                chain_drifts.append(cand_score)

                # Track best ever seen in chain
                if cand_score > best_score:
                    best_score  = cand_score
                    best_tokens = list(candidate)
                    best_pred   = cand_pred

        # Final validity of best candidate
        final_vr = self.pc.check(best_tokens)
        final_smiles = self._tokens_to_smiles(best_tokens)
        n_mut_final  = self._count_mutations(orig_tokens, best_tokens)

        return AttackResult(
            original_smiles    = smiles,
            adversarial_smiles = final_smiles,
            original_pred      = float(orig_pred),
            adversarial_pred   = float(best_pred),
            drift              = float(best_score),
            n_mutations        = n_mut_final,
            is_valid           = final_vr.is_valid,
            validity_reason    = final_vr.reason,
            n_steps            = self.n_steps,
            n_accepted         = n_accepted,
            chain_drifts       = chain_drifts,
        )

    def attack_batch(self, smiles_list: List[str],
                     verbose: bool = True) -> List[AttackResult]:
        """
        Attack a list of SMILES sequences.

        Args:
            smiles_list: list of original sequences.
            verbose:     print a progress indicator.

        Returns:
            List of AttackResult, one per input sequence.
        """
        results = []
        n = len(smiles_list)
        for i, smi in enumerate(smiles_list):
            result = self.attack(smi)
            results.append(result)
            if verbose and (i % max(1, n // 10) == 0 or i == n - 1):
                print(f"  Attacked {i+1}/{n}  "
                      f"best_drift={result.drift:.4f}  "
                      f"accepted={result.n_accepted}/{result.n_steps}  "
                      f"valid={result.is_valid}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Attack summary utilities
# ─────────────────────────────────────────────────────────────────────────────

def summarise_attack_results(results: List[AttackResult],
                              drift_threshold: float = 0.5) -> Dict:
    """
    Compute aggregate statistics over a list of AttackResults.

    Returns:
        dict with keys: n_attacks, mean_drift, max_drift, attack_success_rate,
                        validity_rate, mean_mutations, mean_accepted.
    """
    n = len(results)
    if n == 0:
        return {}

    drifts      = [r.drift          for r in results]
    mutations   = [r.n_mutations    for r in results]
    accepted    = [r.n_accepted     for r in results]
    is_valid    = [r.is_valid       for r in results]
    is_success  = [r.drift > drift_threshold for r in results]

    summary = {
        "n_attacks":          n,
        "mean_drift":         float(np.mean(drifts)),
        "median_drift":       float(np.median(drifts)),
        "max_drift":          float(np.max(drifts)),
        "std_drift":          float(np.std(drifts)),
        "attack_success_rate": float(np.mean(is_success)) * 100,
        "validity_rate":      float(np.mean(is_valid)) * 100,
        "mean_mutations":     float(np.mean(mutations)),
        "mean_accepted":      float(np.mean(accepted)),
        "drift_threshold":    drift_threshold,
    }

    print("\n-- Attack Summary --------------------------------------------------------")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<26}: {v:.4f}" + ("%" if "rate" in k else ""))
        else:
            print(f"  {k:<26}: {v}")
    print("--------------------------------------------------------------------------\n")
    return summary


def results_to_records(results: List[AttackResult]) -> List[Dict]:
    """Convert AttackResult list to a list of flat dicts for table display."""
    records = []
    for r in results:
        records.append({
            "Original":    r.original_smiles[:20],
            "Adversarial": r.adversarial_smiles[:20],
            "Orig Pred":   f"{r.original_pred:.3f}",
            "Adv Pred":    f"{r.adversarial_pred:.3f}",
            "Drift":       f"{r.drift:.3f}",
            "Mutations":   str(r.n_mutations),
            "Valid":       "✓" if r.is_valid else "✗",
            "Accepted":    f"{r.n_accepted}/{r.n_steps}",
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# __main__ quick-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from phase1.representation import build_default_tokenizer, CHEMISTRY_TOKENS
    from phase1.data import generate_synthetic_dataset, clean_dataset

    print("── Quick-test: attack_engine.py ────────────────────────────────")

    # 1. Build tokenizer
    tok   = build_default_tokenizer()
    vocab = tok.vocab
    print(repr(vocab))

    # 2. Build small synthetic corpus
    smi_list, prop_list = generate_synthetic_dataset(n_samples=200, seed=42)
    smi_list, prop_list = clean_dataset(smi_list, prop_list)

    # 3. Fit transition model
    encoded = [tok.encode(s) for s in smi_list]
    tm = TransitionModel(vocab_size=vocab.size, token_list=vocab._tokens, alpha=0.5)
    tm.fit(encoded)

    # 4. Show top-5 replacements for a few tokens
    for tok_str in ["C", "N", "O", "="]:
        tid  = vocab.encode_token(tok_str)
        top5 = tm.top_k_replacements(tid, k=5)
        print(f"  Top-5 replacements for '{tok_str}': {top5}")

    # 5. Build plausibility checker
    allowed = set(CHEMISTRY_TOKENS)
    pc = PlausibilityChecker(allowed_tokens=allowed, min_length=4, max_repeat=3)

    # Test plausibility on some sequences
    test_cases = [
        (["C", "C", "O"],                     True),
        (["(", "C", "C"],                      False),  # unmatched (
        (["C", "1", "C", "C"],                 False),  # odd ring digit
        (["=", "C", "O"],                      False),  # starts with bond
        (["C", "C", "C", "C", "C", "O"],       True),
        (["C", "C", "C", "C", "="],            False),  # ends with bond
    ]
    print("\nPlausibility tests:")
    all_pass = True
    for tokens, expected in test_cases:
        result = pc.check(tokens)
        status = "✓" if result.is_valid == expected else "✗ FAIL"
        if result.is_valid != expected:
            all_pass = False
        print(f"  {status}  {tokens} → valid={result.is_valid}  ({result.reason})")
    print("All plausibility tests passed!" if all_pass else "Some tests FAILED!")

    # 6. Build a trivial predict_fn (no model needed for quick-test)
    def dummy_predict(smiles: str) -> float:
        """Dummy predictor: count heavy atoms as proxy for property."""
        return float(sum(1 for c in smiles if c.isupper())) * 0.15

    # 7. Run MCMC on a few sequences
    sampler = MCMCSampler(
        transition_model=tm,
        plausibility_checker=pc,
        tokenizer=tok,
        predict_fn=dummy_predict,
        beta=2.0,
        n_steps=30,
        max_mutations=3,
        seed=42,
    )

    test_smiles = smi_list[:5]
    print("\n── MCMC Attack Demo ───────────────────────────────────────────")
    results = sampler.attack_batch(test_smiles, verbose=True)
    for r in results:
        print(f"  orig={r.original_smiles:<20} → adv={r.adversarial_smiles:<20} "
              f"drift={r.drift:.3f}  mut={r.n_mutations}  valid={r.is_valid}")

    summary = summarise_attack_results(results, drift_threshold=0.1)
    print("\n[attack_engine.py] Quick-test complete.")
