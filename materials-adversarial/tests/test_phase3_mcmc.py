"""Tests for Phase 3 components: MCMC attacks and Confound Mitigation."""

import numpy as np
from pytest import approx

from materials_adv.evaluation.metrics import (
    length_linear_regression,
    length_stratified_metrics,
)
from materials_adv.attacks.randomization import SmilesRandomizationAttack
from materials_adv.attacks.probabilistic import ProbabilisticMCMCAttack
from materials_adv.data.tokenizer import tokenize

def test_length_linear_regression():
    # Synthetic data where Target = 2 * length + 10
    train_len = np.array([5.0, 10.0, 15.0, 20.0])
    train_tgt = 2.0 * train_len + 10.0
    
    eval_len = np.array([7.0, 12.0, 17.0])
    eval_tgt = 2.0 * eval_len + 10.0
    
    res = length_linear_regression(train_len, train_tgt, eval_len, eval_tgt)
    
    assert res["slope"] == approx(2.0)
    assert res["intercept"] == approx(10.0)
    assert res["mae"] < 1e-5
    assert res["rmse"] < 1e-5

def test_length_stratified_metrics():
    y_true = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    y_pred = np.array([12, 18, 30, 40, 50, 60, 70, 80])
    lengths = np.array([10, 10, 20, 20, 30, 30, 40, 40])
    
    res = length_stratified_metrics(y_true, y_pred, lengths, n_bins=2)
    # n_bins=2 -> median split. Median length is 25.
    # Group 1: length <= 25 -> values (10,12), (20,18), (30,30), (40,40)
    # MAE of group 1: (|10-12| + |20-18| + 0 + 0) / 4 = 1.0
    
    # Group 2: length > 25 -> MAE 0.0
    assert len(res) > 0
    # There should be a bin containing the lower lengths with MAE ~1.0
    maes = [v["mae"] for v in res.values()]
    assert any(abs(m - 1.0) < 1e-5 for m in maes)

def test_smiles_randomization_attack():
    rng = np.random.default_rng(42)
    attack = SmilesRandomizationAttack(rng)
    
    # Simple molecule that has multiple valid SMILES strings (e.g. ethanol)
    # CCO vs OCC
    tokens = list(tokenize("CCO"))
    outcomes = attack.generate(tokens, n_variants=1)
    
    if outcomes:
        outcome = outcomes[0]
        assert outcome.original_representation == "CCO"
        assert outcome.adversarial_representation != "CCO"
        assert len(outcome.adversarial_representation) > 0

class DummyPredictor:
    def __init__(self, target_units="K"):
        self.target_units = target_units
        
    def predict(self, psmiles: list[str]) -> np.ndarray:
        # Predict length
        return np.array([float(len(p)) for p in psmiles])

def test_mcmc_attack():
    rng = np.random.default_rng(42)
    predictor = DummyPredictor()
    
    attack = ProbabilisticMCMCAttack(
        rng=rng,
        predictor=predictor,
        allowed_tokens=["C", "O", "N"],
        steps=5,  # small steps for testing
        temperature=100.0,
    )
    
    tokens = list(tokenize("CCC"))
    outcomes = attack.generate(tokens, n_variants=2)
    
    for outcome in outcomes:
        assert outcome.original_representation == "CCC"
        assert "mcmc_accepted" in outcome.params
        assert outcome.params["mcmc_accepted"] is True
        assert "sampling_probability" in outcome.params
        assert "chemical_constraint_score" in outcome.params
        assert "attack_score" in outcome.params

if __name__ == "__main__":
    test_length_linear_regression()
    test_length_stratified_metrics()
    test_smiles_randomization_attack()
    test_mcmc_attack()
    print("All tests passed!")
