import pytest
import numpy as np
from hpc_oracle.analysis.adversarial_error import compute_adversarial_error

def test_adversarial_error_computation():
    predictions = np.array([1.5, 2.0, 2.5])
    oracle_values = np.array([1.0, 1.8, 2.6])
    original_values = np.array([0.5, 1.0, 1.5])
    
    # E_S = mean(|preds - oracle|) = mean([0.5, 0.2, 0.1]) = 0.8 / 3 = 0.2666...
    # E_G = mean(|oracle - original|) = mean([0.5, 0.8, 1.1]) = 2.4 / 3 = 0.8
    
    result = compute_adversarial_error(predictions, oracle_values, original_values)
    
    assert np.isclose(result["E_S_mean"], 0.266666666)
    assert np.isclose(result["E_G_mean"], 0.8)
