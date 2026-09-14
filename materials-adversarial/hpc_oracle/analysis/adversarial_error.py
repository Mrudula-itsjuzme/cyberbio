import numpy as np
import pandas as pd
from typing import Dict

def compute_adversarial_error(predictions: np.ndarray, 
                              oracle_values: np.ndarray, 
                              original_values: np.ndarray) -> Dict[str, float]:
    """
    Computes the true adversarial error metrics once the oracle ground truth is available.
    
    Args:
        predictions: Model predictions on adversarial candidates (y_hat_adv)
        oracle_values: Ground truth values from HPC oracle (y_adv)
        original_values: Ground truth values of the original structures (y_src)
        
    Returns:
        Dict containing E_S (Standard Error) and E_G (Generalization Error)
    """
    
    # E_S: Does the model accurately predict the property of the modified structure?
    e_s = np.mean(np.abs(predictions - oracle_values))
    
    # E_G: Did the attack actually find a structure whose true property is far from the original?
    e_g = np.mean(np.abs(oracle_values - original_values))
    
    return {
        "E_S_mean": float(e_s),
        "E_G_mean": float(e_g)
    }

if __name__ == "__main__":
    print("Run this module after pipeline execution to calculate final E_S and E_G.")
